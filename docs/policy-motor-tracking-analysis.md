# Policy-Motor Tracking & Jitter Analysis

Analysis of locomotion policy performance on the Berkeley Humanoid Lite robot, based on test runs using the `policy_biped_50hz` checkpoint.

---

## Table of Contents

- [Config Summary](#config-summary)
- [Test Runs Overview](#test-runs-overview)
- [Jitter Metrics Explained](#jitter-metrics-explained)
  - [Action Reversal Rate](#1-action-reversal-rate-40-61)
  - [Oscillation Frequency](#2-oscillation-frequency-7-15-hz)
  - [Motor Smoothing Ratio](#3-motor-smoothing-ratio-2-57x)
  - [Action Saturation](#4-action-saturation)
- [Motor Tracking Performance](#motor-tracking-performance)
  - [Effective Motor Lag](#effective-motor-lag-80-140ms)
  - [Step Response](#step-response)
  - [Command vs Measured Range](#command-vs-measured-range)
  - [Position Bias](#position-bias)
- [Lag-Induced Feedback Destabilization](#lag-induced-feedback-destabilization)
  - [The Overcorrection Cycle](#the-overcorrection-cycle)
  - [Evidence From Data](#evidence-from-data)
- [Control Loop Timing Breakdown](#control-loop-timing-breakdown)
  - [Single Loop Iteration](#single-loop-iteration-20ms-budget)
  - [Structural Lag in Logged Data](#structural-lag-in-logged-data)
  - [Python vs C Overhead](#python-vs-c-overhead)
- [Policy Observation Vector](#policy-observation-vector)
- [Mitigation Strategies](#mitigation-strategies)
- [Key Findings Summary](#key-findings-summary)

---

## Config Summary

From `configs/policy_biped_50hz.yaml`:

| Parameter | Value |
|-----------|-------|
| Policy rate | 50 Hz (policy_dt=0.02) |
| Action scale | 0.25 |
| Action limits | [-10000, 10000] (effectively none) |
| Joint KP | 20.0 (all joints) |
| Joint KD | 2.0 (all joints) |
| Effort limit | 5.0 Nm |
| Num observations | 45 |
| Default joint positions | [0, 0, -0.2, 0.4, -0.3, 0, 0, 0, -0.2, 0.4, -0.3, 0] |

Action pipeline:
```
raw_action (NN output) -> clipped (±10000, no-op) -> scaled (raw * 0.25 + defaults) -> sent to motor
```

---

## Test Runs Overview

### Runs Using policy_biped_50hz

| Run | Frames | Duration | Dry Run | Mean Reversal | Notes |
|-----|--------|----------|---------|---------------|-------|
| 001 | 90 | 1.8s | Yes | -- | Dry run |
| 002 | 148 | 3.0s | Yes | -- | Dry run |
| 003 | 185 | 3.7s | Yes | -- | Dry run |
| **004** | 48 | 1.0s | No | 43% | First live test, R_knee fixed |
| **015** | 404 | 8.1s | No | 65% | Highest jitter, 19.3Hz oscillation |
| **016** | 673 | 13.5s | No | 57% | Best gait emergence |
| **017** | 480 | 9.6s | No | 59% | |
| **018** | 257 | 5.1s | No | 48% | |
| **020** | 232 | 4.6s | No | 51% | |
| **021** | 757 | 15.1s | No | 49% | Longest run, smoothest |
| **022** | 261 | 5.2s | No | 55% | |

Runs 005-014 and 019 used `policy_biped_updated.yaml` (a different config).

### Consistent Patterns Across All Live Runs

- **Knees always saturated**: L_knee and R_knee have raw action magnitude >2 in 88-100% of frames
- **Deep squat start**: Knees consistently start +0.3 to +1.3 rad above the 0.4 rad default (OOD problem)
- **Run 015 is an outlier**: Median dominant frequency 19.3 Hz (near Nyquist limit)
- **Run 021 is the best**: Longest duration (15.1s), lowest mean reversal (49%)

---

## Jitter Metrics Explained

### 1. Action Reversal Rate (40-61%)

**What it measures**: How often the policy output changes direction from one step to the next.

**Computation**:
```
raw_action = [1.2, 1.5, 1.3, 1.6, 1.4, ...]
delta      = [+0.3, -0.2, +0.3, -0.2, ...]
sign       = [ +1,   -1,   +1,   -1,  ...]
reversal when sign flips: +1 -> -1 or -1 -> +1
reversal_rate = count(sign flips) / count(non-zero signs)
```

**Interpretation**:

| Rate | Meaning |
|------|---------|
| 0% | Monotonic — action moves smoothly in one direction |
| ~8% | Clean walking gait (1-2 Hz leg swing at 50 Hz rate = ~2 reversals per 25 steps) |
| 50% | Random noise — direction is a coin flip each step |
| >50% | Anti-correlated oscillation — actively zig-zagging |

**Our data shows 40-65%** — the policy oscillates nearly every step on many joints.

```
Clean gait (8% reversal):     /\    /\    /\       (smooth sine)
Our policy (55% reversal):    /\/\/\/\/\/\/\/\/\   (high-freq buzz on slow wave)
```

### 2. Oscillation Frequency (7-15 Hz)

**What it measures**: The strongest repeating frequency in the action signal via FFT.

**Interpretation**:

| Frequency | Meaning |
|-----------|---------|
| 1-3 Hz | Normal walking gait (~1.5-2 Hz step frequency) |
| 7-15 Hz | 5-10x faster than physical walking — pure jitter |
| 19+ Hz | Near Nyquist limit (25 Hz at 50 Hz sampling) — output flips nearly every frame |

Motors have mechanical bandwidth of ~5-10 Hz depending on load. Commands above this create vibration and heat but cannot be physically tracked.

### 3. Motor Smoothing Ratio (2-57x)

**What it measures**: How much the physical motor filters out commanded jitter.

```
smoothing_ratio = var(step-to-step Δ in COMMANDED position) / var(step-to-step Δ in MEASURED position)
```

**Interpretation**:

| Ratio | Meaning |
|-------|---------|
| 1.0x | Motor perfectly tracks commands |
| 5x | Motor absorbs 80% of jitter |
| 20x | Motor absorbs 95% of jitter |
| 57x | Motor absorbs 98% of jitter (R_knee in run 015) |

Even though the motor smooths the output, it **tries** to follow every command. Each jittery command creates a torque impulse, resulting in audible vibration, wasted energy, and mechanical stress.

### 4. Action Saturation

**What it measures**: How far the raw NN output strays from zero.

The action pipeline: `target = raw_action * 0.25 + default_position`

| Raw Magnitude | Displacement from Default | Interpretation |
|---------------|--------------------------|----------------|
| < 1 | ±0.25 rad (±14°) | Small correction |
| > 2 | ±0.50 rad (±29°) | Moderate displacement |
| > 5 | ±1.25 rad (±72°) | Large — near/beyond joint limits |
| > 10 | ±2.50 rad (±143°) | Physically impossible for most joints |

**Findings**: Knees at |raw|>2 in 88-100% of frames. L_knee mean raw action ~3 (target = 1.15 rad) with peaks of 6-10 (target = 1.9-2.9 rad). This connects directly to ankle stoppers breaking — extreme positions with no clamping.

---

## Motor Tracking Performance

### Effective Motor Lag: 80-140ms

RMSE between commanded and measured position is minimized at **lag 4-7 steps** in logged data. After correcting for the structural 1-step logging offset, this corresponds to **60-120ms of physical motor response time**.

```
Policy commands position at step t=0:
  t+1 (20ms):  motor barely started moving (~0% of change achieved)
  t+2 (40ms):  still barely moving
  t+3 (60ms):  starting to respond (~5-15% achieved)
  t+4 (80ms):  partial tracking (~15-30% achieved)
  t+5 (100ms): best joints reach ~20-50% of commanded change
```

By the time the motor meaningfully responds, the policy has issued 4-7 new commands on top.

### Step Response

When the policy commands a position change, here's how much the motor achieves:

| Time After Command | Fraction Achieved (median) |
|--------------------|---------------------------|
| +1 step (20ms) | ~0% |
| +2 steps (40ms) | ~0-5% |
| +3 steps (60ms) | ~5-15% |
| +4 steps (80ms) | ~15-30% |
| +5 steps (100ms) | ~20-50% |

The motor needs 5-10 policy steps to achieve even half of a commanded change.

### Command vs Measured Range

The policy commands far more motion than the motors deliver:

| Joint | Commanded Range | Measured Range | Motor Achieves |
|-------|----------------|----------------|----------------|
| L_knee | ~4.5 rad | ~1.3 rad | **30%** |
| R_knee | ~3.4 rad | ~1.0 rad | **30%** |
| L_ankle_pitch | ~3.1 rad | ~1.7 rad | **56%** |
| R_ankle_pitch | ~2.5 rad | ~1.2 rad | **48%** |
| L_hip_pitch | ~1.2 rad | ~0.9 rad | **75%** |
| R_hip_pitch | ~1.1 rad | ~0.8 rad | **73%** |

Hip pitch joints track best (~75%). Knees are worst (~30%).

### Position Bias

Some joints consistently sit offset from where they're commanded:

| Joint | Typical Bias (meas - cmd) | Likely Cause |
|-------|--------------------------|--------------|
| L_hip_roll | -0.10 to -0.21 rad | Calibration offset |
| R_hip_yaw | +0.06 to +0.23 rad | Calibration offset |
| L/R_ankle_pitch | -0.16 to -1.19 rad | Gravity load exceeds torque limit |

---

## Lag-Induced Feedback Destabilization

### The Overcorrection Cycle

Motor lag creates a feedback destabilization loop:

```
Step 0:  Policy commands knee to 0.8 rad, motor at 0.5 rad (error = +0.3)
Step 1:  Motor barely moved (0.51). Policy sees big error -> commands harder
Step 2:  Motor still catching up (0.55). Policy pushes harder still
Step 3:  Motor accelerating (0.65). Policy still pushing
Step 4:  Motor overshoots (0.85). Error flips sign -> policy REVERSES
Step 5:  Motor has momentum going up (0.90). Policy pushes down harder
Step 6:  Motor starts slowing... cycle repeats in opposite direction
```

### Evidence From Data

**1. Overcorrection Rate: 77-96%**

When the policy issues a large command, it reverses direction within 1-2 steps in 77-96% of cases:

| Run | Mean Overcorrection Rate |
|-----|--------------------------|
| test_run_015 | **96%** |
| test_run_017 | **90%** |
| test_run_016 | **88%** |
| test_run_021 | **77%** |

**2. Motor Responds at Lag 3-5 (60-100ms)**

Cross-correlation between command changes and measurement changes:
- lag 0-1: **Negative** (anti-correlated — motor hasn't responded)
- lag 3-5: **Peak positive** (motor finally tracking)

The policy issues 3-5 corrections before seeing ANY motor movement.

**3. Error Signal Oscillates at 1.8-2.9 Hz**

The tracking error (cmd - meas) has oscillatory autocorrelation, confirming the closed-loop system generates oscillations the policy never intended.

**4. Error Drives Commands (Negative Correlation)**

Tracking error at time t is negatively correlated (-0.2 to -0.7) with the next command change — confirming the policy reactively corrects errors the motor hasn't resolved yet.

### Why prev_actions Doesn't Help

The policy observation vector includes `prev_actions` (the previous step's raw output). However:

1. **Trained with instant motors** — the policy learned that prev_action gets executed immediately. On real hardware with 60-100ms lag, prev_action hasn't taken effect yet
2. **One step isn't enough** — the motor is responding to actions from 3-5 steps ago. The policy would need 3-5 steps of history to account for in-flight commands
3. The policy interprets prev_actions in the context of instant actuators, so it misreads the feedback and overcorrects

---

## Control Loop Timing Breakdown

### Single Loop Iteration (20ms budget)

```
+-- Step t --------------------------------------------------------+
|                                                                   |
|  1. actions = controller.update(obs)                              |
|     +-- Build observation vector (numpy)          ~0.1ms          |
|     +-- ONNX inference (policy.forward())         ~1-5ms          |
|     +-- Clip + scale actions (numpy)              ~0.1ms          |
|                                                                   |
|  2. Log frame data (dict + list append)           ~0.1ms          |
|                                                                   |
|  3. obs = robot.step(actions)                                     |
|     +-- Set joint_position_target = actions       ~0.0ms          |
|     +-- update_joints() -- 6 joint pairs:                         |
|     |   +-- transmit_pdo_2(L) -> CAN TX           ~0.1ms          |
|     |   +-- transmit_pdo_2(R) -> CAN TX           ~0.1ms          |
|     |   +-- receive_pdo_2(L)  -> CAN RX           ~0.1-1ms        |
|     |   +-- receive_pdo_2(R)  -> CAN RX           ~0.1-1ms        |
|     |   x 6 pairs                                 ~1-6ms total    |
|     +-- get_observations() (numpy)                ~0.1ms          |
|                                                                   |
|  4. udp.send_numpy(obs)                           ~0.1ms          |
|                                                                   |
|  5. rate.sleep() -- waits for remainder of 20ms                   |
|                                                                   |
|  Total active time:                               ~3-12ms         |
|  Sleep time:                                      ~8-17ms         |
+-------------------------------------------------------------------+
```

### Structural Lag in Logged Data

Logging happens BEFORE `robot.step(actions)`:

```python
actions = controller.update(obs)           # Compute from PREVIOUS obs
frame["policy_output"] = scaled_actions    # NEW command (not sent yet)
frame["robot_measured"] = joint_position   # OLD measurement (from previous step)
obs = robot.step(actions)                  # NOW send command, get new measurement
```

| Log Lag | Physical Time After Command Sent | Meaning |
|---------|----------------------------------|---------|
| lag=0 | -20ms (before cmd sent) | Structural artifact |
| lag=1 | ~0ms | Same step as transmit, motor hasn't moved |
| lag=2 | 20ms | Motor has had 1 step |
| lag=3 | 40ms | Motor has had 2 steps |
| lag=4 | 60ms | Motor has had 3 steps |
| lag=5 | 80ms | Motor has had 4 steps |

Subtract 1 from log lag to get actual physical response time.

### Python vs C Overhead

| Lag Source | Contribution | Fixable by C? |
|------------|-------------|---------------|
| Structural log offset | 1 step (20ms) | No — measurement artifact |
| Python overhead | ~0.5 steps (3-10ms) | Yes — saves ~2-8ms |
| Python timing jitter | occasional +5-10ms | Yes — RT scheduler eliminates |
| CAN bus communication | ~0.1-0.3 steps (2-6ms) | No — hardware speed |
| **Motor mechanical response** | **3-5 steps (60-100ms)** | **No — physics** |

Switching to C would save ~1 step (20ms) and eliminate GC-induced timing jitter. The dominant 60-100ms motor lag is mechanical (inertia + 15:1 gearbox) and cannot be fixed in software.

---

## Policy Observation Vector

45 dimensions total (`num_observations: 45`):

| Index | Content | Dimensions |
|-------|---------|------------|
| 0-2 | command_velocity (x, y, yaw) | 3 |
| 3-5 | base_angular_velocity | 3 |
| 6-8 | projected_gravity | 3 |
| 9-20 | joint_positions (minus defaults) | 12 |
| 21-32 | joint_velocities | 12 |
| 33-44 | prev_actions (raw, clipped) | 12 |

The policy sees its previous action, but only 1 step of history. With 3-5 steps of motor lag, this is insufficient for proper delay compensation.

---

## Mitigation Strategies

### Option A: Action Smoothing (no retraining needed)

Add exponential moving average filter on policy output:
```python
alpha = 0.3  # lower = smoother, higher = more responsive
smoothed = alpha * new_action + (1 - alpha) * previous_action
```
Limits how fast commanded position can change, preventing overcorrection buildup.

### Option B: Joint Position Clamping (no retraining needed)

Add safety clamps on scaled actions to prevent dangerous extremes:
```python
position_limits_lower = [-0.5, -0.5, -1.0, 0.0, -1.0, -0.5, ...]  # per joint
position_limits_upper = [+0.5, +0.5, +0.2, 1.5, +0.2, +0.5, ...]
scaled_actions = np.clip(scaled_actions, position_limits_lower, position_limits_upper)
```

### Option C: Add Motor Delay to Simulation Training

Train with realistic actuator dynamics:
- Add 60-100ms PD response lag in sim
- Policy learns not to overcorrect
- Standard sim-to-real approach

### Option D: Retrain at Lower Frequency

Train with `policy_dt=0.05` (20 Hz):
- Motor lag becomes 1-2 steps instead of 3-5
- Less overcorrection compounding
- Lower Nyquist limit (10 Hz) eliminates highest-frequency jitter
- **Cannot just change rate without retraining** — action magnitudes and timing are baked into NN weights

### Option E: Feedforward Velocity

Instead of `velocity_target=0`, compute expected velocity:
```python
velocity_target = (position_target - previous_position) / dt
bus.transmit_pdo_2(device_id, position_target, velocity_target)
```
Tells the motor where it should be going, reducing the correction burden on the PD controller.

### Option F: Tune KP/KD Ratio

| Setting | KP | KD | Behavior |
|---------|----|----|----------|
| Current | 20 | 2 | Responsive but oscillation-prone |
| More damped | 15 | 3 | Less overshoot, slower tracking |
| Conservative | 10 | 3 | Smooth but less accurate |

Higher damping (KD) reduces overshoot at the cost of tracking speed.

---

## Key Findings Summary

1. **Motor lag is 60-120ms (3-5 policy steps)** — the dominant source of latency, caused by mechanical inertia and 15:1 gearbox, not software

2. **Motor lag causes feedback destabilization** — the policy issues 3-5 unresolved corrections before seeing motor movement, leading to 77-96% overcorrection rates

3. **Policy jitter is severe** — 40-65% action reversal rates (vs ~8% expected for clean gait), with 7-15 Hz oscillation frequencies

4. **Motors heavily filter the jitter** — smoothing ratios of 2-57x, but residual vibration is still visible and wastes energy

5. **Knees and ankles are worst affected** — raw actions regularly saturated (|raw|>2 in 88-100% of frames for knees), commanded range 2-3x beyond what motors achieve

6. **Hip pitch joints work best** — 73-75% tracking, cleanest gait signal, lowest jitter

7. **Out-of-distribution start position** compounds the problem — knees at ~1.5 rad vs 0.4 default puts the policy in OOD territory from frame 1

8. **prev_actions in observation is insufficient** — only 1 step of history vs 3-5 steps of motor lag; policy was trained with instant actuators

9. **Python vs C is minor** — ~1 step (20ms) overhead; switching to C helps timing consistency but does not fix the 60-100ms mechanical lag

10. **Running at 20 Hz without retraining would make things worse** — same aggressive corrections applied for 2.5x longer
