#!/usr/bin/env python3
"""
Comprehensive side-by-side comparison of two jitter measurement logs.

Computes per-joint position, velocity, jerk, and FFT statistics,
plus IMU stats, and highlights which joints got worse/better
between standing and squatting poses.
"""

import json
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
LOG_DIR = Path("/home/gaurav/code/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/logs")
FILE_STANDING = LOG_DIR / "jitter_standing.json"
FILE_SQUATTING = LOG_DIR / "jitter_squating.json"

RAD2DEG = 180.0 / np.pi


def load_log(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "metadata": d["metadata"],
        "target": np.array(d["target_positions"]),           # (12,)
        "t": np.array(d["timestamps"]),                      # (N,)
        "pos": np.array(d["positions"]),                     # (N, 12)
        "vel": np.array(d["velocities"]),                    # (N, 12)
        "imu_quat": np.array(d["imu_quaternions"]),          # (N, 4)
        "imu_angvel": np.array(d["imu_angular_velocities"]), # (N, 3)
    }


def compute_joint_stats(log):
    """Return a dict of per-joint stats arrays (length 12)."""
    pos = log["pos"]           # (N, 12) in radians
    vel = log["vel"]           # (N, 12) in rad/s
    target = log["target"]     # (12,)   in radians
    t = log["t"]               # (N,)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    N = pos.shape[0]
    n_joints = pos.shape[1]

    # ── Position stats (converted to degrees) ──────────────────────
    pos_mean_deg = np.mean(pos, axis=0) * RAD2DEG
    pos_std_deg = np.std(pos, axis=0) * RAD2DEG
    pos_ptp_deg = np.ptp(pos, axis=0) * RAD2DEG
    pos_mean_err_deg = np.mean(pos - target[None, :], axis=0) * RAD2DEG

    # ── Velocity stats (rad/s) ─────────────────────────────────────
    vel_rms = np.sqrt(np.mean(vel**2, axis=0))
    vel_std = np.std(vel, axis=0)
    vel_maxabs = np.max(np.abs(vel), axis=0)

    # ── Jerk = d(velocity)/dt  (rad/s^2) ──────────────────────────
    dvel = np.diff(vel, axis=0) / dt  # (N-1, 12) acceleration
    jerk_rms = np.sqrt(np.mean(dvel**2, axis=0))

    # ── FFT: dominant non-DC frequency per joint ───────────────────
    freqs = np.fft.rfftfreq(N, d=dt)
    dom_freq = np.zeros(n_joints)
    dom_amp = np.zeros(n_joints)
    for j in range(n_joints):
        sig = pos[:, j] - np.mean(pos[:, j])  # remove DC
        spectrum = np.abs(np.fft.rfft(sig)) / N
        # skip bin 0 (DC)
        spectrum[0] = 0.0
        idx = np.argmax(spectrum)
        dom_freq[j] = freqs[idx]
        dom_amp[j] = spectrum[idx] * RAD2DEG  # amplitude in degrees

    return {
        "pos_mean_deg": pos_mean_deg,
        "pos_std_deg": pos_std_deg,
        "pos_ptp_deg": pos_ptp_deg,
        "pos_mean_err_deg": pos_mean_err_deg,
        "vel_rms": vel_rms,
        "vel_std": vel_std,
        "vel_maxabs": vel_maxabs,
        "jerk_rms": jerk_rms,
        "fft_freq": dom_freq,
        "fft_amp_deg": dom_amp,
        "dt": dt,
        "fs": fs,
    }


def compute_imu_stats(log):
    """Quaternion std per component, angular-velocity RMS per axis."""
    q = log["imu_quat"]       # (N, 4)
    w = log["imu_angvel"]     # (N, 3)

    quat_mean = np.mean(q, axis=0)
    quat_std = np.std(q, axis=0)
    angvel_rms = np.sqrt(np.mean(w**2, axis=0))
    angvel_maxabs = np.max(np.abs(w), axis=0)

    return {
        "quat_mean": quat_mean,
        "quat_std": quat_std,
        "angvel_rms": angvel_rms,
        "angvel_maxabs": angvel_maxabs,
    }


# ── Pretty-printing helpers ───────────────────────────────────────────
SEP = "=" * 140
THIN = "-" * 140

JOINT_NAMES = [
    "L_hip_roll", "L_hip_yaw", "L_hip_pitch", "L_knee",
    "L_ankle_pitch", "L_ankle_roll",
    "R_hip_roll", "R_hip_yaw", "R_hip_pitch", "R_knee",
    "R_ankle_pitch", "R_ankle_roll",
]

SHORT_NAMES = [
    "L_hRoll", "L_hYaw", "L_hPit", "L_knee",
    "L_aPit", "L_aRoll",
    "R_hRoll", "R_hYaw", "R_hPit", "R_knee",
    "R_aPit", "R_aRoll",
]


def delta_arrow(val_stand, val_squat, lower_is_better=True):
    """Return a small indicator: better / worse / ~same."""
    if val_stand == 0:
        return "  "
    ratio = val_squat / max(val_stand, 1e-12)
    if ratio > 1.15:
        return "WORSE" if lower_is_better else "BETTER"
    elif ratio < 0.85:
        return "BETTER" if lower_is_better else "WORSE"
    else:
        return "~same"


def print_section(title, metric_name, unit, stand_vals, squat_vals, lower_is_better=True, fmt=".4f"):
    print(f"\n  {title}  ({unit})")
    print(f"  {'Joint':<14s}  {'Standing':>12s}  {'Squatting':>12s}  {'Delta':>12s}  {'Verdict':>8s}")
    print(f"  {THIN[:70]}")
    for j in range(12):
        sv = stand_vals[j]
        qv = squat_vals[j]
        delta = qv - sv
        verdict = delta_arrow(sv, qv, lower_is_better)
        print(f"  {SHORT_NAMES[j]:<14s}  {sv:>12{fmt}}  {qv:>12{fmt}}  {delta:>+12{fmt}}  {verdict:>8s}")


def main():
    # Load data
    log_stand = load_log(FILE_STANDING)
    log_squat = load_log(FILE_SQUATTING)

    s_stand = compute_joint_stats(log_stand)
    s_squat = compute_joint_stats(log_squat)

    imu_stand = compute_imu_stats(log_stand)
    imu_squat = compute_imu_stats(log_squat)

    # ── Header ─────────────────────────────────────────────────────
    print(SEP)
    print("  JITTER COMPARISON:  Standing  vs  Squatting")
    print(SEP)
    m1 = log_stand["metadata"]
    m2 = log_squat["metadata"]
    print(f"  Standing log  : {FILE_STANDING.name}  ({m1['n_samples']} samples, {m1['duration']}s @ {m1['rate_hz']} Hz)")
    print(f"  Squatting log : {FILE_SQUATTING.name}  ({m2['n_samples']} samples, {m2['duration']}s @ {m2['rate_hz']} Hz)")
    print(f"  Effective dt  : standing={s_stand['dt']*1000:.2f} ms, squatting={s_squat['dt']*1000:.2f} ms")
    print()

    # ── Target positions ───────────────────────────────────────────
    print(THIN)
    print("  TARGET POSITIONS (deg)")
    print(f"  {'Joint':<14s}  {'Standing':>12s}  {'Squatting':>12s}")
    print(f"  {THIN[:42]}")
    for j in range(12):
        print(f"  {SHORT_NAMES[j]:<14s}  {log_stand['target'][j]*RAD2DEG:>12.3f}  {log_squat['target'][j]*RAD2DEG:>12.3f}")

    # ── Position ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("  POSITION STATISTICS")
    print(SEP)
    print_section("Position Std Dev", "pos_std", "deg",
                  s_stand["pos_std_deg"], s_squat["pos_std_deg"])
    print_section("Position Peak-to-Peak", "pos_ptp", "deg",
                  s_stand["pos_ptp_deg"], s_squat["pos_ptp_deg"])
    print_section("Mean Error from Target", "pos_err", "deg",
                  np.abs(s_stand["pos_mean_err_deg"]), np.abs(s_squat["pos_mean_err_deg"]))

    # ── Velocity ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("  VELOCITY STATISTICS")
    print(SEP)
    print_section("Velocity RMS", "vel_rms", "rad/s",
                  s_stand["vel_rms"], s_squat["vel_rms"])
    print_section("Velocity Std Dev", "vel_std", "rad/s",
                  s_stand["vel_std"], s_squat["vel_std"])
    print_section("Velocity Max |v|", "vel_max", "rad/s",
                  s_stand["vel_maxabs"], s_squat["vel_maxabs"])

    # ── Jerk ───────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  JERK (dv/dt) STATISTICS")
    print(SEP)
    print_section("Jerk RMS (accel RMS)", "jerk_rms", "rad/s^2",
                  s_stand["jerk_rms"], s_squat["jerk_rms"], fmt=".3f")

    # ── FFT ────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  FFT: DOMINANT NON-DC FREQUENCY")
    print(SEP)
    print(f"\n  {'Joint':<14s}  {'Stand Freq':>11s}  {'Stand Amp':>11s}  {'Squat Freq':>11s}  {'Squat Amp':>11s}")
    print(f"  {'':14s}  {'(Hz)':>11s}  {'(deg)':>11s}  {'(Hz)':>11s}  {'(deg)':>11s}")
    print(f"  {THIN[:64]}")
    for j in range(12):
        print(f"  {SHORT_NAMES[j]:<14s}"
              f"  {s_stand['fft_freq'][j]:>11.3f}  {s_stand['fft_amp_deg'][j]:>11.5f}"
              f"  {s_squat['fft_freq'][j]:>11.3f}  {s_squat['fft_amp_deg'][j]:>11.5f}")

    # ── IMU ────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  IMU STATISTICS")
    print(SEP)

    quat_labels = ["w", "x", "y", "z"]
    print(f"\n  Quaternion Std Dev")
    print(f"  {'Comp':<6s}  {'Standing':>12s}  {'Squatting':>12s}  {'Delta':>12s}  {'Verdict':>8s}")
    print(f"  {THIN[:56]}")
    for i in range(4):
        sv = imu_stand["quat_std"][i]
        qv = imu_squat["quat_std"][i]
        delta = qv - sv
        verdict = delta_arrow(sv, qv, lower_is_better=True)
        print(f"  {quat_labels[i]:<6s}  {sv:>12.6f}  {qv:>12.6f}  {delta:>+12.6f}  {verdict:>8s}")

    angvel_labels = ["roll", "pitch", "yaw"]
    print(f"\n  Angular Velocity RMS (rad/s)")
    print(f"  {'Axis':<6s}  {'Standing':>12s}  {'Squatting':>12s}  {'Delta':>12s}  {'Verdict':>8s}")
    print(f"  {THIN[:56]}")
    for i in range(3):
        sv = imu_stand["angvel_rms"][i]
        qv = imu_squat["angvel_rms"][i]
        delta = qv - sv
        verdict = delta_arrow(sv, qv, lower_is_better=True)
        print(f"  {angvel_labels[i]:<6s}  {sv:>12.6f}  {qv:>12.6f}  {delta:>+12.6f}  {verdict:>8s}")

    print(f"\n  Angular Velocity Max |w| (rad/s)")
    print(f"  {'Axis':<6s}  {'Standing':>12s}  {'Squatting':>12s}  {'Delta':>12s}  {'Verdict':>8s}")
    print(f"  {THIN[:56]}")
    for i in range(3):
        sv = imu_stand["angvel_maxabs"][i]
        qv = imu_squat["angvel_maxabs"][i]
        delta = qv - sv
        verdict = delta_arrow(sv, qv, lower_is_better=True)
        print(f"  {angvel_labels[i]:<6s}  {sv:>12.6f}  {qv:>12.6f}  {delta:>+12.6f}  {verdict:>8s}")

    # ── Summary: rank joints by "how much worse" squatting is ──────
    print()
    print(SEP)
    print("  SUMMARY: JOINTS RANKED BY CHANGE IN POSITION STD-DEV  (squatting vs standing)")
    print(SEP)
    diffs = s_squat["pos_std_deg"] - s_stand["pos_std_deg"]
    ratios = s_squat["pos_std_deg"] / np.maximum(s_stand["pos_std_deg"], 1e-12)
    order = np.argsort(diffs)[::-1]  # most increased first

    print(f"\n  {'Joint':<14s}  {'Stand std':>11s}  {'Squat std':>11s}  {'Abs Delta':>11s}  {'Ratio':>7s}  {'Verdict':>8s}")
    print(f"  {'':14s}  {'(deg)':>11s}  {'(deg)':>11s}  {'(deg)':>11s}  {'':>7s}  {'':>8s}")
    print(f"  {THIN[:72]}")
    for j in order:
        verdict = delta_arrow(s_stand["pos_std_deg"][j], s_squat["pos_std_deg"][j])
        print(f"  {SHORT_NAMES[j]:<14s}"
              f"  {s_stand['pos_std_deg'][j]:>11.5f}"
              f"  {s_squat['pos_std_deg'][j]:>11.5f}"
              f"  {diffs[j]:>+11.5f}"
              f"  {ratios[j]:>7.2f}x"
              f"  {verdict:>8s}")

    # ── Overall aggregate ──────────────────────────────────────────
    print()
    print(THIN)
    print("  AGGREGATE METRICS (mean across all 12 joints)")
    print(THIN)
    agg_metrics = [
        ("Pos Std (deg)", "pos_std_deg"),
        ("Pos PtP (deg)", "pos_ptp_deg"),
        ("Vel RMS (rad/s)", "vel_rms"),
        ("Vel Max (rad/s)", "vel_maxabs"),
        ("Jerk RMS (rad/s^2)", "jerk_rms"),
    ]
    print(f"  {'Metric':<22s}  {'Standing':>12s}  {'Squatting':>12s}  {'Ratio':>8s}")
    print(f"  {THIN[:58]}")
    for label, key in agg_metrics:
        sv = np.mean(s_stand[key])
        qv = np.mean(s_squat[key])
        ratio = qv / max(sv, 1e-12)
        print(f"  {label:<22s}  {sv:>12.5f}  {qv:>12.5f}  {ratio:>7.2f}x")

    print()
    print(SEP)
    print("  Analysis complete.")
    print(SEP)


if __name__ == "__main__":
    main()
