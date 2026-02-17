sh scripts/start_can_transports.sh

ip -details link show can0

python scripts/check_connection.py

python scripts/robot/write_control_params_only.py

python scripts/motor/read_all_configurations.py

python scripts/test_imu.py

jstest /dev/input/js0

python scripts/calibrate_standing.py

python scripts/run_idle.py

python scripts/show_joint_positions.py

python scripts/run_locomotion.py  --dry-run --log test_run_004.json --config ../../configs/policy_biped_50hz.yaml

python scripts/run_locomotion.py --max-steps 50 --log test_run_005.json --config ../../configs/policy_biped_50hz.yaml

python scripts/run_locomotion.py --log test_run_004.json --config ../../configs/policy_biped_50hz.yaml

python scripts/run_locomotion.py --skip-init-position --max-steps 40 --log test_run_007.json --config ../../configs/policy_biped_updated.yaml

python scripts/run_locomotion.py --skip-init-position --log test_run_022.json --config ../../configs/policy_biped_50hz.yaml

A+L = INIT
A+R = RUN
Y = IDLE

mjpython ./scripts/sim2sim/play_mujoco_from_log.py --log logs/test_run_004.json --config ./configs/policy_biped_updated.yaml
