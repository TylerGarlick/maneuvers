# Maneuvers catalog — synthetic signal intuition ✅

This file lists the maneuver names used in the demo dataset and a short note
on how each maneuver is represented in the synthetic signals (accel, gyro).
These descriptions are intentionally terse and focused on which signal
components carry the strongest signature for the maneuver.

| Name | Signal intuition |
|------|------------------|
| left_roll | Roll-rate spike (gyro x) + lateral accel (ay) negative |
| right_roll | Roll-rate spike (gyro x) + lateral accel (ay) positive |
| left_bank | Sustained lateral accel left (ay) |
| right_bank | Sustained lateral accel right (ay) |
| pitch_up | Positive pitch-rate (gyro y) and negative z-accel |
| pitch_down | Negative pitch-rate (gyro y) and positive z-accel |
| climb | Gradual positive z-accel and small pitch gyro |
| descent | Negative z-accel and small pitch gyro |
| loop | Large cyclic pitch gyro + vertical accel cycle |
| barrel_roll | Roll oscillation combined with some yaw gyro |
| aileron_roll | Short, sharp roll (gyro x) |
| spiral_climb | Combined roll+yaw with increasing pitch/vertical accel |
| spiral_descent | Combined roll+yaw with decreasing vertical accel |
| corkscrew | High-frequency roll + yaw modulations |
| s_turn_left | Two-phase lateral acceleration (S-shape) |
| s_turn_right | Two-phase lateral acceleration (S-shape) |
| yaw_left | Yaw-rate spike (gyro z) |
| yaw_right | Yaw-rate spike (gyro z) |
| stall | Noisy high-angular-rate pattern and slow changes in accel |
| spin | Sustained high yaw/roll oscillation (gyro) |
| hammerhead | Strong pitch maneuver with yaw component |
| immelmann | Pitch up plus roll/yaw signature |
| split_s | Pitch down with yaw/roll coupling |
| pull_up | Strong positive z-accel and pitch gyro |
| push_over | Negative z-accel with pitch gyro reversal |
| sideslip | Lateral accel with small yaw/gyro |
| skid | Short yaw + lateral accel spike |
| surge_forward | Forward accel spike (ax) |
| heave_up | Vertical accel (az) spike |
| none | Background / no maneuver |

> These are deliberately simple, hand-crafted approximations that capture the
qualitative signatures in accelerometer and gyroscope channels. Use them as a
starting point and refine for any real dataset or collection of flight logs.
