# Findings

## The Car

The car has one rigid hull and four independent wheel bodies connected with
revolute joints.

- Front wheels are used for steering.
- Rear wheels receive engine torque.
- All four wheels receive braking.
- Steering is limited by the wheel joint to approximately +/-0.4 rad.
- The environment negates the continuous steering action before applying it:
  `car.steer(-action[0])`. The action sign must therefore be tested against the
  observed turn direction instead of assumed.
- Gas and brake are clipped to the range 0..1 by the car model.
- Gas is applied gradually: the wheel gas value cannot increase by more than
  0.1 per physics step, but it can drop immediately.
- Brake values of 0.9 or more lock the wheel angular velocity to zero.

## The Wheel

Each Wheel has:

- `omega`: wheel angular velocity in rad/s.
- `tire speed`: `omega * wheel_rad`; this is the rotational surface speed.
- `wheel speed`: world velocity resolved into wheel-forward and wheel-side
  components.
- `grip`: road friction is normally 1.0; grass friction is 0.6.
- `tiles`: the set of road tiles currently touching the wheel. This is the
  reliable road/grass contact signal.
- `gas`, `brake`, and `steer`: the currently applied wheel controls, which may
  differ from the action because gas ramps up and steering uses a joint motor.
- `phase`: accumulated visual wheel rotation.
- `skid_start` and `skid_particle`: skid-mark state, useful as a qualitative
  slip indicator.

The Grid is 0.6 on the Curb and still 1.0 when slightly of track.

The longitudinal slip estimate is:

`longitudinal_slip = tire_speed - forward_speed`

The lateral slip estimate is the wheel-side speed. Large absolute longitudinal
or lateral slip means the tire is using its friction budget and may be
skidding.

The tire force calculation is:

1. `f_force = -forward_speed + tire_speed`
2. `p_force = -side_speed`
3. Combine both components into `force = sqrt(f_force^2 + p_force^2)`.
4. If the force exceeds the friction limit, normalize both components and
   saturate their combined magnitude at the friction limit.
5. Apply the resulting force to the wheel and update `omega`.

This means longitudinal acceleration, braking, and lateral cornering compete
for one combined friction limit. A tire cannot provide full braking and full
cornering force simultaneously.

The physics model also tracks fuel usage, but the environment resets the
per-step fuel counter after calculating the reward. Fuel is therefore mainly a
diagnostic value, not a useful optimization target in the default environment.

## Actions and Timing

The default environment is continuous:

- Action shape: `[steering, gas, brake]`.
- Steering range: -1..1.
- Gas range: 0..1.
- Brake range: 0..1.
- The action space is `Box([-1, 0, 0], 1, shape=(3,))`.
- Physics runs at 50 Hz (`dt = 1/50` seconds).
- The controller receives one observation and chooses one action per
  environment step.
- The world solver uses 6 velocity iterations and 2 position iterations per
  physics step.

The controller should not command gas and heavy braking at the same time unless
that behavior is intentional. It should also rate-limit steering and consider
the delay caused by the front steering joint motor.

## Observation and Coordinates

The default observation is a `96 x 96 x 3` uint8 RGB image with values 0..255.
It is a camera image, not a metric world-state vector.

The wrapper adds these values to `info`:

- `speed`: magnitude of the hull linear velocity.
- `trajectory`: ground-truth centerline points projected into the state image.
- `left_lane_boundary` and `right_lane_boundary`: ground-truth lane
  boundaries projected into the state image.

The lane boundaries and trajectory are currently privileged information:
they are calculated from the simulator road geometry, not inferred from the
camera image. They are useful for debugging and benchmarking, but an
autonomous system intended to operate from vision alone must not use them.

The projected coordinates use image coordinates: x increases to the right and
y increases downwards. The wrapper filters points to approximately
`0 < x < 96` and `0 < y < 84`, so the bottom 12 image rows are intentionally
excluded from the ground-truth geometry.

The wrapper reports scalar speed as:

`speed = sqrt(vx^2 + vy^2)`

This is world-frame speed. For control, also use velocity resolved in the
vehicle or wheel frame so that forward speed is not confused with lateral
sliding speed.

## The Simulation

The environment is `CarRacing-v3` with continuous controls, 96x96 RGB
observations, a default lap-completion threshold of 95%, and a 1000-step
episode time limit.

Track and reward behavior:

- Each road tile gives positive reward only the first time it is visited.
- The total positive track reward is approximately 1000 for a complete track,
  distributed as `1000 / number_of_track_tiles`.
- Every normal environment step costs 0.1 reward.
- Leaving the playfield at `abs(x) > 333.33` or `abs(y) > 333.33` terminates
  the episode and gives a -100 reward for that step.
- Completing at least 95% of the track terminates the episode and sets
  `info["lap_finished"] = True`.
- Leaving the playfield terminates the episode and sets
  `info["lap_finished"] = False`.
- The 1000-step time limit is truncation, not a physical termination. A
  controller should distinguish `terminated` from `truncated`.
- `info` is otherwise empty in the base environment; the additional wrapper
  values are computed after reset and after every step.

The reward is therefore primarily a progress signal, not a smooth measure of
driving quality. A controller should additionally monitor speed, road contact,
slip, heading error, lateral error, curvature, and recovery distance.

## Track Geometry and Planning

The track is generated procedurally from a seed. The seed is part of the
experiment configuration and must be recorded for reproducible comparisons.
Different seeds can produce different curvature, widths, and difficulty.

Important planning quantities include:

- Distance from the vehicle center to the track centerline.
- Heading error relative to the local centerline tangent.
- Upcoming curvature and curvature rate.
- Available road width and distance to both boundaries.
- Target speed derived from curvature and available grip.
- Braking distance, approximately `v^2 / (2 * available_deceleration)`,
  increased for grass, low grip, and large lateral demand.

A good target speed must account for combined friction. Reducing speed before
the corner is safer than braking while already using most lateral grip.

### Difficult Seeds

| Seed | Score | Notes |
| ---- | ----- | ----- |
| 197453 | - | S-Turn and some U-Turns |
| 180888 | 158.34 | Sharp U-Turn after long straight |
| 373468 | 136.72 | Sharp U-Turn after long straight |
| 693592 | 073.01 | Sharp U-Turn after long straight |

Also, maybe we can Use the First Frame of the Simulation to estimate a path planning. As we can expect some Turns from the global view.

## Control and State Estimation

An autonomous controller should maintain at least:

- Current forward and lateral velocity.
- Heading and heading error.
- Lateral distance from the planned path.
- Steering command and steering rate.
- Gas, brake, and their actual wheel values.
- Per-wheel longitudinal and lateral slip.
- Road contact and grip for every wheel.
- Current curvature and target speed.
- Whether the vehicle is recovering from grass or a spin.

Useful derived signals:

- `slip_ratio = (tire_speed - forward_speed) / max(abs(forward_speed), epsilon)`
- `slip_angle` from lateral speed and forward speed, using
  `atan2(lateral_speed, abs(forward_speed) + epsilon)`
- `friction_usage` from the ratio of combined longitudinal and lateral demand
  to the available tire friction limit
- `stopping_distance = speed^2 / (2 * deceleration)`

All derived signals need low-speed guards and light filtering. At very low
speed, slip ratios and slip angles are numerically unstable.

## Project-Specific Findings

- `Car.next_action` currently returns a constant action before the lane
  detection, path planning, and control code below it. The autonomous pipeline
  is therefore unreachable until that early return is removed.
- `main.py` and `test_pipeline.py` use different execution paths. The test
  pipeline runs lane detection, planning, and control directly, while
  `main.py` currently uses `Car.next_action`.
- The wrapper's ground-truth trajectory and boundaries are valuable for
  validation, but should be kept separate from a vision-only production
  controller to avoid accidental simulator-state leakage.
- A reset must clear controller history such as filtered velocities, previous
  errors, integral terms, target-speed history, and recovery state.
- Evaluation should record seed, lap completion, total reward, distance
  completed, average speed, maximum slip, off-road time, and termination reason.

## Camera Geometry

- The 1000x800 window is scaled down to 96x96, so the image is stretched
  differently: 1.56 px per unit across and 1.94 px per unit along the car
  after the zoom. Angles and curvatures measured directly in pixels are
  distorted, the controllers convert to units first (`geometry.py`).
- The car origin is at pixel (48, 72), the front axle 1.6 units ahead and the
  rear axle 1.64 units behind (wheelbase 3.24).
- The camera zooms from 0.6 to 16.2 during the first second. The observation
  after `reset` is already rendered at t = 0.02 s, full zoom is reached at
  frame 49. Paths from the first frames have the wrong scale.

## Measured Car Limits

Measured on asphalt with short test runs (full gas, or constant brake from
70 units/s):

| Quantity | Value | Note |
| -------- | ----- | ---- |
| Acceleration at full gas | ~44 units/s^2 | rear wheels spin, less above ~60 units/s |
| Deceleration per brake value | ~300 * brake units/s^2 | linear up to ~0.7 |
| Deceleration with locked wheels | 219 units/s^2 | brake >= 0.9, no steering |
| Tire friction limit | 4 * 400 / 7.3 = 219 units/s^2 | same limit for cornering |

- The steering action is the target wheel angle in rad (with the sign
  flipped by the env). The joint stops at +/-0.4 rad, so `0.2` means 0.2 rad
  and not 50 % of the steering lock.
- Each wheel has a moment of inertia of 1.6, i.e. an effective mass of
  `I / r^2 = 5.5` per wheel. The free rolling front wheels therefore brake the
  car while accelerating, that is why full gas only gives 44 instead of
  110 units/s^2.
- Brake values above ~0.72 decelerate the wheel faster than the car and lock
  it after some steps, even below 0.9.
- Full gas in a curve saturates the rear tires and the car spins. Gas has to
  be reduced by the lateral acceleration (friction circle).
