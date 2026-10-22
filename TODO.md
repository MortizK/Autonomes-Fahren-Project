# TODO

## Lane Detection

- [x] Edge Detection
- [x] Deciding how to represent the two lanes 
  - [x] List of Points?
  - [ ] Interpret Points as mathematical Curves?
- [x] Transform Edge Detection into two lanes
- [ ] Edge Cases for Car not on Track
- [ ] Edge Cases for multiple Starting Points
- [x] Performance Analysis (Real Time?, how many ms?)

## Path Planning

- [x] Implement a really simple centerline
- [ ] Find out the characteristics of the car
- [ ] Find a Path that minimizes a function (time)
- [ ] Discuss if we are allowed to use the first frame global view
  - [ ] Do we want to use it and how?
  - [ ] Rally Notation? ([Pace Notes](https://en.wikipedia.org/wiki/Pacenotes))
- [ ] Performance Analysis (Real Time?, how many ms?)

## Lateral Control

- [x] Transform the trajectory from pixels into a metric vehicle frame (x and y are scaled differently)
- [x] Use the front axle as reference point (Stanley)
- [x] Find the closest point on the trajectory (projection onto the segments, not only the points)
- [x] Compute cross track error and heading error
- [x] Stanley control law with default parameters
- [x] Map the steering angle onto the action (target wheel angle in rad, steering lock +/-0.4 rad)
- [x] Damping similar to the D-term of a PID controller
- [x] Edge cases: no or too few trajectory points, standstill, gaps in the trajectory
- [x] Edge case: camera zoom during the first second
- [x] Tune the parameters (`benchmark_control.py`, `test_lateral_control.py`)
- [x] Optional: Pure Pursuit as alternative (selectable via parameter) and compare both
- [ ] Check the test with `--domain_randomize` once the lane detection supports it
- [x] Performance Analysis (Real Time?, how many ms?)

## Longitudinal Control

- [x] Find out the braking and cornering limits of the car
- [x] Compute the curvature of the trajectory
- [x] Target speed from the curvature (lateral acceleration limit)
- [x] Brake early enough before curves (braking distance)
- [x] PID controller with separate gas and brake output
- [x] Anti-windup and D-term on the measured speed
- [x] Reduce gas while steering hard (combined tire friction)
- [x] Keep the brake below 0.9, otherwise the wheels lock
- [x] Align the interface with `car.py` and `test_longitudinal_control.py`
- [x] Tune the parameters
- [ ] Use the curvature from `PathPlanning` once it plans a racing line
- [x] Performance Analysis (Real Time?, how many ms?)

## Pipeline

- [ ]
- [ ] Performance Analysis (Real Time?, how many ms?)