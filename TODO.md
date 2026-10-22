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

- [ ] Transform the trajectory from pixels into a metric vehicle frame (x and y are scaled differently)
- [ ] Use the front axle as reference point (Stanley)
- [ ] Find the closest point on the trajectory (projection onto the segments, not only the points)
- [ ] Compute cross track error and heading error
- [ ] Stanley control law with default parameters
- [ ] Map the steering angle onto the action (target wheel angle in rad, steering lock +/-0.4 rad)
- [ ] Damping similar to the D-term of a PID controller
- [ ] Edge cases: no or too few trajectory points, standstill, gaps in the trajectory
- [ ] Tune the parameters with `test_lateral_control.py`
- [ ] Optional: Pure Pursuit as alternative (selectable via parameter) and compare both
- [ ] Performance Analysis (Real Time?, how many ms?)

## Longitudinal Control

- [ ] Find out the braking and cornering limits of the car
- [ ] Compute the curvature of the trajectory
- [ ] Target speed from the curvature (lateral acceleration limit)
- [ ] Brake early enough before curves (braking distance)
- [ ] PID controller with separate gas and brake output
- [ ] Anti-windup and filtered D-term
- [ ] Reduce gas and brake while steering hard (combined tire friction)
- [ ] Keep the brake below 0.9, otherwise the wheels lock
- [ ] Align the interface with `car.py` and `test_longitudinal_control.py`
- [ ] Tune the parameters
- [ ] Performance Analysis (Real Time?, how many ms?)

## Pipeline

- [ ]
- [ ] Performance Analysis (Real Time?, how many ms?)