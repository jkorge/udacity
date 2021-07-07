# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

## Notes
* Model
  * The `MPC` class member method `Solve` implements the predictive control solution
  * This method takes in an initial state vector describing the vehicle and a coefficient vector describing a polynomial fitted to a set of known waypoints
  * A set of constraints are defined (based on the simulation at hand) before instantiating the `FG_eval` class and calling upon the Ipopt solver to produce an optimum trajectory and accompanying controls
  * `FG_eval` class member method `operator` uses the same coeffients as `MPC::Solve` to compute a reference trajectory
  * `FG_eval::operator` further defines the constraint equations which model the dynamics of the vehicle's motion
  * `FG_eval::operator` also computes the error sustained in the vehicle's motion. The total error is a linear combination of the residual squared errors of the vehicles position, orientation, and controls.
    * The amplitudes of the terms in this sum were determined heuristically, incrementing by an order of magnitude until appropriate ranges were found for each. Some fine-tuning was implemented thereafter
    * Examples of some of the error-term amplitude trials are found in the `Videos/Error Weights` directory
* Vehicle State & Actuators
  * The vehicle's state at any given time is described by a set of 6 parameters:
    1. X-coordinate
    2. Y-coordinate
    3. Orientation Angle
    4. Speed
    5. Cross-Track Error
    6. Orientation Error
  * These values are updated, respectively, according to the following equations:
  ```
  fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);                 //x
  fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);                 //y
  fg[1 + psi_start + t] = psi1 - (psi0 + (v0/Lf) * del0 * dt);                  //psi
  fg[1 + v_start + t] = v1 - (v0 + (a0 * dt));                                  //v
  fg[1 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));   //cte
  fg[1 + epsi_start + t] = epsi1 - ((psi0 - psi_des0) + (v0/Lf) * del0 * dt);   //epsi
  ```
  where fg is a vector containing information regarding the constraints and total error of the system. Values appended with a 0 are the state values of the previous timestep; values appended with a 1 are those of the next timestep.
  * Similarly, the vehicle's actuators are described by 2 parameters:
    1. Acceleration
    2. Steering
  * These values are controlled by the output of the MPC solver which seeks the optimum path given a set of constraints including the aforementioned state update equations.

* Time Horizon
  * The values defining the predictive model's time hoizon are `N` and `dt` in `src/MPC.cpp`
  * These were determined heuristically and videos documenting some of these trials are found in the `Videos/Time Horizons` directory
  * It is observed to be advantageous to select a relatively small value for `N` as a larger value increases computation time
  * It is also advantageous that `dt` be small to ensure appropriate temporal resolution is available in the determination of an optimum path
  * Lastly, it is observed that when the time horizon grows too large, the optimum path becomes one which includes a greater velocity. Consequently, the vehicle's ability to navigate sharper turns becomes limited.
* Preprocessing & Latency
  * Before fitting a curve to the waypoints, the waypoints' coordinates are transformed from the Map space to the Vehicle space with a simple rotation
    * Flat course => 2D Transformation => Rotation
  * The vehicle state passed to the `MPC::Solve` method is the vehicle's current state (in the Vehicle's coordinate space) plus a 100 ms time delay
    * This delay accounts for the vehicle actuators' delayed responsiveness
    * The delay is calculated based on the dynamics of the vehicle's motion:
    ```
    double delay = 0.1;                                         // 100ms
    double x_delayed = (v * delay);                             // cos(0)=1
    double y_delayed = 0;                                       // sin(0)=0
    double psi_delayed = -(v * steer_value * 0.1 / mpc.Lf);
    double v_delayed = v + throttle_value * delay;
    double cte_delayed = cte + (v * sin(epsi) * delay);
    double epsi_delayed = epsi - (v * steer_value * delay / mpc.Lf);
    ```


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.

* **Ipopt and CppAD:** Please refer to [this document](https://github.com/udacity/CarND-MPC-Project/blob/master/install_Ipopt_CppAD.md) for installation instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

## Tips

1. It's recommended to test the MPC on basic examples to see if your implementation behaves as desired. One possible example
is the vehicle starting offset of a straight line (reference). If the MPC implementation is correct, after some number of timesteps
(not too many) it should find and track the reference line.
2. The `lake_track_waypoints.csv` file has the waypoints of the lake track. You could use this to fit polynomials and points and see of how well your model tracks curve. NOTE: This file might be not completely in sync with the simulator so your solution should NOT depend on it.
3. For visualization this C++ [matplotlib wrapper](https://github.com/lava/matplotlib-cpp) could be helpful.)
4.  Tips for setting up your environment are available [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)
5. **VM Latency:** Some students have reported differences in behavior using VM's ostensibly a result of latency.  Please let us know if issues arise as a result of a VM environment.
