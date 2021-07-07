# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Notes and Observations

The videos directory included in this repo shows how the simulator responds when the hyperparamters (Kp, Ki, and Kd) are tuned.
The initial run was to test the build and usability of the compiled code. This run set the hyperparameters to zero and implemented no course correction. The car drove in a straight line off the road.
After this, each hyperparamter was set to 1.0 in turn. This was to observe the differing effects they would each have on the outcome of the simulation

Setting Kp to 1.0 caused the vehicle to, at first, correct just enough to stay on the road. However, before the first turn the error became too great and course correction caused the car to veer off-road.

Setting Ki to 1.0 had the car moving in circles almost immediately. It was clear that this parameter should be kept small (in porportion to the other parameters) to prevent the accumulated error from causing overdramatic course corrections.

Setting Kd to 1.0 did not much affect the direction of the car. While the simulator indicated small steering angles were applied to the vehicle's motion, the outcome was similar to setting all hyperparameters to 0. It was expected that this parameter would need to be larger than the others to add an appreciable amount of correction.

The final result is set to (Kp, Ki, Kd) = (1.5, 0.0, 2.5). This was determined by first setting each value to the same order of magnitude as those used in the lecture videos ~ (0.1, 0.001, 1.0). From there the values were individually changed in increments of 0.5 until the car was able to (safely) travarse the course with minimal corrections.


** Note: ** The code in PID.cpp and PID.h includes blocks that have been commented out. These were used to implement a gradient descent (aka Twiddle) algorithm. However, I was unsuccessful at finding good hyperparameters for this method. An alternative would have been to reset the simulator until course-completing parameters are found, but methinks that would take too long.
The same applies to the PID I attempted to use for the vehicle's throttle.

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
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

