# Programming A Real Self-Driving Car

## By Team Autonomous Mobility

### Team Lead: Tony Lin

### Members:

|        Name         |            Email             |
|---------------------|:-----------------------------|
|     James Korge	  |       jk5128@nyu.edu         |
|     Yulong Li	      |      hiyulongl@gmail.com	 |
|    Islam Sherif     |   eng.islam.sherif@gmail.com |
|   Fuqiang Huang	  |       fuqiang@wustl.edu      |
|     Tony Lin        |     dr.tony.lin@gmail.com	 |

## Credits

* The traffic light detector/classifier used SSD Inception V2 from Tensorflow: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz

. The training data for traffic light detector was from https://github.com/alex-lechner/Traffic-Light-Classification

. THe usage section was from the original README.md from https://github/udacity/CarND-Capstone

## Build

### Install Dependent Packages

Before you can run catkin_make, you need to install dependent packages first:

```
cd ros
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
```

### Make ros/src/CMakeLists.txt is linked

After dependent packages are installed, make sure ros/src/CMakeLists.txt is linked to /opt/ros/kinetic/share/catkin/cmake/toplevel.cmake.
Otherwise, make a symbolic link:

'''
cd ros/src
rm CMakeLists.txt
ln -s /opt/ros/kinetic/share/catkin/cmake/toplevel.cmake src/CMakeLists.txt
'''

## Usage

### Run the simulator testing

1. Run the followings

```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

If catkin_make failed with error, try 
'''
catkin_make -j1
'''

to run the build sequentially.

2. Launch the simulator, check <B>Camera</B> and uncheck <B>Manual</B>

### Real world testing

1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.

2. Unzip the file

```bash
unzip traffic_light_bag_file.zip
```

3. Play the bag file

```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```

4. Launch your project in site mode

```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```

5. Confirm that traffic light detection works on real life images

### Trouble Shooting

1. On docker, the traffic light detecter might encounter camera image shape index out of bound error. If this happen, update pillow with:

'''
pip install --upgrade pillow
'''

2. Running the simulator on a under powered system can be challenging, from time to time, the simulator will either stop sending pose and velocity updates, or send invalid pose and velocity (mostly 0 velocity) updates. It is important to make sure that your system has sufficient computation power to run ROS with the simulator.

## Configuration Parameters

### Global Parameters

The following global parameters are provided under ros/launch/styx.launch, and ros/launch/site.launch:

* loglevel: we use this parameter to control logging on top of ros's logging framework
* traffic_light_lookahead_wps: the number of waypoints to lookahead for traffic light
* traffic_light_classifier_model: the trained model for traffic light detection

Parameters for the vehicle:

* vehicle_mass: mass of the vehicle
* fuel_capacity: fuel capacity
* brake_deadband: brake deadband
* decel_limit: deceleration limit
* accel_limit: acceleration limit
* wheel_radius: wheel radius
* wheel_base: wheel base length
* steer_ratio: steering ratio
* max_lat_accel: maximum lateral acceleration
* max_throttle: maximum throttle
* max_steer_angle: maximum steering angle
* full_stop_brake_keep: the amount of brake to keep when the stopping the vehicle
* full_stop_brake_limit: the maximum amount of brake that could be applied to the vehicle
* brake_deceleration_start: to avoid braking the vehicle too frequently, this parameter provides the threshold on when brake should be applied

#### Waypoint Update

Waypoint updater takes the following parameters:

* lookahead_wps: the number of waypoints to updates in every cycle
* waypoint_update_frequency: the waypoint update frequency
* traffic_light_stop_distance: the stop distance for traffic light
* /traffic_light_lookahead_wps: the number of waypoints that we need to decelerate for red light
* acceleration_start_velocity: the velocity to start with when start the vehicle after stopping at traffic light
* acceleration_distance: the target distance to accelerate to full velocity
* far_traffic_light_distance: the distance to the next traffic from which we should reduce the velocity in order to be able to stop for red lights
* near_traffic_light_distance: the distance to the next traffic from which we should regain the velocity so we will not pass green light at a velocity that is too slow
* /accel_limit: maximum acceleration fo the vehicle
* /decel_limit: maximum deceleration of the vehicle

#### Waypoint Follower

Waypoint follower's pure pursuit algorithm takes the following parameters:

* publish_frequency: the frequency for publishing twist
* subscriber_queue_length: the subscriber queue length for final_waypoints, current_pose, and current_velocity topics
* const_lookahead_distance: pure pursuit constant lookahead distance
* minimum_lookahead_distance: pure pursuit maximum lookahead distance
* lookahead_distance_ratio: pure pursuit lookahead distance ratio with respect to velocity
* maximum_lookahead_distance_ratio: pure pursuit maximum lookahead distance ratio with respect to velocity

#### Twist Controller

The twist controller takes the following parameters:

* control_update_frequency: the frequency that the controller should publish throttle, steering, and brake
* full_stop_brake_keep: the brake level for to stop the vehicle before the traffic light
* full_stop_brake_limit: the minimal velocity for when the full break stop (specified by full_stop_brake_keep) should be applied when stopping the vehicle.
* brake_deceleration_start: the minimal deceleration from where brake should be applied so to avoid unnecessary brake.

#### Traffic Light Detecter

The traffic light detector use the global parameter, traffic_light_classifier_model, to refer to the Tensorflow classification model.
In addition, the following parameters are used for the classification:

* min_positive_score: the minimal score for the result to be considered positive
* /traffic_light_classifier_model: the trained classifier/detecter model

## Traffic Light Detection Package

### For the Simulator

The trained frozen model is: sim_frozen_inference_graph.pb

### For the Site

The trained frozen model is: frozen_inference_graph.pb

## Waypoint Updater Package

THe WaypointUpdater provide waypoints for driving the vehicle alone the road. THe waypoints include the desired position, and velocity for the vehicle. It works as follows:

* When a yellow or red light is detected, it will generate waypoints to slow down the vehicle if the distance and velocity of the vehicle permit, otherwise, it will generate waypoints to pass the traffic light at the highest velocity

* Green light is detected, it will generate waypoints to drive the vehicle at the designated velocity of the road

* When stopping at red light and green light turns on, it will generate waypoints that will accelerate the vehicle to reach the designated road velocity

* In all other case, it will try to generate waypoints according to the velocity of the road

### Deal with missing or late pose/velocity updates

We have observed that from time to time, the simulator may not publish pose and velocity update properly when the system cannot not deliver the computational power required, this can cause the navigation to go off the road. In order to deal with, we added some extra logic in the WaypointUpdater to detect and compensate missing updates. This is done in every loop where the number of missing updates is above a threshold.

### Allow Higher Velocity

To allow the vehicle to run over 80 KM/h, we adjust the maximum speed of a waypoint according to the distance to the next traffic light, so the vehicle will not run at a speed that is difficult for the vehicle to stop.

## DBW Package

The DBW package's PID controller for the throttle has the following parameters:

* P: 0.1
* I: 0.05
* D: 1.2

THe low pass filter for vehicle has the following parameters to make it more sensitive to the latest velocity:

* tau: 0.2
* ts: 0.2

## Waypoint Follower Package

The PurePursuit::calcTwist function has been changed to always computing the angular velocity in order to avoid the vehicle to wander around the road.

## Waypoint Util Package

The Waypoints class defined in waypoint_util package provides the following Waypoint functions for WaypointUpdater and TLDetector. 

* Indexing and slicing of way points
* find_closest_waypoint: to find the waypoint that is closest to the given position
* distance: compute the distance between two waypoint indices
* before: a predicate for determining if a waypoint is before another 

### Diagnostic Tool

1. Make sure that the python dependencies installed before using the diagnosis tool

```
pip install -r requirements_debug.txt
```

2. Open a new terminal and source

```bash
cd ~/CarND-Capstone
cd ros
source devel/setup.bash
```

3. Make the diagnosis python file executable

```bash
cd src/tools
chmod +x diagScreen.py
cd ~/CarND-Capstone/ros
```

4. Run the diagnosis file

```bash
rosrun tools diagScreen.py --screensize 5 --maxhistory 800 --textspacing 75 --fontsize 1.5
```

Note that there are five options to choose the screensize:  
help='Screen sizes: 1:2500x2500px, 2:1250x1250px, 3:833x833px, 4:625x625px, 5:500x500px '   
You can choose anyone that you like.

5. Run the simulator or real world testing


## Discussion

1. The baseline project uses the Pure Pursuit algorithm for controling the lateral and angular velocity of the vehicle, and PID for determining the throttle. It would be interesting to compare this with MPC.

2. When passing traffic light, the latency caused by traffic light detection/classification can cause the vehicle to run red light when the light changes at sufficiently short distance. However, this may not be a serious issue as comparing to real life scenario, it would be more dangerous to apply fully brake at high speed.
