#include <iostream>
#include <vector>
#include <string>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/Dense"
#include "spline.h"
#include "vehicle.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
*Instantiate
**/
Vehicle::Vehicle(){}
Vehicle::~Vehicle(){}


void Vehicle::generate_trajectory(vector<vector <double> > waypoints){
	/**
	*Generates a trajectory using the spline library
	*INPUT: Goal lane
	*OUTPUT: None - Sets vehicle's next_x_vals and next_y_vals vectors
	**/

	vector<double> ptsx;
	vector<double> ptsy;
	const int prev_size = previous_path_x.size();

	if(prev_size > 0){
		s = end_path_s;
	}

	//Get get last two points in previous path for smooth transitions
	if(prev_size < 2){

		//If no previous path, extrapolate last position from current position and heading
		double prev_x = x - cos(yaw);
		double prev_y = y - sin(yaw);

		ptsx.push_back(prev_x);
		ptsx.push_back(x);

		ptsy.push_back(prev_y);
		ptsy.push_back(y);
	}
	else{

		//Get last two points from previous path
		ref_x = previous_path_x[prev_size-1];
		ref_y = previous_path_y[prev_size-1];

		double ref_x_prev = previous_path_x[prev_size-2];
		double ref_y_prev = previous_path_y[prev_size-2];

		ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

		ptsx.push_back(ref_x_prev);
		ptsx.push_back(ref_x);

		ptsy.push_back(ref_y_prev);
		ptsy.push_back(ref_y);
	}

	for(int i=0;i<waypoints.size();i++){
		ptsx.push_back(waypoints[i][0]);
		ptsy.push_back(waypoints[i][1]);
	}

	for(int i=0;i<ptsx.size();i++){
		//Translate/rotate reference frame so car is at origin
		double shift_x = ptsx[i] - ref_x;
		double shift_y = ptsy[i] - ref_y;

		ptsx[i] = (shift_x * cos(0-ref_yaw) - shift_y * sin(0-ref_yaw));
		ptsy[i] = (shift_x * sin(0-ref_yaw) + shift_y * cos(0-ref_yaw));
	}

	
	//Use spline to compute trajectory function
	tk::spline s;
	for(int i=1;i<ptsx.size();i++){
		//cout << "X: " << ptsx[i] << " , Y: " << ptsy[i] << endl;
		if(ptsx[i] == ptsx[i-1]){
			ptsx[i] += 0.1;
		}
	}

	s.set_points(ptsx, ptsy);
	

	int n = 50-previous_path_x.size();
	double target_x = buffer;
	double target_y = s(target_x);
	double target_dist = sqrt((target_x)*(target_x) + (target_y)*(target_y));
	double x_add_on = 0;

	//Start with whatever's currently in the path (since last frame was ran)
	for(int i=0;i<previous_path_x.size();i++){
		next_x_vals.push_back(previous_path_x[i]);
		next_y_vals.push_back(previous_path_y[i]);
	}


	//Fill up the rest of our path planner after filling it with previous points
	for(int i=1;i<=n;i++){

		//Update acceleration
		ref_acc += ref_jerk * 0.02;
		if(ref_acc >= ACC_LIMIT){
			ref_acc = ACC_LIMIT-0.5;
		}
		else if(ref_acc <= -ACC_LIMIT){
			ref_acc = (-ACC_LIMIT)+0.5;
		}
		//Update velocity
		ref_vel += (0.02 * ref_acc)*2.24;	//2.24 to get vel in mph from mps
		if((abs(ref_vel - target_speed)/2.24) <= (0.02*ACC_LIMIT)){
			ref_vel = target_speed;
			ref_acc = 0.0;
			ref_jerk = 0.0;
		}
		if(ref_vel > SPEED_LIMIT-0.5){
			ref_vel = SPEED_LIMIT-0.5;
		}

		//Compute next trajectory point
		double N = target_dist / (0.02 * (ref_vel/2.24));	//2.24 to get vel from mph to mps
		double x_point = x_add_on + (target_x)/N;
		double y_point = s(x_point);

		x_add_on = x_point;
		double x_ref = x_point;
		double y_ref = y_point;

		//Translate/rotate back from car's coordinates
		x_point = (x_ref * cos(ref_yaw) - y_ref*sin(ref_yaw));
		y_point = (x_ref * sin(ref_yaw) + y_ref*cos(ref_yaw));

		x_point += ref_x;
		y_point += ref_y;

		next_x_vals.push_back(x_point);
		next_y_vals.push_back(y_point);
	}

	slow_car_on_left_ahead = false;
	fast_car_on_left_behind = false;
	slow_car_on_right_ahead = false;
	fast_car_on_right_behind = false;
	car_straight_ahead = false;

	return;
}

void Vehicle::predict(vector<vector <double> > sensor_fusion){
	/**
	*Determines behavior of surrounding vehicles
	*Assumes other cars are moving in straight lines
	*INPUT: Vector of sensor fusion data vectors - {id,x,y,vx,vy,s,d}
	*OUTPUT: None - Sets vehicle's car_on_left, car_on_right, and car_ahead variables
	**/

	int prev_size = previous_path_x.size();
	Vehicle check_car;

	//Each lane gets a target_speed
	//Chosen to be equal to the speed of closest car in front of vehicle in that lane
	double closest_gap_ahead = numeric_limits<double>::max();
	double closest_gap_left = numeric_limits<double>::max();
	double closest_gap_right = numeric_limits<double>::max();

	target_speed_ahead = numeric_limits<double>::max();;
	target_speed_left = numeric_limits<double>::max();;
	target_speed_right = numeric_limits<double>::max();;


	for (int i=0;i<sensor_fusion.size();i++){

		//Get lane of other car
		check_car.d = sensor_fusion[i][6];
		for(int i=0;i<3;i++){
			if(check_car.d < (2+4*i+2) && check_car.d > (2+4*i-2)){
				check_car.lane = i;
			}
		}

		check_car.vx = sensor_fusion[i][3];
		check_car.vy = sensor_fusion[i][4];
		check_car.speed = sqrt(check_car.vx*check_car.vx + check_car.vy*check_car.vy);
		check_car.s = sensor_fusion[i][5];
		check_car.s += ((double)prev_size*0.02*check_car.speed);

		if(check_car.lane == lane){
			//Car ahead w/in buffer
			car_straight_ahead |= ((check_car.s > s) && ((check_car.s - s) <= buffer));
			//Save car's speed as target
			if(car_straight_ahead && (check_car.s - s <= closest_gap_ahead)){
				target_speed_ahead = 2.24*check_car.speed;
				closest_gap_ahead = check_car.s - s;
			}
		}
		else if(check_car.lane == (lane - 1)){
			//car ahead on the left, moving slower than car's current path
			slow_car_on_left_ahead |= ((check_car.s > s) && (check_car.s - s <= buffer) && (check_car.speed < target_speed));
			//car behind on the left, moving faster than car's current path
			fast_car_on_left_behind |= ((check_car.s < s) && (s - check_car.s <= buffer) && (check_car.speed > target_speed));

			if(slow_car_on_left_ahead && (check_car.s - s <= closest_gap_left)){
				target_speed_left = 2.24*check_car.speed;
				closest_gap_left = check_car.s - s;
			}
		}
		else if(check_car.lane == (lane + 1)){
			//car ahead on the right, moving slower than car's current path
			slow_car_on_right_ahead |= ((check_car.s > s) && (check_car.s - s <= buffer) && (check_car.speed < target_speed));
			//car ahead on the right, moving slower than car's current path
			fast_car_on_right_behind |= ((check_car.s < s) && (s - check_car.s <= buffer) && (check_car.speed > target_speed));

			if(slow_car_on_right_ahead && (check_car.s - s <= closest_gap_right)){
				target_speed_right = 2.24*check_car.speed;
				closest_gap_right = check_car.s - s;
			}
		}
	}
	return;
}

void Vehicle::choose_action(){
	/**
	*Assess validity of possible moves
	*3 possible states - Change lane left, Change lane right, Keep lane
	*INPUT: None
	*OUTPUT: None - Sets vehicle's lane or adjust speed as necessary and safe
	**/

	if(car_straight_ahead){
		adjust_speed();
		//cout << "CAR STRAIGHT AHEAD" << endl;
		if(!slow_car_on_left_ahead && !fast_car_on_left_behind && lane != 0){
			//Left lane clear, change left
			//cout << "LCL" << endl;
			lane--;
			target_speed = target_speed_left;
		}
		else if(!slow_car_on_right_ahead && !fast_car_on_right_behind && lane != 2){
			//Can't pass left, right lane clear, change right
			//cout << "LCR" << endl;
			lane++;
			target_speed = target_speed_right;
		}
		else{
			target_speed = target_speed_ahead;
		}
	}
	else{
		//Lane is clear, seek speed limit
		target_speed = SPEED_LIMIT;
		adjust_speed();
	}

	return;
}

void Vehicle::adjust_speed(){
	/**
	*Sets vehicle's ref_jerk to ramp acceleration up/down as need to reach target_speed
	*INPUT: None - Uses vehicle's target_speed value
	*OUTPUT: None - Sets vehicle's ref_jerk; Sets ref_acc to 0 if needed
	**/

	if(ref_vel > target_speed-0.5){
		//Ramp down acceleration if moving too fast
		//cout << "SLOWING DOWN" << endl;
		ref_jerk = -JERK_LIMIT;
	}
	else if(ref_vel < target_speed-0.5){
		//Ramp up acceleration if moving too slow
		//cout << "SPEEDING UP" << endl;
		ref_jerk = JERK_LIMIT;
	}
	else{
		//Stop accelerating if speed is close to target
		//cout << "MAINTAINING SPEED" << endl;
		ref_jerk = 0.0;
		ref_acc = 0.0;
	}
	return;
}
