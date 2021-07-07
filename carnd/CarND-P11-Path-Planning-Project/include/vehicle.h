#ifndef VEHICLE_H
#define VEHICLE_H
#include <iostream>
#include <vector>
#include <string>
#include "spline.h"

using namespace std;

class Vehicle {
public:
	/**
	*Variables
	**/

	//Road Rules
	double SPEED_LIMIT;
	double ACC_LIMIT;
	double JERK_LIMIT;

	//Car state values
	double x;
	double y;
	double s;
	double d;
	double yaw;
	double speed;
	double vx;
	double vy;
	int lane = 1;

	//Awareness of surroundings
	bool slow_car_on_left_ahead = false;
	bool fast_car_on_left_behind = false;
	bool slow_car_on_right_ahead = false;
	bool fast_car_on_right_behind = false;
	bool car_straight_ahead = false;
	double target_speed_ahead;
	double target_speed_left;
	double target_speed_right;

	//Remaining values from previous trajectory
	vector<double> previous_path_x;
	vector<double> previous_path_y;
	double end_path_s;
	double end_path_d;

	//Reference values for computing trajectory
	double ref_x;
	double ref_y;
	double ref_yaw;
	double ref_vel = 0.0;
	double ref_acc = 0.0;
	double ref_jerk = 0.0;
	double target_speed = 0.0;
	int buffer = 60;	//preferred distance from other cars

	//Trajectory vectors
	vector<double> next_x_vals;
	vector<double> next_y_vals;

	/**
	*Constructor
	**/
	Vehicle();

	/**
	*Destructor
	**/
	virtual ~Vehicle();

	/**
	*Functions
	**/

	void generate_trajectory(vector<vector <double> > waypoints);

	void predict(vector<vector <double> > sensor_fusion);

	void choose_action();

	void adjust_speed();

	vector<double> JMT(vector<double> start_state, vector<double> goal_state, float T);

};



#endif
