#include "PID.h"
#include <vector>
#include <limits>
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {


	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;

	p_error = 0.0;
	i_error = 0.0;
	d_error = 0.0;

	/*
	//swtich for attempting to twiddle
	twiddle = false;

	// Add inputs to K vector
	K.push_back(Kp);
	K.push_back(Ki);
	K.push_back(Kd);



	// Initialize dp
	dp.push_back(1.0);
	dp.push_back(0.75);
	dp.push_back(1.25);

	// Start at end of K (will wrap around in first run)
	adjust_param = K.size() - 1;
	n_steps = 0;
	// Wait some steps before accumulating error
	n_hold_steps = 10;
	// Wait more steps before adjusting dp
	n_run_steps = 20;

	// Start at infinity and work downwards
	best_err = numeric_limits<double>::max();
	err = 0;

	// Haven't adjusted anything at first
	checked_add = false;
	checked_sub = false;
	*/
}

void PID::UpdateError(double cte) {

	// Adjust errors
	d_error = cte - p_error;
	p_error = cte;
	i_error += cte;
	/*
	// Start accumulating error after n_hold_steps
	if((n_steps % n_run_steps) > n_hold_steps){
		err += cte*cte;
	}

	// Run single loop of twiddle every n_run_steps
	if(twiddle && ((n_steps % n_run_steps) == 0)){

		// Check if error has improved
		if(err < best_err){
			best_err = err;


			if(n_steps != n_run_steps){
				// Guaranted err<best_err on first run
				// Senseless to adjust dp yet
				dp[adjust_param] *= 1.1;
			}
			
			// Move on to next param
			adjust_param = (adjust_param + 1) % (K.size());
			checked_add = false;
			checked_sub = false;
		}
		
		// First, try incrementing by dp
		if(!checked_add && !checked_sub){
			cout<<"Adding "<<dp[adjust_param]<<" to param "<<adjust_param<<endl;
			K[adjust_param] += dp[adjust_param];
			checked_add = true;
		}

		// If already added, try decrementing by dp
		else if(checked_add && !checked_sub){
			cout<<"Subtracting "<<dp[adjust_param]<<" from param "<<adjust_param<<endl;
			K[adjust_param] -= 2*dp[adjust_param];
			checked_sub = true;
		}

		// If tried both, return to original value & decrement dp
		else{
			cout<<"Returning param "<<adjust_param<<" to its original value"<<endl;
			K[adjust_param] += dp[adjust_param];
			dp[adjust_param] *= 0.9;
			
			// Move on to next param
			adjust_param = (adjust_param + 1) % K.size();
			checked_add = false;
			checked_sub = false;
		}
		
		// Reset error for next n_run_steps
		err = 0.0;
	}
	n_steps++;
	*/
}

double PID::TotalError() {

	double res;
	/*
	if(!twiddle){
		res = (Kp*p_error) + (Kd*d_error) + (Ki*i_error);
	}
	else{
		res = (K[0]*p_error) + (K[1]*d_error) + (K[2]*i_error);
	}
	*/
	res = (-Kp*p_error) - (Kd*d_error) - (Ki*i_error);

	return res;
}

