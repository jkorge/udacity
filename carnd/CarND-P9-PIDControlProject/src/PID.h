#ifndef PID_H
#define PID_H
#include <vector>

using namespace std;

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;


  /*
  * Coefficient Vector
  */ 
  //vector<double> K;
  double Kp;
  double Ki;
  double Kd;

  
  /*
  * Twiddle Variables
  

  // Values by which to adjust K
  vector<double> dp;

  // Step counters
  int n_steps;
  int n_hold_steps;
  int n_run_steps;

  // Best and Current errors
  double best_err;
  double err;

  // Tracking variables for twiddle algorithm
  int adjust_param;
  bool checked_add;
  bool checked_sub;

  //Twiddle on/off switch
  bool twiddle;
  */

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
