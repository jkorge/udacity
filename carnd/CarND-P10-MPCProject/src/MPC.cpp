#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// Set N and dt
// Time horizon = N*dt
size_t N = 10;
double dt = 0.1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Reference Velocity
double ref_v = 70;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
 public:

  // Coefficients of the fitted polynomial.
  Eigen::VectorXd coeffs;

  // Constructor
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  // Shorthand for CppAD TestVector type
  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  // `fg` is a vector containing the cost and constraints.
  // `vars` is a vector containing the variable values (state & actuators).
  void operator()(ADvector& fg, const ADvector& vars) {


    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`
    fg[0] = 0;

    /********************************************************************************************
    * Reference State Cost
    ********************************************************************************************/
    // Accumulate cost

    // += square of difference from reference for each of CTE, orientation, and velocity
    // CTE and orientation reference are both 0
    for(unsigned int t=0;t<N;t++){
      fg[0] += 100 * CppAD::pow(vars[cte_start + t], 2);
      fg[0] += 100 * CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // += square of difference from reference for both actuator values
    // seek to minimize use of actuators => reference value=0
    // don't penalize use in final timestep since it doesn't matter at that point
    for(unsigned int t=0;t<N-1;t++){
      fg[0] += 50 * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += 50 * CppAD::pow(vars[a_start + t], 2);
    }

    // += square of difference in actuator values between two timestemps
    // seek to minimize jerk (da/dt)
    // don't penalize jerk in final timestep since we don't care where it goes from there
    for(unsigned int t=0;t<N-2;t++){
      fg[0] += 5000 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += 5000 * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }


    /********************************************************************************************
    * Model Constraints
    ********************************************************************************************/

    // Initial constraints
    // Add 1 to each of the starting indices due to cost being located at index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (unsigned int t = 1; t < N; t++) {

      // State @ time t
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // State @ time t+1
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // Controls taking state from t to t+1
      AD<double> del0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      // Reference trajectory - Path and orientation
      // Polyeval declared out of scope from this point - replicate method here
      AD<double> f0 = coeffs[0] + coeffs[1]*x0 + coeffs[2]*CppAD::pow(x0,2) + coeffs[3]*CppAD::pow(x0,3);
      AD<double> psi_des0 = CppAD::atan(coeffs[1] + coeffs[2]*x0 + coeffs[3]*CppAD::pow(x0,2));


      /********************************************************************************************
      * Model Equations:
      * x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      * y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      * psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      * v_[t+1] = v[t] + a[t] * dt
      * cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      * epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      ********************************************************************************************/

      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);                 //x
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);                 //y
      fg[1 + psi_start + t] = psi1 - (psi0 + (v0/Lf) * del0 * dt);                  //psi
      fg[1 + v_start + t] = v1 - (v0 + (a0 * dt));                                  //v
      fg[1 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));   //cte
      fg[1 + epsi_start + t] = epsi1 - ((psi0 - psi_des0) + (v0/Lf) * del0 * dt);   //epsi
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  //size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  /********************************************************************************************
  * Initial State Values
  ********************************************************************************************/

  const double x = state[0];
  const double y = state[1];
  const double psi = state[2];
  const double v = state[3];
  const double cte = state[4];
  const double epsi = state[5];

  // number of independent variables
  // N timesteps == N - 1 actuations
  const size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  const size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (unsigned int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }
  // Set the initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  /********************************************************************************************
  * Model Constraints - Limiting Values
  ********************************************************************************************/

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (unsigned int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -25 and 25 degrees (values in radians).
  for (unsigned int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  for (unsigned int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (unsigned int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);
  
  /********************************************************************************************
  * IPOPT Solver
  ********************************************************************************************/

  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  //auto cost = solution.obj_value;
  //std::cout << "Cost " << cost << std::endl;

  // Return actuator results
  vector<double> result;
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  // Include predicted path in results for visualization
  for(unsigned int i=0;i<N-1;i++){
    result.push_back(solution.x[1 + x_start + i]);
    result.push_back(solution.x[1 + y_start + i]);
  }

  return result;
}
