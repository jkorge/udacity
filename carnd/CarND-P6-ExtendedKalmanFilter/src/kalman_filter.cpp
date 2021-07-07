#include "kalman_filter.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */

  //Section 5.8
  x_ = F_ * x_;
  //Section 5.9
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  //Section 5.7
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new state
  x_ = x_ + (K * y);

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */

  float x = x_(0);
  float y = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho = sqrt((x*x)+(y*y));
  float theta = atan2(y,x);
  float ro_dot = ((x*vx) + (y*vy))/rho;
  VectorXd z_pred = VectorXd(3);
  z_pred << rho, theta, ro_dot;

  VectorXd y_ = z - z_pred;
  while(abs(y_(1)) > M_PI ){
    if(y_(1) < 0){
      y_(1) = y_(1) + 2*M_PI;
    }
    else{
      y_(1) = y_(1) - 2*M_PI;
    }
  }

  //Section 5.7
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new state
  x_ = x_ + (K * y_);

  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  P_ = (I - K * H_) * P_;
}
