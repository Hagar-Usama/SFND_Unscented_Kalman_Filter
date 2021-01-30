#include "ukf.h"
#include "Eigen/Dense"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**   * Hint: one or more values initialized above might be wildly off...

   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
   is_initialized_ = false;
   // dimensions of matrix
   n_x_ = 5;
   n_aug_ = 7;
   lambda_ = 3 - n_aug_;
   time_us_ = 0;
   Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1); 

   weights_ = VectorXd(n_x_);
   weights_(0) = lambda_/(lambda_ + n_aug_);
   for(int i = 1; i < 2 * n_aug_ + 1; i++)
      weights_(i) = 1/(2*(lambda_ + n_aug_));

   R_radar_ = MatrixXd(3, 3);
   R_radar_ << std_radr_*std_radr_, 0, 0,
               0, std_radphi_*std_radphi_, 0,
               0, 0, std_radrd_*std_radrd_;

   R_lidar_ = MatrixXd(2,2);
   R_lidar_ << std_laspx_*std_laspx_, 0,
               0, std_laspy_*std_laspy_;
   
   NIS_radar_ = 0.;
   NIS_lidar_ = 0.;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
   if(!is_initialized_)
   {
      if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
      {
         P_ << pow(std_radr_, 2), 0, 0, 0, 0,
               0, pow(std_radphi_, 2), 0, 0, 0,
               0, 0, pow(std_radrd_, 2), 0, 0,
               0, 0, 0, 1, 0,
               0, 0, 0, 0, 1;

         double vx = meas_package.raw_measurements_[2] * cos(meas_package.raw_measurements_[1]);
         double vy = meas_package.raw_measurements_[2] * sin(meas_package.raw_measurements_[1]); 
         x_ << meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]),
               meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]),
               meas_package.raw_measurements_[2], //sqrt(pow(vx, 2) + pow(vy, 2)),
               0,
               0;
      }
      else
      {
         P_ << pow(std_laspx_, 2), 0, 0, 0, 0,
               0, pow(std_laspy_, 2), 0, 0, 0,
               0, 0, 0, 1, 0,
               0, 0, 0, 0, 1;

         x_ << meas_package.raw_measurements_[0],
               meas_package.raw_measurements_[1],
               0,
               0,
               0;
      }
   }

   double delta_t = (meas_package.timestamp_ - time_us_)/1000000;
   time_us_ = meas_package.timestamp_;
   Prediction(delta_t);

   if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
      UpdateRadar(meas_package);
   if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
      UpdateLidar(meas_package);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

   VectorXd x_aug_ = VectorXd(n_aug_);
   MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);

   x_aug_.head(n_x_) = x_;
   x_aug_(5) = 0;
   x_aug_(6) = 0;

   P_aug_.fill(0.0);
   P_aug_.topLeftCorner(n_x_, n_x_) = P_;
   P_aug_(5,5) = pow(std_a_, 2);
   P_aug_(6,6) = pow(std_yawdd_, 2);

   MatrixXd Xsig_aug_ = MatrixXd(n_aug_, n_aug_);
   Xsig_aug_.col(0) = x_aug_;
   MatrixXd l = P_aug_.llt().matrixL();

   for(int i = 1; i <= n_aug_; i++)
   {
      Xsig_aug_.col(i) = x_aug_ + sqrt(lambda_ + n_aug_) * l.col(i - 1);
      Xsig_aug_.col(i + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * l.col(i + n_aug_ - 1); 
   }

   for(int i = 0; i < 2 * n_aug_ + 1; i++)
   {
      VectorXd point = Xsig_aug_.col(i);
      double p_x = point(0);
      double p_y = point(1);
      double v = point(2);
      double yaw = point(3);
      double yaw_d = point(4);
      double noise_v = point(5);
      double noise_yaw = point(6);

      if(fabs(yaw_d) > 0.0001)
      {
         Xsig_pred_(0, i) = p_x + v/yaw_d * (sin(yaw + yaw_d * delta_t) - sin(yaw)) + 1/2 * pow(delta_t, 2) * cos(yaw) * noise_v;
         Xsig_pred_(1, i) = p_y + v/yaw_d * (-cos(yaw + yaw_d * delta_t) + cos(yaw)) + 1/2 * pow(delta_t, 2) * sin(yaw) * noise_v;
         Xsig_pred_(2, i) = v + delta_t * noise_v;
         Xsig_pred_(3, i) = yaw + yaw_d*delta_t + noise_yaw * pow(delta_t, 2) * 1/2;
         Xsig_pred_(4, i) = yaw_d + delta_t * noise_yaw;
      }else
      {
         Xsig_pred_(0, i) = p_x + v*cos(yaw) *delta_t + 1/2 * pow(delta_t, 2) * cos(yaw) * noise_v;
         Xsig_pred_(1, i) = p_y + v*sin(yaw) *delta_t+ 1/2 * pow(delta_t, 2) * sin(yaw) * noise_v;
         Xsig_pred_(2, i) = v + delta_t * noise_v;
         Xsig_pred_(3, i) = yaw + yaw_d*delta_t + noise_yaw * pow(delta_t, 2) * 1/2;
         Xsig_pred_(4, i) = yaw_d + delta_t * noise_yaw;
      }


   }

   VectorXd x = VectorXd(n_x_);
   MatrixXd P = MatrixXd(n_x_, n_x_);

   x.fill(0.0);
   for(int i = 0; i < 2 * n_aug_ + 1; i++)
   {
      x = x + weights_(i) * Xsig_pred_.col(i);
   }

   P.fill(0.0);
   VectorXd x_diff;
   for(int i = 0; i < 2 * n_aug_+1; i++)
   {
      x_diff = Xsig_pred_.col(i) - x;
      while(x_diff(3) < M_PI) x_diff(3)+=2*M_PI;
      while(x_diff(3) > M_PI) x_diff(3)-=2*M_PI;

      P = P + weights_(i) * x_diff * x_diff.transpose();
   }

   x_ = x;
   P_ = P;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  //incoming lidar measurement
  VectorXd z = meas_package.raw_measurements_;
  
  // set measurement dimension
  int n_z = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);


  // transform sigma points into measurement space
  for(int i = 0; i < 2*n_aug_+1; i++)
  {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) 
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }  

  //covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose(); 
  }
  
  // add measurement noise covariance matrix
  S = S + R_lidar_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;
  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //calculate NIS
  NIS_lidar = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
   static int n_z = 3;                                         // distance , angle , velocity

   static VectorXd z = VectorXd(n_z);
   z = meas_package.raw_measurements_;

   static MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
   static double p_x,p_y,v,yaw;
   
   for(int i = 0; i < 2 * n_aug_ + 1; i++)
   {
      
      p_x = Xsig_pred_(0, i);
      p_y = Xsig_pred_(1, i);
      v = Xsig_pred_(2, i);
      yaw = Xsig_pred_(3, i);
      
      Zsig(0,i) = sqrt(pow(p_x,2)+pow(p_x,2));                 //distance
      Zsig(1,i) = atan(p_y/p_x);                               //angle
      Zsig(2,i) = v*(p_x*cos(yaw) + p_y*sin(yaw))/Zsig(0,i);   //velocity
   }   

   static MatrixXd z_pred = VectorXd(n_z);
   z_pred.fill(0.0);
   for(int i = 0; i < 2*n_aug_+1; i++)
   { 
     z_pred += weights_(i)*Zsig.col(i)  ;
   }

   
   static MatrixXd Tc = MatrixXd(n_x_ , n_z);
   Tc.fill(0.0);
   static MatrixXd S = MatrixXd(n_x_ , n_z);
   S.fill(0.0);
   static VectorXd x_diff,z_diff;

   for (int i = 0; i < 2 * n_aug_ + 1; ++i)  
   {  
      // residual
      z_diff = Zsig.col(i) - z_pred;

      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      S += weights_(i)* z_diff * z_diff.transpose();

      // state difference
      x_diff = Xsig_pred_.col(i) - x_;
      
      // angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      Tc += weights_(i) * x_diff * z_diff.transpose();
   }

   static MatrixXd IDR = MatrixXd(3,3);
   IDR << pow(std_radr_, 2), 0, 0, 
         0, pow(std_radphi_, 2), 0,
         0, 0, pow(std_radrd_, 2);
   S += IDR;

   // Kalman gain K;
   static MatrixXd K = Tc * S.inverse();

   // residual
   z_diff = z - z_pred;

   // angle normalization
   while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
   while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

   // update state mean and covariance matrix
   x_ = x_ + K * z_diff;
   P_ = P_ - K*S*K.transpose();


}
