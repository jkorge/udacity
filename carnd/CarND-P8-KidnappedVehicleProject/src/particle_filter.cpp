/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *	Updated on: Aug 18, 2018
 *			Functions filled-in by: James Korge
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	/**************************************************************************************************
	*	INSTANTIATE AN ASSEMBLANCE OF PARTICLES
	**************************************************************************************************/


	// Set the number of particles
	num_particles = 100;

	// Sample from a normal distribution for each particle's initial state
	default_random_engine generator;
	normal_distribution<double> x_dist(x, std[0]);
	normal_distribution<double> y_dist(y, std[1]);
	normal_distribution<double> theta_dist(theta, std[2]);

	for(int i=0;i<num_particles;i++){
		// Set particle values
		Particle particle;
		particle.id = i;
		particle.x = x_dist(generator);
		particle.y = y_dist(generator);
		particle.theta = theta_dist(generator);
		particle.weight = 1.0;

		//Add particle to list
		particles.push_back(particle);

		//Intialize weights to 1
		weights.push_back(1.0);
	}

	is_initialized = true;



}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {


	/**************************************************************************************************
	*	UPDATE PARTICLES' POSITIONS USING VELOCITY & YAW_RATE OVER DELTA_T
	**************************************************************************************************/

	default_random_engine generator;

	for(int i=0;i<num_particles;i++){

		// Get old values for readability
		double x_prev = particles[i].x;
		double y_prev = particles[i].y;
		double theta_prev = particles[i].theta;

		//Calculate new particle values
		double theta_upd = theta_prev + (yaw_rate * delta_t);
		double x_upd;
		double y_upd;

		if(yaw_rate != 0){
			x_upd = x_prev + (velocity / yaw_rate) *(sin(theta_upd) - sin(theta_prev));
			y_upd = y_prev + (velocity / yaw_rate) *(cos(theta_prev) - cos(theta_upd));
		}
		else{
			x_upd = x_prev + velocity * delta_t * cos(theta_prev);
			y_upd = y_prev + velocity * delta_t * sin(theta_prev);
		}

		// Add noise to prediction
		normal_distribution<double> x_dist(x_upd, std_pos[0]);
		normal_distribution<double> y_dist(y_upd, std_pos[1]);
		normal_distribution<double> theta_dist(theta_upd, std_pos[2]);

		particles[i].x = x_dist(generator);
		particles[i].y = y_dist(generator);
		particles[i].theta = theta_dist(generator);


	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	/**************************************************************************************************
	*	FIND LANDMARK NEAREST EACH TRANSFORMED OBSERVATION
	**************************************************************************************************/

	for(unsigned int i=0;i<observations.size();i++){
		double min_dist = -1.0;
		double dist;

		for(unsigned int j=0;j<predicted.size();j++){

			// Euclidean distance from observation[i] to pre-selected landmark[j]
			dist = sqrt(pow(observations[i].x - predicted[j].x, 2) + pow(observations[i].y - predicted[j].y, 2));

			if((min_dist == -1.0) || (dist < min_dist)){
				min_dist = dist;
				observations[i].id = j;
			}
		}
	}



}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	for(int j=0;j<num_particles;j++){

		/**************************************************************************************************
		*	FILTER LANDMARKS USING SENSOR_RANGE
		**************************************************************************************************/

		vector<LandmarkObs> predicted; 		//Vector of landmarks <= sensor_range meters away from particle

		double dist;

		for(unsigned int i=0;i<map_landmarks.landmark_list.size();i++){
			// Euclidean distance to from particle to landmark
			dist = sqrt(pow(particles[j].x - map_landmarks.landmark_list[i].x_f, 2) + pow(particles[j].y - map_landmarks.landmark_list[i].y_f, 2));

			// If within sensor_range, keep
			if(dist <= sensor_range){
				LandmarkObs pred;
				pred.x = map_landmarks.landmark_list[i].x_f;
				pred.y = map_landmarks.landmark_list[i].y_f;
				pred.id = map_landmarks.landmark_list[i].id_i;
				predicted.push_back(pred);
			}
		}


		/**************************************************************************************************
		*	TRANSFORM OBSERVATIONS INTO MAP SPACE
		**************************************************************************************************/

		vector<LandmarkObs> transformed(observations.size());	//Vector of observations transformed to map space

		for(unsigned int i=0;i<observations.size();i++){

			/**********Mapping function**********
			* cos()	-sin() x_par		 			x_obv *
			* sin()	 cos() y_par		X  		y_obv *
			* 0			 0		 1				 			1		  *
			*************************************/

			transformed[i].x = particles[j].x + cos(particles[j].theta) * observations[i].x - sin(particles[j].theta)*observations[i].y;
			transformed[i].y = particles[j].y + sin(particles[j].theta)*observations[i].x + cos(particles[j].theta)*observations[i].y;
		}

		// Associate transformed coordinates to predicted landmarks
		dataAssociation(predicted, transformed);


		/**************************************************************************************************
		*	SET PARTICLE ASSOCIATIONS & UPDATE PARTICLE WEIGHT
		**************************************************************************************************/

		// Vectors for accumulating association id's and coordinates
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		// Reset particle's weight
		particles[j].weight = 1.0;

		for(unsigned int i=0;i<transformed.size();i++){

			// Accumulate associations
			associations.push_back(predicted[transformed[i].id].id);
			sense_x.push_back(transformed[i].x);
			sense_y.push_back(transformed[i].y);

			// Update the particle's weight using a Multivariate Gaussian
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double coeff = 1 / (2 * M_PI * std_x * std_y);
			double power1 = (pow(transformed[i].x - predicted[transformed[i].id].x, 2) / (2 * std_x*std_x));
			double power2 = (pow(transformed[i].y - predicted[transformed[i].id].y, 2) / (2 * std_y*std_y));
			
			particles[j].weight *= coeff * exp(-(power1 + power2));
		}

		// Set particle's associations & corresponding parameters
		SetAssociations(particles[j], associations, sense_x, sense_y);

		// Update the list of weights
		weights[j] = particles[j].weight;

	}



}

void ParticleFilter::resample() {

	/**************************************************************************************************
	*	OVERWRITE PARTICLE LIST BY SAMPLING FROM EXISTING PARTICLES IN PROPORTION TO THEIR WEIGHTS
	**************************************************************************************************/

	// Instantiate discrete_distribution
	default_random_engine generator;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	// Collect samples into new vector
	vector<Particle> particles_samp;
	for(int i=0;i<num_particles;i++){
		particles_samp.push_back(particles[distribution(generator)]);
	}

	// Overwrite existing vector
	particles = particles_samp;

}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	/**************************************************************************************************
	*	UPDATE PARTICLE'S ASSOCIATION IDs AND THEIR CORRESPONDING X AND Y COORDS
	**************************************************************************************************/

	// associations => List of map landmark id's associated to observations
  particle.associations= associations;

  //sense_<x/y> => Coordinates of the landmark according to provided map
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
