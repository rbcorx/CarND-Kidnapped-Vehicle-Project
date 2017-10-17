/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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

default_random_engine gen;


//for the update step
double bivariate_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
	return exp(-((x-mu_x)*(x-mu_x)/(2*sig_x*sig_x) + (y-mu_y)*(y-mu_y)/(2*sig_y*sig_y))) / (2.0*3.14159*sig_x*sig_y);
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//create normal distributions for each feature
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 250;
	weights.resize(num_particles);

	// init each particle
	for (int i = 0; i < num_particles; i++){
		Particle particle_i;

		particle_i.id = i;
		particle_i.x = dist_x(gen);
		particle_i.y = dist_y(gen);
		particle_i.theta = dist_theta(gen);
		particle_i.weight = 1;

		weights[i] = 1;

		particles.push_back(particle_i);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	for (int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) > 0.001) {

			//update the particle position
			Particle particle_i;
			particle_i.x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particle_i.y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particle_i.theta = particles[i].theta + yaw_rate*delta_t;

			//add gaussian noise
			normal_distribution<double> dist_x(particle_i.x, std_pos[0]);
			normal_distribution<double> dist_y(particle_i.y, std_pos[1]);
			normal_distribution<double> dist_theta(particle_i.theta, std_pos[2]);

			//set particle state
			particles[i].x = dist_x(gen);
			particles[i].y = dist_y(gen);
			particles[i].theta = dist_theta(gen);
		}

		else {
			//update the particle position
			Particle particle_i;
			particle_i.x = particles[i].x + velocity*cos(particles[i].theta)*delta_t;
			particle_i.y = particles[i].y + velocity*sin(particles[i].theta)*delta_t;
			particle_i.theta = particles[i].theta + yaw_rate*delta_t;

			//add gaussian noise
			normal_distribution<double> dist_x(particle_i.x, std_pos[0]);
			normal_distribution<double> dist_y(particle_i.y, std_pos[1]);
			normal_distribution<double> dist_theta(particle_i.theta, std_pos[2]);

			//set particle state
			particles[i].x = dist_x(gen);
			particles[i].y = dist_y(gen);
			particles[i].theta = dist_theta(gen);
		}

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//Nearest neighbour calculation

	double m_dist = 0;
	for (int i = 0; i<observations.size(); i++){

		double min_dist = 999999;
		int closest_landmark = -1;

		for (int j = 0; j < predicted.size(); j++) {
			m_dist = dist(observations[i].x,observations[i].y, predicted[j].x, predicted[j].y);

			if (m_dist < min_dist){
				min_dist = m_dist;
				closest_landmark = predicted[j].id;
			}
		}
		observations[i].id = closest_landmark;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();

	for (int i=0; i<particles.size();i++){

		vector<LandmarkObs> observations_map;

		for (int j=0; j<observations.size(); j++){
			LandmarkObs m_observation;

			m_observation.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
			m_observation.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
			m_observation.id = -1;

			observations_map.push_back(m_observation);
		}


		vector<LandmarkObs> pred_meas;

		for (int j = 0; j <map_landmarks.landmark_list.size(); j++) {
			double m_dist_particle_obs;

			m_dist_particle_obs = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f );

			if (m_dist_particle_obs<sensor_range){
				LandmarkObs pred_landmark;
				pred_landmark.id = map_landmarks.landmark_list[j].id_i;
				pred_landmark.x = map_landmarks.landmark_list[j].x_f;
				pred_landmark.y = map_landmarks.landmark_list[j].y_f;

				pred_meas.push_back(pred_landmark);

			}
		}

		dataAssociation(pred_meas, observations_map);
		double prob = 1;
		double prob_j;

		for (int j = 0; j < pred_meas.size(); j++) {
			int id_min = -1;
			double min_dist = 99999;

			for (int k = 0; k < observations_map.size(); k++) {
				double m_dist = dist(pred_meas[j].x, pred_meas[j].y, observations_map[k].x, observations_map[k].y);

				if (m_dist< min_dist){
					min_dist = m_dist;
					id_min = k;
				}
			}

			if (id_min != -1){
				prob_j = bivariate_normal(pred_meas[j].x, pred_meas[j].y, observations_map[id_min].x, observations_map[id_min].y, std_landmark[0], std_landmark[1]);

				prob = prob * prob_j;
			}
		}

		weights.push_back(prob);
		particles[i].weight = prob;

	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd_wts;
	std::mt19937 generator_wts(rd_wts());


	// Creates a discrete distribution for weight.
	std::discrete_distribution<int> distribution_wts(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;

	// Resample
	for(int i=0;i<num_particles;i++){
		Particle particles_i = particles[distribution_wts(generator_wts)];
		resampled_particles.push_back(particles_i);
	}
	particles = resampled_particles;



}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
