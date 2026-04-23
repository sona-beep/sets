#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "plane_model.hpp"

struct MissileState {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double speed = 0.0;
    double theta = 0.0;
    double psi = 0.0;

    static MissileState from_joint_state(const Eigen::VectorXd &joint_state) {
        assert(joint_state.size() >= 12);
        MissileState state;
        state.x = joint_state(6);
        state.y = joint_state(7);
        state.z = joint_state(8);
        state.speed = joint_state(9);
        state.theta = joint_state(10);
        state.psi = joint_state(11);
        return state;
    }

    Eigen::Vector3d position() const {
        return Eigen::Vector3d(x, y, z);
    }

    Eigen::Vector3d direction() const {
        return Eigen::Vector3d(
            std::cos(theta) * std::cos(psi),
            std::cos(theta) * std::sin(psi),
            -std::sin(theta));
    }

    Eigen::Vector3d velocity() const {
        return speed * direction();
    }

    void write_to_joint_state(Eigen::VectorXd &joint_state) const {
        assert(joint_state.size() >= 12);
        joint_state(6) = x;
        joint_state(7) = y;
        joint_state(8) = z;
        joint_state(9) = speed;
        joint_state(10) = theta;
        joint_state(11) = psi;
    }
};


class MissileModel {

    public:

        explicit MissileModel(const YAML::Node &config) {
            m_min_speed = node_value_or<double>(config, "missile_min_speed", 250.0);
            m_max_speed = node_value_or<double>(config, "missile_max_speed", 1400.0);
            m_speed_command = node_value_or<double>(config, "missile_speed_command", 650.0);
            m_speed_response = node_value_or<double>(config, "missile_speed_response", 1.5);
            m_turn_gain = node_value_or<double>(config, "missile_turn_gain", 2.5);
            m_max_pitch_rate = node_value_or<double>(config, "missile_max_pitch_rate", 1.0);
            m_max_yaw_rate = node_value_or<double>(config, "missile_max_yaw_rate", 1.2);
            m_max_pitch = node_value_or<double>(config, "missile_max_pitch", 1.3);
            m_lead_gain = node_value_or<double>(config, "missile_lead_gain", 0.75);
            m_navigation_constant = node_value_or<double>(config, "missile_navigation_constant", 4.0);
            m_guidance_lookahead = node_value_or<double>(config, "missile_guidance_lookahead", 1.0);
            m_alignment_gain = node_value_or<double>(config, "missile_alignment_gain", 0.40);
        }

        MissileState derivative(const MissileState& missile_state, const PlaneState& plane_state) const {
            const Eigen::Vector3d relative_position = plane_state.position() - missile_state.position();
            const Eigen::Vector3d relative_velocity = plane_state.velocity() - missile_state.velocity();
            const double relative_distance = std::max(relative_position.norm(), 1.0e-6);
            const Eigen::Vector3d los_direction = relative_position / relative_distance;

            const double missile_speed = std::max(missile_state.speed, m_min_speed);
            const Eigen::Vector3d missile_direction = missile_state.direction();
            const double closing_speed = std::max(0.0, -los_direction.dot(relative_velocity));
            const Eigen::Vector3d los_rate =
                relative_position.cross(relative_velocity) / std::max(relative_distance * relative_distance, 1.0);
            const Eigen::Vector3d pn_direction_delta =
                m_guidance_lookahead *
                m_navigation_constant *
                closing_speed / std::max(missile_speed, 1.0) *
                los_rate.cross(missile_direction);

            Eigen::Vector3d desired_direction =
                missile_direction +
                pn_direction_delta +
                m_alignment_gain * (compute_intercept_direction(missile_state, plane_state) - missile_direction);
            if (desired_direction.norm() < 1.0e-6) {
                desired_direction = los_direction;
            }
            if (desired_direction.norm() < 1.0e-6) {
                desired_direction = missile_direction;
            }
            desired_direction.normalize();

            const double desired_theta =
                std::atan2(-desired_direction.z(),
                    std::sqrt(desired_direction.x() * desired_direction.x() +
                              desired_direction.y() * desired_direction.y()));
            const double desired_psi = std::atan2(desired_direction.y(), desired_direction.x());

            MissileState dxdt;
            dxdt.x = missile_speed * std::cos(missile_state.theta) * std::cos(missile_state.psi);
            dxdt.y = missile_speed * std::cos(missile_state.theta) * std::sin(missile_state.psi);
            dxdt.z = -missile_speed * std::sin(missile_state.theta);
            dxdt.speed = m_speed_response * (m_speed_command - missile_speed);
            dxdt.theta = clamp_value(
                m_turn_gain * wrap_angle(desired_theta - missile_state.theta),
                -m_max_pitch_rate,
                m_max_pitch_rate);
            dxdt.psi = clamp_value(
                m_turn_gain * wrap_angle(desired_psi - missile_state.psi),
                -m_max_yaw_rate,
                m_max_yaw_rate);
            return dxdt;
        }

        void project_state(MissileState& state) const {
            state.speed = clamp_value(state.speed, m_min_speed, m_max_speed);
            state.theta = clamp_value(state.theta, -m_max_pitch, m_max_pitch);
            state.psi = wrap_angle(state.psi);
        }

    private:

        template<typename T>
        T node_value_or(const YAML::Node& node, const std::string& key, const T& default_value) const {
            if (!node[key]) {
                return default_value;
            }
            return node[key].as<T>();
        }

        Eigen::Vector3d compute_intercept_direction(
            const MissileState& missile_state,
            const PlaneState& plane_state) const {

            Eigen::Vector3d lookahead_vector =
                (plane_state.position() + m_lead_gain * plane_state.velocity()) - missile_state.position();
            if (lookahead_vector.norm() < 1.0e-6) {
                lookahead_vector = plane_state.position() - missile_state.position();
            }
            if (lookahead_vector.norm() < 1.0e-6) {
                return missile_state.direction();
            }
            return lookahead_vector.normalized();
        }

        static double clamp_value(double value, double lower, double upper) {
            return std::max(lower, std::min(value, upper));
        }

        static double wrap_angle(double angle) {
            return std::atan2(std::sin(angle), std::cos(angle));
        }

        double m_min_speed = 250.0;
        double m_max_speed = 1400.0;
        double m_speed_command = 650.0;
        double m_speed_response = 1.5;
        double m_turn_gain = 2.5;
        double m_max_pitch_rate = 1.0;
        double m_max_yaw_rate = 1.2;
        double m_max_pitch = 1.3;
        double m_lead_gain = 0.75;
        double m_navigation_constant = 4.0;
        double m_guidance_lookahead = 1.0;
        double m_alignment_gain = 0.40;
};
