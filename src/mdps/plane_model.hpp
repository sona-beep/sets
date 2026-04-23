#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

struct PlaneControl {
    double alpha = 0.0;
    double gamma = 0.0;
    double throttle = 0.0;

    static PlaneControl from_action(const Eigen::VectorXd &action) {
        assert(action.size() >= 3);
        PlaneControl control;
        control.alpha = action(0);
        control.gamma = action(1);
        control.throttle = action(2); 
        return control;
    }
};


struct PlaneState {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double speed = 0.0;
    double theta = 0.0;
    double psi = 0.0;

    static PlaneState from_joint_state(const Eigen::VectorXd &joint_state) {
        assert(joint_state.size() >= 6);
        PlaneState state;
        state.x = joint_state(0);
        state.y = joint_state(1);
        state.z = joint_state(2);
        state.speed = joint_state(3);
        state.theta = joint_state(4);
        state.psi = joint_state(5);
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
        assert(joint_state.size() >= 6);
        joint_state(0) = x;
        joint_state(1) = y;
        joint_state(2) = z;
        joint_state(3) = speed;
        joint_state(4) = theta;
        joint_state(5) = psi;
    }
};


class PlaneModel {

    public:

        explicit PlaneModel(const YAML::Node &config) {
            m_mass = node_value_or<double>(config, "aircraft_mass", 13680.0);
            m_gravity = node_value_or<double>(config, "gravity", 9.8);
            m_ref_area = node_value_or<double>(config, "aircraft_ref_area", 49.24);
            m_base_thrust = node_value_or<double>(config, "aircraft_base_thrust", 80000.0);
            m_delta_thrust = node_value_or<double>(config, "aircraft_delta_thrust", 70000.0);
            m_min_speed = node_value_or<double>(config, "aircraft_min_speed", 150.0);
            m_max_speed = node_value_or<double>(config, "aircraft_max_speed", 1200.0);
            m_max_pitch = node_value_or<double>(config, "aircraft_max_pitch", 1.2);
            m_min_cos_pitch = node_value_or<double>(config, "aircraft_min_cos_pitch", 0.05);
            m_lift_bias = node_value_or<double>(config, "aircraft_lift_bias", -0.0434);
            m_lift_alpha = node_value_or<double>(config, "aircraft_lift_alpha", 0.1369);
            m_normal_bias = node_value_or<double>(config, "aircraft_normal_bias", 0.1310);
            m_normal_alpha = node_value_or<double>(config, "aircraft_normal_alpha", 3.0825);
        }

        PlaneState derivative(const PlaneState& state, const PlaneControl& control) const {
            const double speed = std::max(state.speed, m_min_speed);
            const double rho = atmosphere_density(std::abs(state.z));

            const double lift_like = m_lift_bias + m_lift_alpha * control.alpha;
            const double normal_like = m_normal_bias + m_normal_alpha * control.alpha;
            const double CL = lift_like * std::sin(control.alpha) + normal_like * std::cos(control.alpha);
            const double CD = -lift_like * std::cos(control.alpha) + normal_like * std::sin(control.alpha);

            const double lift = 0.5 * rho * speed * speed * CL * m_ref_area;
            const double drag = 0.5 * rho * speed * speed * CD * m_ref_area;
            const double thrust = m_base_thrust + control.throttle * m_delta_thrust;

            PlaneState dxdt;
            dxdt.x = speed * std::cos(state.theta) * std::cos(state.psi);
            dxdt.y = speed * std::cos(state.theta) * std::sin(state.psi);
            dxdt.z = -speed * std::sin(state.theta);
            dxdt.speed = (thrust * std::cos(control.alpha) - drag) / m_mass - m_gravity * std::sin(state.theta);
            dxdt.theta =
                (lift + thrust * std::sin(control.alpha)) * std::cos(control.gamma) / (m_mass * speed) -
                m_gravity * std::cos(state.theta) / speed;
            dxdt.psi =
                (lift + thrust * std::sin(control.alpha)) * std::sin(control.gamma) /
                (m_mass * speed * safe_cosine(state.theta));
            return dxdt;
        }

        void project_state(PlaneState &state) const {
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

        static PlaneState add_scaled(const PlaneState& state, const PlaneState& delta, double scale) {
            PlaneState out = state;
            out.x += scale * delta.x;
            out.y += scale * delta.y;
            out.z += scale * delta.z;
            out.speed += scale * delta.speed;
            out.theta += scale * delta.theta;
            out.psi += scale * delta.psi;
            return out;
        }

        double atmosphere_density(double altitude_magnitude) const {
            const double altitude_km = altitude_magnitude / 1000.0;
            constexpr double temperature0 = 288.15;
            constexpr double density0 = 1.225;
            if (altitude_km < 11.0) {
                const double temperature = temperature0 - 6.5 * altitude_km;
                return density0 * std::pow(temperature / temperature0, 4.256);
            }
            return 0.0;
        }

        double safe_cosine(double angle) const {
            const double cosine = std::cos(angle);
            if (std::abs(cosine) >= m_min_cos_pitch) {
                return cosine;
            }
            return (cosine >= 0.0 ? 1.0 : -1.0) * m_min_cos_pitch;
        }

        static double clamp_value(double value, double lower, double upper) {
            return std::max(lower, std::min(value, upper));
        }

        static double wrap_angle(double angle) {
            return std::atan2(std::sin(angle), std::cos(angle));
        }

        double m_mass = 13680.0;
        double m_gravity = 9.8;
        double m_ref_area = 49.24;
        double m_base_thrust = 80000.0;
        double m_delta_thrust = 70000.0;
        double m_min_speed = 150.0;
        double m_max_speed = 1200.0;
        double m_max_pitch = 1.2;
        double m_min_cos_pitch = 0.05;
        double m_lift_bias = -0.0434;
        double m_lift_alpha = 0.1369;
        double m_normal_bias = 0.1310;
        double m_normal_alpha = 3.0825;
};
