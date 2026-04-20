#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "mdp.hpp"
#include "../util/util.hpp"

class PlaneMissileEscape : public MDP {

    public:

        PlaneMissileEscape(std::string config_path) {
            YAML::Node config = YAML::LoadFile(config_path);

            m_state_dim = 13;
            m_action_dim = 3;

            m_verbose = node_value_or<bool>(config, "ground_mdp_verbose", false);
            m_dt = config["ground_mdp_dt"].as<double>();
            m_gamma = config["ground_mdp_gamma"].as<double>();
            m_H = config["ground_mdp_H"].as<int>();
            m_control_hold = node_value_or<int>(config, "ground_mdp_control_hold", 1);
            m_V_alive = node_value_or<double>(config, "ground_mdp_V_alive", 0.0);

            if (config["my_idxs"]) {
                m_my_idxs = config["my_idxs"].as<std::vector<int>>();
            } else {
                m_my_idxs = {0, 1, 2, 3, 4, 5};
            }

            const std::vector<double> x0_yml = config["ground_mdp_x0"].as<std::vector<double>>();
            const std::vector<double> xd_yml = config["ground_mdp_xd"].as<std::vector<double>>();
            const std::vector<double> X_yml = config["ground_mdp_X"].as<std::vector<double>>();
            const std::vector<double> U_yml = config["ground_mdp_U"].as<std::vector<double>>();
            const std::vector<double> Qx_yml = config["ground_mdp_Qx"].as<std::vector<double>>();
            const std::vector<double> Qx_equ_yml = config["ground_mdp_Qx_equ"].as<std::vector<double>>();
            const std::vector<double> Qu_yml = config["ground_mdp_Qu"].as<std::vector<double>>();
            const std::vector<double> Qf_yml = config["ground_mdp_Qf"].as<std::vector<double>>();

            assert(x0_yml.size() == m_state_dim);
            assert(xd_yml.size() == m_state_dim);
            assert(X_yml.size() == m_state_dim * 2);
            assert(U_yml.size() == m_action_dim * 2);
            assert(Qx_yml.size() == m_state_dim);
            assert(Qx_equ_yml.size() == m_state_dim);
            assert(Qu_yml.size() == m_action_dim);
            assert(Qf_yml.size() == m_state_dim);

            m_x0 = Eigen::Map<const Eigen::VectorXd>(x0_yml.data(), x0_yml.size());
            m_xd = Eigen::Map<const Eigen::VectorXd>(xd_yml.data(), xd_yml.size());
            m_X = Eigen::Map<const Eigen::Matrix<double, -1, 2>>(X_yml.data(), m_state_dim, 2);
            m_U = Eigen::Map<const Eigen::Matrix<double, -1, 2>>(U_yml.data(), m_action_dim, 2);
            m_Qx = Eigen::Map<const Eigen::VectorXd>(Qx_yml.data(), Qx_yml.size()).asDiagonal();
            m_Qx_equ = Eigen::Map<const Eigen::VectorXd>(Qx_equ_yml.data(), Qx_equ_yml.size()).asDiagonal();
            m_Qu = Eigen::Map<const Eigen::VectorXd>(Qu_yml.data(), Qu_yml.size()).asDiagonal();
            m_Qf = Eigen::Map<const Eigen::VectorXd>(Qf_yml.data(), Qf_yml.size()).asDiagonal();

            if (config["target_position"]) {
                const std::vector<double> target_position_yml = config["target_position"].as<std::vector<double>>();
                assert(target_position_yml.size() == 3);
                m_target_position = Eigen::Map<const Eigen::VectorXd>(target_position_yml.data(), 3);
            } else {
                m_target_position = m_xd.segment(0, 3);
            }
            m_default_target_position = m_target_position;

            m_reward_alive_weight = node_value_or<double>(config, "reward_alive_weight", 0.05);
            m_reward_target_weight = node_value_or<double>(config, "reward_target_weight", 0.45);
            m_reward_escape_weight = node_value_or<double>(config, "reward_escape_weight", 0.40);
            m_reward_heading_weight = node_value_or<double>(config, "reward_heading_weight", 0.10);
            m_reward_control_weight = node_value_or<double>(config, "reward_control_weight", 0.0);
            m_target_distance_scale = node_value_or<double>(config, "target_distance_scale", 2000.0);
            m_missile_distance_scale = node_value_or<double>(config, "missile_distance_scale", 1200.0);
            m_target_success_radius = node_value_or<double>(config, "target_success_radius", 150.0);
            m_target_success_bonus = node_value_or<double>(config, "target_success_bonus", 1.0);
            m_missile_hit_penalty = node_value_or<double>(config, "missile_hit_penalty", 2.0);
            m_capture_radius = node_value_or<double>(config, "missile_capture_radius", 80.0);

            m_aircraft_mass = node_value_or<double>(config, "aircraft_mass", 13680.0);
            m_gravity = node_value_or<double>(config, "gravity", 9.8);
            m_aircraft_ref_area = node_value_or<double>(config, "aircraft_ref_area", 49.24);
            m_aircraft_base_thrust = node_value_or<double>(config, "aircraft_base_thrust", 80000.0);
            m_aircraft_delta_thrust = node_value_or<double>(config, "aircraft_delta_thrust", 70000.0);
            m_aircraft_min_speed = node_value_or<double>(config, "aircraft_min_speed", 150.0);
            m_aircraft_max_speed = node_value_or<double>(config, "aircraft_max_speed", 1200.0);
            m_aircraft_max_pitch = node_value_or<double>(config, "aircraft_max_pitch", 1.2);
            m_aircraft_min_cos_pitch = node_value_or<double>(config, "aircraft_min_cos_pitch", 0.05);
            m_aircraft_lift_bias = node_value_or<double>(config, "aircraft_lift_bias", -0.0434);
            m_aircraft_lift_alpha = node_value_or<double>(config, "aircraft_lift_alpha", 0.1369);
            m_aircraft_normal_bias = node_value_or<double>(config, "aircraft_normal_bias", 0.1310);
            m_aircraft_normal_alpha = node_value_or<double>(config, "aircraft_normal_alpha", 3.0825);

            m_missile_speed_command = node_value_or<double>(config, "missile_speed_command", 650.0);
            m_missile_speed_response = node_value_or<double>(config, "missile_speed_response", 1.5);
            m_missile_min_speed = node_value_or<double>(config, "missile_min_speed", 250.0);
            m_missile_max_speed = node_value_or<double>(config, "missile_max_speed", 1400.0);
            m_missile_turn_gain = node_value_or<double>(config, "missile_turn_gain", 2.5);
            m_missile_max_pitch_rate = node_value_or<double>(config, "missile_max_pitch_rate", 1.0);
            m_missile_max_yaw_rate = node_value_or<double>(config, "missile_max_yaw_rate", 1.2);
            m_missile_max_pitch = node_value_or<double>(config, "missile_max_pitch", 1.3);
            m_missile_lead_gain = node_value_or<double>(config, "missile_lead_gain", 0.75);
        }

        std::string name() override {
            return "PlaneMissileEscape";
        }

        std::vector<int> velocity_idxs() override {
            return {3, 4, 5, 9, 10, 11};
        }

        std::vector<int> position_idxs() override {
            return {0, 1, 2, 6, 7, 8};
        }

        std::vector<int> my_idxs() override {
            return m_my_idxs;
        }

        int state_dim() override {
            return m_state_dim;
        }

        int not_augmented_state_dim() override {
            return m_state_dim;
        }

        int action_dim() override {
            return m_action_dim;
        }

        double dt() override {
            return m_dt;
        }

        int timestep_idx() override {
            return m_state_dim - 1;
        }

        int H() override {
            return m_H;
        }

        double gamma() override {
            return m_gamma;
        }

        Eigen::VectorXd initial_state() override {
            return m_x0;
        }

        Eigen::VectorXd empty_control() override {
            return 0.5 * (m_U.col(0) + m_U.col(1));
        }

        void set_xd(Eigen::VectorXd xd) override {
            assert(xd.size() == m_state_dim);
            m_xd = xd;
            m_target_position = m_xd.segment(0, 3);
            m_default_target_position = m_target_position;
        }

        void set_x0(Eigen::VectorXd x0) override {
            assert(x0.size() == m_state_dim);
            m_x0 = x0;
        }

        void set_dt(double dt) override {
            m_dt = dt;
        }

        Eigen::Matrix<double, -1, 2> X() override {
            return m_X;
        }

        Eigen::Matrix<double, -1, 2> U() override {
            return m_U;
        }

        bool is_state_valid(const Eigen::VectorXd &state) override {
            if (!is_vec_in_cube(state, m_X)) {
                return false;
            }

            const Eigen::Vector3d plane_position = state.segment(0, 3);
            for (const Eigen::MatrixXd &obstacle : m_obstacles) {
                if (obstacle.rows() < 3 || obstacle.cols() != 2) {
                    continue;
                }
                if (is_vec_in_cube(plane_position, obstacle.block(0, 0, 3, 2))) {
                    return false;
                }
            }

            return plane_missile_distance(state) > m_capture_radius;
        }

        void add_obstacle(Eigen::Matrix<double, -1, 2> obstacle) override {
            m_obstacles.push_back(obstacle);
        }

        void clear_obstacles() override {
            m_obstacles.clear();
        }

        void add_target(Eigen::VectorXd target) override {
            if (target.size() >= 3) {
                m_target_position = target.segment(0, 3);
            }
        }

        void clear_targets() override {
            m_target_position = m_default_target_position;
        }

        void add_thermal(Eigen::MatrixXd X_thermal, Eigen::VectorXd V_thermal) override {
            m_thermals_bounds.push_back(X_thermal);
            m_thermals_forces.push_back(V_thermal);
        }

        void clear_thermals() override {
            m_thermals_bounds.clear();
            m_thermals_forces.clear();
        }

        void add_traj(Trajectory traj) override {
            m_trajs.push_back(traj);
        }

        void set_trajs(std::vector<Trajectory> trajs) override {
            m_trajs = trajs;
        }

        Eigen::VectorXd F(const Eigen::VectorXd &state, const Eigen::VectorXd &action) override {
            if (state.size() == m_state_dim) {
                Eigen::VectorXd next_state = state;
                Eigen::VectorXd state_wo_time = state.head(m_state_dim - 1);
                const Eigen::VectorXd clipped_action = clip_action(action);
                for (int ii = 0; ii < m_control_hold; ++ii) {
                    state_wo_time = rk4_step(state_wo_time, clipped_action, m_dt);
                    project_state_in_place(state_wo_time);
                }
                next_state.head(m_state_dim - 1) = state_wo_time;
                next_state(timestep_idx()) = state(timestep_idx()) + 1.0;
                return next_state;
            }

            if (state.size() == m_state_dim - 1) {
                Eigen::VectorXd next_state = state;
                const Eigen::VectorXd clipped_action = clip_action(action);
                for (int ii = 0; ii < m_control_hold; ++ii) {
                    next_state = rk4_step(next_state, clipped_action, m_dt);
                    project_state_in_place(next_state);
                }
                return next_state;
            }

            throw std::logic_error("PlaneMissileEscape::F unexpected state dimension");
        }

        Eigen::MatrixXd B(const Eigen::VectorXd &state) override {
            return Eigen::MatrixXd::Zero(state.size(), m_action_dim);
        }

        Eigen::VectorXd F_non_augmented(const Eigen::VectorXd &state, const Eigen::VectorXd &action) override {
            return F(state, action);
        }

        double R(const Eigen::VectorXd &state, const Eigen::VectorXd &action) override {
            return R_verbose(state, action, false);
        }

        double R_verbose(const Eigen::VectorXd &state, const Eigen::VectorXd &action, bool verbose) override {
            const Eigen::Vector3d plane_position = state.segment(0, 3);
            const Eigen::Vector3d to_target = m_target_position.head(3) - plane_position;
            const double target_distance = to_target.norm();
            const double missile_distance = plane_missile_distance(state);

            double reward_alive = 1.0;
            double reward_target = std::exp(-target_distance / std::max(m_target_distance_scale, 1.0));
            double reward_escape = std::tanh(missile_distance / std::max(m_missile_distance_scale, 1.0));

            double reward_heading = 0.0;
            if (target_distance > 1.0e-6) {
                const Eigen::Vector3d plane_forward = aircraft_velocity_direction(state.head(m_state_dim - 1));
                reward_heading = 0.5 * (1.0 + plane_forward.dot(to_target / target_distance));
            }

            const Eigen::VectorXd clipped_action = clip_action(action);
            const double reward_control =
                -(clipped_action.transpose() * m_Qu * clipped_action).coeff(0, 0);

            double reward = 0.0;
            reward += m_reward_alive_weight * reward_alive;
            reward += m_reward_target_weight * reward_target;
            reward += m_reward_escape_weight * reward_escape;
            reward += m_reward_heading_weight * reward_heading;
            reward += m_reward_control_weight * reward_control;

            if (target_distance < m_target_success_radius) {
                reward += m_target_success_bonus;
            }
            if (missile_distance < m_capture_radius) {
                reward -= m_missile_hit_penalty;
            }

            if (verbose) {
                std::cout << "reward_alive: " << reward_alive << std::endl;
                std::cout << "reward_target: " << reward_target << std::endl;
                std::cout << "reward_escape: " << reward_escape << std::endl;
                std::cout << "reward_heading: " << reward_heading << std::endl;
                std::cout << "reward_control: " << reward_control << std::endl;
                std::cout << "target_distance: " << target_distance << std::endl;
                std::cout << "missile_distance: " << missile_distance << std::endl;
                std::cout << "reward_total: " << reward << std::endl;
            }

            return reward;
        }

        double V(Eigen::VectorXd state) override {
            return R(state, empty_control()) + m_V_alive;
        }

        double V(Eigen::VectorXd state, RNG& rng) override {
            (void) rng;
            return V(state);
        }

        Eigen::MatrixXd dFdx(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return F(x, action);
            };
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(state, f, fjac, fd::SECOND, 1.0e-5);
            return fjac;
        }

        Eigen::MatrixXd dFdu(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> Eigen::VectorXd {
                return F(state, u);
            };
            Eigen::MatrixXd fjac;
            fd::finite_jacobian(action, f, fjac, fd::SECOND, 1.0e-5);
            return fjac;
        }

        Eigen::VectorXd dRdx(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> double {
                return R(x, action);
            };
            Eigen::VectorXd fgrad;
            fd::finite_gradient(state, f, fgrad, fd::SECOND, 1.0e-5);
            return fgrad;
        }

        Eigen::VectorXd dRdu(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> double {
                return R(state, u);
            };
            Eigen::VectorXd fgrad;
            fd::finite_gradient(action, f, fgrad, fd::SECOND, 1.0e-5);
            return fgrad;
        }

        Eigen::MatrixXd d2Rdx2(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& x) -> double {
                return R(x, action);
            };
            Eigen::MatrixXd fhess;
            fd::finite_hessian(state, f, fhess, fd::SECOND, 1.0e-5);
            return fhess;
        }

        Eigen::MatrixXd d2Rdx2_inv(Eigen::VectorXd state, Eigen::VectorXd action) override {
            return d2Rdx2(state, action).completeOrthogonalDecomposition().pseudoInverse();
        }

        Eigen::MatrixXd d2Rdu2(Eigen::VectorXd state, Eigen::VectorXd action) override {
            const auto f = [&](const Eigen::VectorXd& u) -> double {
                return R(state, u);
            };
            Eigen::MatrixXd fhess;
            fd::finite_hessian(action, f, fhess, fd::SECOND, 1.0e-5);
            return fhess;
        }

        Eigen::VectorXd get_xd() override {
            return m_xd;
        }

        Eigen::MatrixXd sqrtQx() override {
            return m_Qx.cwiseSqrt();
        }

        Eigen::MatrixXd sqrtQx_equ() override {
            return m_Qx_equ.cwiseSqrt();
        }

        Eigen::MatrixXd sqrtQu() override {
            return m_Qu.cwiseSqrt();
        }

        Eigen::MatrixXd sqrtQf() override {
            return m_Qf.cwiseSqrt();
        }

        void set_weights(std::vector<Eigen::MatrixXd> weightss, std::vector<Eigen::MatrixXd> biass) override {
            (void) weightss;
            (void) biass;
        }

        Eigen::VectorXd eval_ff(const Eigen::VectorXd &state, const Eigen::VectorXd &action) override {
            return joint_dxdt(state.head(m_state_dim - 1), clip_action(action));
        }

    private:

        template<typename T>
        T node_value_or(const YAML::Node& node, const std::string& key, const T& default_value) {
            if (!node[key]) {
                return default_value;
            }
            return node[key].as<T>();
        }

        double clamp(double value, double lower, double upper) const {
            return std::max(lower, std::min(value, upper));
        }

        double wrap_angle(double angle) const {
            return std::atan2(std::sin(angle), std::cos(angle));
        }

        double safe_cosine(double angle) const {
            const double c = std::cos(angle);
            if (std::abs(c) >= m_aircraft_min_cos_pitch) {
                return c;
            }
            return (c >= 0.0 ? 1.0 : -1.0) * m_aircraft_min_cos_pitch;
        }

        double atmosphere_density(double altitude_magnitude) const {
            const double altitude_km = altitude_magnitude / 1000.0;
            constexpr double T0 = 288.15;
            constexpr double rho0 = 1.225;
            if (altitude_km < 11.0) {
                const double temperature = T0 - 6.5 * altitude_km;
                return rho0 * std::pow(temperature / T0, 4.256);
            }
            return 0.0;
        }

        Eigen::VectorXd clip_action(const Eigen::VectorXd &action) const {
            Eigen::VectorXd clipped = action;
            for (int ii = 0; ii < m_action_dim; ++ii) {
                clipped(ii) = clamp(action(ii), m_U(ii, 0), m_U(ii, 1));
            }
            return clipped;
        }

        Eigen::Vector3d aircraft_velocity_direction(const Eigen::VectorXd &state_wo_time) const {
            const double theta = state_wo_time(4);
            const double psi = state_wo_time(5);
            Eigen::Vector3d direction;
            direction << std::cos(theta) * std::cos(psi),
                         std::cos(theta) * std::sin(psi),
                        -std::sin(theta);
            return direction.normalized();
        }

        Eigen::Vector3d aircraft_velocity_vector(const Eigen::VectorXd &state_wo_time) const {
            return state_wo_time(3) * aircraft_velocity_direction(state_wo_time);
        }

        Eigen::Vector3d missile_velocity_direction(const Eigen::VectorXd &state_wo_time) const {
            const double theta = state_wo_time(10);
            const double psi = state_wo_time(11);
            Eigen::Vector3d direction;
            direction << std::cos(theta) * std::cos(psi),
                         std::cos(theta) * std::sin(psi),
                        -std::sin(theta);
            return direction.normalized();
        }

        Eigen::Vector3d missile_velocity_vector(const Eigen::VectorXd &state_wo_time) const {
            return state_wo_time(9) * missile_velocity_direction(state_wo_time);
        }

        Eigen::VectorXd aircraft_dxdt(const Eigen::VectorXd &state_wo_time, const Eigen::VectorXd &action) const {
            const double z = state_wo_time(2);
            const double speed = std::max(state_wo_time(3), m_aircraft_min_speed);
            const double theta = state_wo_time(4);
            const double psi = state_wo_time(5);

            const double alpha = action(0);
            const double gamma_cmd = action(1);
            const double throttle = action(2);

            const double rho = atmosphere_density(std::abs(z));
            const double lift_like = m_aircraft_lift_bias + m_aircraft_lift_alpha * alpha;
            const double normal_like = m_aircraft_normal_bias + m_aircraft_normal_alpha * alpha;
            const double CL = lift_like * std::sin(alpha) + normal_like * std::cos(alpha);
            const double CD = -lift_like * std::cos(alpha) + normal_like * std::sin(alpha);

            const double lift = 0.5 * rho * speed * speed * CL * m_aircraft_ref_area;
            const double drag = 0.5 * rho * speed * speed * CD * m_aircraft_ref_area;
            const double thrust = m_aircraft_base_thrust + throttle * m_aircraft_delta_thrust;

            Eigen::VectorXd dxdt(6);
            dxdt(0) = speed * std::cos(theta) * std::cos(psi);
            dxdt(1) = speed * std::cos(theta) * std::sin(psi);
            dxdt(2) = -speed * std::sin(theta);
            dxdt(3) = (thrust * std::cos(alpha) - drag) / m_aircraft_mass - m_gravity * std::sin(theta);
            dxdt(4) = (lift + thrust * std::sin(alpha)) * std::cos(gamma_cmd) / (m_aircraft_mass * speed)
                    - m_gravity * std::cos(theta) / speed;
            dxdt(5) = (lift + thrust * std::sin(alpha)) * std::sin(gamma_cmd)
                    / (m_aircraft_mass * speed * safe_cosine(theta));
            return dxdt;
        }

        Eigen::VectorXd missile_dxdt(const Eigen::VectorXd &state_wo_time) const {
            const Eigen::Vector3d plane_position = state_wo_time.segment(0, 3);
            const Eigen::Vector3d plane_velocity = aircraft_velocity_vector(state_wo_time);
            const Eigen::Vector3d missile_position = state_wo_time.segment(6, 3);
            const Eigen::Vector3d missile_velocity = missile_velocity_vector(state_wo_time);

            const Eigen::Vector3d relative_position = plane_position - missile_position;
            const Eigen::Vector3d relative_velocity = plane_velocity - missile_velocity;

            Eigen::Vector3d desired_direction = relative_position + m_missile_lead_gain * relative_velocity;
            if (desired_direction.norm() < 1.0e-6) {
                desired_direction = relative_position;
            }
            if (desired_direction.norm() < 1.0e-6) {
                desired_direction = missile_velocity_direction(state_wo_time);
            }
            desired_direction.normalize();

            const double missile_speed = std::max(state_wo_time(9), m_missile_min_speed);
            const double missile_theta = state_wo_time(10);
            const double missile_psi = state_wo_time(11);

            const double desired_theta = std::atan2(-desired_direction.z(),
                std::sqrt(square(desired_direction.x()) + square(desired_direction.y())));
            const double desired_psi = std::atan2(desired_direction.y(), desired_direction.x());

            Eigen::VectorXd dxdt(6);
            dxdt(0) = missile_speed * std::cos(missile_theta) * std::cos(missile_psi);
            dxdt(1) = missile_speed * std::cos(missile_theta) * std::sin(missile_psi);
            dxdt(2) = -missile_speed * std::sin(missile_theta);
            dxdt(3) = m_missile_speed_response * (m_missile_speed_command - missile_speed);
            dxdt(4) = clamp(m_missile_turn_gain * wrap_angle(desired_theta - missile_theta),
                -m_missile_max_pitch_rate, m_missile_max_pitch_rate);
            dxdt(5) = clamp(m_missile_turn_gain * wrap_angle(desired_psi - missile_psi),
                -m_missile_max_yaw_rate, m_missile_max_yaw_rate);
            return dxdt;
        }

        Eigen::VectorXd joint_dxdt(const Eigen::VectorXd &state_wo_time, const Eigen::VectorXd &action) const {
            Eigen::VectorXd dxdt(12);
            dxdt.segment(0, 6) = aircraft_dxdt(state_wo_time, action);
            dxdt.segment(6, 6) = missile_dxdt(state_wo_time);
            return dxdt;
        }

        Eigen::VectorXd rk4_step(const Eigen::VectorXd &state_wo_time, const Eigen::VectorXd &action, double step) const {
            const Eigen::VectorXd k1 = joint_dxdt(state_wo_time, action);
            const Eigen::VectorXd k2 = joint_dxdt(state_wo_time + 0.5 * step * k1, action);
            const Eigen::VectorXd k3 = joint_dxdt(state_wo_time + 0.5 * step * k2, action);
            const Eigen::VectorXd k4 = joint_dxdt(state_wo_time + step * k3, action);
            return state_wo_time + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        }

        void project_state_in_place(Eigen::VectorXd &state_wo_time) const {
            state_wo_time(3) = clamp(state_wo_time(3), m_aircraft_min_speed, m_aircraft_max_speed);
            state_wo_time(4) = clamp(state_wo_time(4), -m_aircraft_max_pitch, m_aircraft_max_pitch);
            state_wo_time(5) = wrap_angle(state_wo_time(5));

            state_wo_time(9) = clamp(state_wo_time(9), m_missile_min_speed, m_missile_max_speed);
            state_wo_time(10) = clamp(state_wo_time(10), -m_missile_max_pitch, m_missile_max_pitch);
            state_wo_time(11) = wrap_angle(state_wo_time(11));
        }

        double plane_missile_distance(const Eigen::VectorXd &state) const {
            const Eigen::Vector3d plane_position = state.segment(0, 3);
            const Eigen::Vector3d missile_position = state.segment(6, 3);
            return (plane_position - missile_position).norm();
        }

        Eigen::VectorXd m_x0;
        Eigen::VectorXd m_xd;
        Eigen::VectorXd m_target_position;
        Eigen::VectorXd m_default_target_position;
        Eigen::Matrix<double, -1, 2> m_X;
        Eigen::Matrix<double, -1, 2> m_U;
        Eigen::MatrixXd m_Qx;
        Eigen::MatrixXd m_Qx_equ;
        Eigen::MatrixXd m_Qu;
        Eigen::MatrixXd m_Qf;

        bool m_verbose = false;
        double m_dt = 0.05;
        double m_gamma = 1.0;
        double m_V_alive = 0.0;
        int m_H = 1;
        int m_state_dim = 13;
        int m_action_dim = 3;
        int m_control_hold = 1;

        double m_reward_alive_weight = 0.05;
        double m_reward_target_weight = 0.45;
        double m_reward_escape_weight = 0.40;
        double m_reward_heading_weight = 0.10;
        double m_reward_control_weight = 0.0;
        double m_target_distance_scale = 2000.0;
        double m_missile_distance_scale = 1200.0;
        double m_target_success_radius = 150.0;
        double m_target_success_bonus = 1.0;
        double m_missile_hit_penalty = 2.0;
        double m_capture_radius = 80.0;

        double m_aircraft_mass = 13680.0;
        double m_gravity = 9.8;
        double m_aircraft_ref_area = 49.24;
        double m_aircraft_base_thrust = 80000.0;
        double m_aircraft_delta_thrust = 70000.0;
        double m_aircraft_min_speed = 150.0;
        double m_aircraft_max_speed = 1200.0;
        double m_aircraft_max_pitch = 1.2;
        double m_aircraft_min_cos_pitch = 0.05;
        double m_aircraft_lift_bias = -0.0434;
        double m_aircraft_lift_alpha = 0.1369;
        double m_aircraft_normal_bias = 0.1310;
        double m_aircraft_normal_alpha = 3.0825;

        double m_missile_speed_command = 650.0;
        double m_missile_speed_response = 1.5;
        double m_missile_min_speed = 250.0;
        double m_missile_max_speed = 1400.0;
        double m_missile_turn_gain = 2.5;
        double m_missile_max_pitch_rate = 1.0;
        double m_missile_max_yaw_rate = 1.2;
        double m_missile_max_pitch = 1.3;
        double m_missile_lead_gain = 0.75;

        std::vector<int> m_my_idxs;
        std::vector<Eigen::MatrixXd> m_thermals_bounds;
        std::vector<Eigen::VectorXd> m_thermals_forces;
        std::vector<Trajectory> m_trajs;
};
