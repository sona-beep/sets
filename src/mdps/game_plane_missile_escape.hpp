#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "mdp.hpp"
#include "plane_model.hpp"
#include "missile_model.hpp"
#include "../util/util.hpp"

class GamePlaneMissileEscape : public MDP {

    public:

        explicit GamePlaneMissileEscape(std::string config_path)
            : m_config_path(std::move(config_path)),
              m_plane_model(YAML::LoadFile(m_config_path)),
              m_missile_model(YAML::LoadFile(m_config_path)) {

            YAML::Node config = YAML::LoadFile(m_config_path);

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
            m_reward_danger_weight = node_value_or<double>(config, "reward_danger_weight", 0.30);
            m_reward_heading_weight = node_value_or<double>(config, "reward_heading_weight", 0.10);
            m_reward_precision_weight = node_value_or<double>(config, "reward_precision_weight", 0.0);
            m_reward_control_weight = node_value_or<double>(config, "reward_control_weight", 0.0);
            m_target_distance_scale = node_value_or<double>(config, "target_distance_scale", 2000.0);
            m_target_precision_scale = node_value_or<double>(config, "target_precision_scale", 0.0);
            m_missile_distance_scale = node_value_or<double>(config, "missile_distance_scale", 1200.0);
            m_missile_danger_scale = node_value_or<double>(config, "missile_danger_scale", 300.0);
            m_target_success_radius = node_value_or<double>(config, "target_success_radius", 150.0);
            m_default_target_success_radius = m_target_success_radius;
            m_target_success_bonus = node_value_or<double>(config, "target_success_bonus", 1.0);
            m_missile_hit_penalty = node_value_or<double>(config, "missile_hit_penalty", 2.0);
            m_capture_radius = node_value_or<double>(config, "missile_capture_radius", 80.0);
        }

        std::string name() override {
            return "GamePlaneMissileEscape";
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

        double target_distance(const Eigen::VectorXd &state) const {
            return (state.segment(0, 3) - m_target_position.head(3)).norm();
        }

        double missile_distance(const Eigen::VectorXd &state) const {
            return (state.segment(0, 3) - state.segment(6, 3)).norm();
        }

        bool target_reached(const Eigen::VectorXd &state) const {
            return target_distance(state) <= m_target_success_radius;
        }

        bool missile_hit(const Eigen::VectorXd &state) const {
            return missile_distance(state) <= m_capture_radius;
        }

        bool out_of_bounds(const Eigen::VectorXd &state) const {
            const Eigen::MatrixXd cube = m_X.block(0, 0, state.size(), 2);
            return !is_vec_in_cube(state, cube);
        }

        bool obstacle_collision(const Eigen::VectorXd &state) const {
            const Eigen::Vector3d plane_position = state.segment(0, 3);
            for (const Eigen::MatrixXd &obstacle : m_obstacles) {
                if (obstacle.rows() < 3 || obstacle.cols() != 2) {
                    continue;
                }
                if (is_vec_in_cube(plane_position, obstacle.block(0, 0, 3, 2))) {
                    return true;
                }
            }
            return false;
        }

        bool is_state_valid(const Eigen::VectorXd &state) override {
            return !out_of_bounds(state) && !obstacle_collision(state) && !missile_hit(state);
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
                m_xd.segment(0, 3) = m_target_position.head(3);
                m_target_success_radius = target.size() >= 4
                    ? std::max(target(3), 1.0)
                    : m_default_target_success_radius;
            }
        }

        void clear_targets() override {
            m_target_position = m_default_target_position;
            m_target_success_radius = m_default_target_success_radius;
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
            Eigen::VectorXd state_without_time;
            if (state.size() == m_state_dim) {
                state_without_time = state.head(m_state_dim - 1);
            } else if (state.size() == m_state_dim - 1) {
                state_without_time = state;
            } else {
                throw std::logic_error("GamePlaneMissileEscape::F unexpected state dimension");
            }

            const Eigen::VectorXd clipped_action = clip_action(action);
            for (int ii = 0; ii < m_control_hold; ++ii) {
                state_without_time = rk4_step(state_without_time, clipped_action, m_dt);
                project_joint_state(state_without_time);
            }

            if (state.size() == m_state_dim) {
                Eigen::VectorXd next_state = state;
                next_state.head(m_state_dim - 1) = state_without_time;
                next_state(timestep_idx()) = state(timestep_idx()) + 1.0;
                return next_state;
            }

            return state_without_time;
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
            const double distance_to_target = target_distance(state);
            const double distance_to_missile = missile_distance(state);
            const double missile_margin = std::max(distance_to_missile - m_capture_radius, 0.0);

            const double reward_alive = 1.0;
            const double reward_target =
                std::exp(-distance_to_target / std::max(m_target_distance_scale, 1.0));
            const double precision_scale = target_precision_scale();
            const double reward_precision =
                std::exp(-std::pow(distance_to_target / precision_scale, 2.0));
            const double reward_escape =
                std::tanh(missile_margin / std::max(m_missile_distance_scale, 1.0));
            const double reward_danger =
                std::exp(-missile_margin / std::max(m_missile_danger_scale, 1.0));

            double reward_heading = 0.0;
            if (distance_to_target > 1.0e-6) {
                const PlaneState plane_state = PlaneState::from_joint_state(state);
                const Eigen::Vector3d heading = plane_state.direction();
                const Eigen::Vector3d to_target = (m_target_position.head(3) - plane_state.position()) / distance_to_target;
                reward_heading = 0.5 * (1.0 + heading.dot(to_target));
            }

            const Eigen::VectorXd clipped_action = clip_action(action);
            const double reward_control =
                -(clipped_action.transpose() * m_Qu * clipped_action).coeff(0, 0);

            double reward = 0.0;
            reward += m_reward_alive_weight * reward_alive;
            reward += m_reward_target_weight * reward_target;
            reward += m_reward_precision_weight * reward_precision;
            reward += m_reward_escape_weight * reward_escape;
            reward -= m_reward_danger_weight * reward_danger;
            reward += m_reward_heading_weight * reward_heading;
            reward += m_reward_control_weight * reward_control;

            if (target_reached(state)) {
                reward += m_target_success_bonus;
            }
            if (missile_hit(state)) {
                reward -= m_missile_hit_penalty;
            }

            if (verbose) {
                std::cout << "reward_alive: " << reward_alive << std::endl;
                std::cout << "reward_target: " << reward_target << std::endl;
                std::cout << "reward_precision: " << reward_precision << std::endl;
                std::cout << "reward_escape: " << reward_escape << std::endl;
                std::cout << "reward_danger: " << reward_danger << std::endl;
                std::cout << "reward_heading: " << reward_heading << std::endl;
                std::cout << "reward_control: " << reward_control << std::endl;
                std::cout << "target_distance: " << distance_to_target << std::endl;
                std::cout << "missile_distance: " << distance_to_missile << std::endl;
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
        T node_value_or(const YAML::Node& node, const std::string& key, const T& default_value) const {
            if (!node[key]) {
                return default_value;
            }
            return node[key].as<T>();
        }

        static double clamp_value(double value, double lower, double upper) {
            return std::max(lower, std::min(value, upper));
        }

        double target_precision_scale() const {
            if (m_target_precision_scale > 0.0) {
                return std::max(m_target_precision_scale, 1.0);
            }
            return std::max(m_target_success_radius, 1.0);
        }

        Eigen::VectorXd clip_action(const Eigen::VectorXd &action) const {
            Eigen::VectorXd clipped = action;
            for (int ii = 0; ii < m_action_dim; ++ii) {
                clipped(ii) = clamp_value(action(ii), m_U(ii, 0), m_U(ii, 1));
            }
            return clipped;
        }

        Eigen::VectorXd joint_dxdt(const Eigen::VectorXd &joint_state, const Eigen::VectorXd &action) const {
            assert(joint_state.size() == m_state_dim - 1);
            PlaneState plane_state = PlaneState::from_joint_state(joint_state);
            MissileState missile_state = MissileState::from_joint_state(joint_state);
            const PlaneControl control = PlaneControl::from_action(action);

            const PlaneState plane_dxdt = m_plane_model.derivative(plane_state, control);
            const MissileState missile_dxdt = m_missile_model.derivative(missile_state, plane_state);

            Eigen::VectorXd dxdt = Eigen::VectorXd::Zero(m_state_dim - 1);
            plane_dxdt.write_to_joint_state(dxdt);
            missile_dxdt.write_to_joint_state(dxdt);
            return dxdt;
        }

        Eigen::VectorXd rk4_step(const Eigen::VectorXd &joint_state, const Eigen::VectorXd &action, double step) const {
            const Eigen::VectorXd k1 = joint_dxdt(joint_state, action);
            const Eigen::VectorXd k2 = joint_dxdt(joint_state + 0.5 * step * k1, action);
            const Eigen::VectorXd k3 = joint_dxdt(joint_state + 0.5 * step * k2, action);
            const Eigen::VectorXd k4 = joint_dxdt(joint_state + step * k3, action);
            return joint_state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        }

        void project_joint_state(Eigen::VectorXd &joint_state) const {
            PlaneState plane_state = PlaneState::from_joint_state(joint_state);
            MissileState missile_state = MissileState::from_joint_state(joint_state);
            m_plane_model.project_state(plane_state);
            m_missile_model.project_state(missile_state);
            plane_state.write_to_joint_state(joint_state);
            missile_state.write_to_joint_state(joint_state);
        }

        std::string m_config_path;
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

        PlaneModel m_plane_model;
        MissileModel m_missile_model;

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
        double m_reward_danger_weight = 0.30;
        double m_reward_heading_weight = 0.10;
        double m_reward_precision_weight = 0.0;
        double m_reward_control_weight = 0.0;
        double m_target_distance_scale = 2000.0;
        double m_target_precision_scale = 0.0;
        double m_missile_distance_scale = 1200.0;
        double m_missile_danger_scale = 300.0;
        double m_target_success_radius = 150.0;
        double m_default_target_success_radius = 150.0;
        double m_target_success_bonus = 1.0;
        double m_missile_hit_penalty = 2.0;
        double m_capture_radius = 80.0;

        std::vector<int> m_my_idxs;
        std::vector<Eigen::MatrixXd> m_thermals_bounds;
        std::vector<Eigen::VectorXd> m_thermals_forces;
        std::vector<Trajectory> m_trajs;
};
