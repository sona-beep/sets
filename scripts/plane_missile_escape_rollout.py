from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from util import util
from build.bindings import get_dots_mdp, get_mdp, get_uct2, RNG, run_uct2


def build_dots(config_dict, ground_mdp):
    dots_mdp = get_dots_mdp()
    dots_mdp.set_param(
        ground_mdp,
        config_dict["dots_expansion_mode"],
        config_dict["dots_initialize_mode"],
        config_dict["dots_special_actions"],
        config_dict["dots_num_branches"],
        config_dict["dots_decision_making_horizon"],
        config_dict["dots_dynamics_horizon"],
        config_dict["dots_spectral_branches_mode"],
        config_dict["dots_control_mode"],
        config_dict["dots_scale_mode"],
        config_dict["dots_modal_damping_mode"],
        config_dict["dots_modal_damping_gains"],
        np.diag(np.array(config_dict["dots_rho"], dtype=float)),
        config_dict["dots_greedy_gain"],
        config_dict["dots_greedy_rate"],
        config_dict["dots_greedy_min_dist"],
        config_dict["dots_baseline_mode_on"],
        config_dict["dots_num_discrete_actions"],
        config_dict["dots_verbose"],
    )
    return dots_mdp


def build_uct(config_dict):
    uct = get_uct2()
    uct.set_param(
        config_dict["uct_N"],
        config_dict["uct_max_depth"],
        config_dict["uct_wct"],
        config_dict["uct_c"],
        config_dict["uct_export_topology"],
        config_dict["uct_export_node_states"],
        config_dict["uct_export_trajs"],
        config_dict["uct_export_cbdsbds"],
        config_dict["uct_export_tree_statistics"],
        config_dict["uct_heuristic_mode"],
        config_dict["uct_tree_exploration"],
        config_dict["uct_downsample_traj_on"],
        config_dict["uct_verbose"],
    )
    return uct


def plane_position(state):
    return np.array(state[0:3], dtype=float)


def missile_position(state):
    return np.array(state[6:9], dtype=float)


def plane_target_distance(state, target):
    return float(np.linalg.norm(plane_position(state) - np.array(target, dtype=float)))


def plane_missile_distance(state):
    return float(np.linalg.norm(plane_position(state) - missile_position(state)))


def direction_to_target(state, target):
    delta = np.array(target, dtype=float) - plane_position(state)
    norm = np.linalg.norm(delta)
    if norm < 1.0e-9:
        return np.zeros(3)
    return delta / norm


def state_limits(config_dict):
    state_dim = len(config_dict["ground_mdp_x0"])
    return np.array(config_dict["ground_mdp_X"], dtype=float).reshape((state_dim, 2), order="F")


def state_out_of_bounds(state, config_dict):
    limits = state_limits(config_dict)
    return not (((state >= limits[:, 0]) & (state <= limits[:, 1])).all())


def controls_from_traj(traj):
    controls = np.array(traj.us, dtype=float)
    if controls.ndim != 2 or controls.shape[0] == 0:
        return None
    if not np.isfinite(controls).all():
        return None
    return controls


def choose_execution_controls(result):
    mpc_controls = controls_from_traj(result.mpc_traj)
    if mpc_controls is not None:
        return mpc_controls
    return controls_from_traj(result.planned_traj)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


@dataclass
class MissionTarget:
    name: str
    position: np.ndarray
    radius: float

    @classmethod
    def from_config(cls, raw_target, default_radius, idx):
        if isinstance(raw_target, dict):
            name = raw_target.get("name", f"target_{idx + 1}")
            position = np.array(raw_target["position"], dtype=float)
            radius = float(raw_target.get("radius", default_radius))
        else:
            name = f"target_{idx + 1}"
            position = np.array(raw_target, dtype=float)
            radius = float(default_radius)
        if position.shape != (3,):
            raise ValueError(f"{name} must have a 3D position")
        return cls(name=name, position=position, radius=radius)


@dataclass
class ObstacleBox:
    name: str
    bounds: np.ndarray

    @classmethod
    def from_config(cls, raw_obstacle, idx):
        if isinstance(raw_obstacle, dict):
            name = raw_obstacle.get("name", f"obstacle_{idx + 1}")
            raw_bounds = raw_obstacle["bounds"]
        else:
            name = f"obstacle_{idx + 1}"
            raw_bounds = raw_obstacle
        bounds_flat = np.array(raw_bounds, dtype=float)
        if bounds_flat.shape == (3, 2):
            bounds = bounds_flat
        elif bounds_flat.size == 6:
            bounds = bounds_flat.reshape((3, 2))
        else:
            raise ValueError(f"{name} bounds must be [xmin,xmax,ymin,ymax,zmin,zmax]")
        return cls(name=name, bounds=bounds)

    def as_mdp_obstacle(self, config_dict):
        obstacle = state_limits(config_dict).copy()
        obstacle[0:3, :] = self.bounds
        return obstacle

    def contains_plane(self, state):
        plane = plane_position(state)
        return bool(((plane >= self.bounds[:, 0]) & (plane <= self.bounds[:, 1])).all())


class PlaneMissileEscapeRunner:
    def __init__(self, config_dict, config_path, seed=0, make_plots=True):
        self.config = config_dict
        self.config_path = config_path
        self.seed = seed
        self.make_plots = make_plots

        self.rng = RNG()
        self.rng.set_seed(seed)

        self.targets = self._load_targets()
        self.obstacles = self._load_obstacles()
        self.active_target_idx = 0
        self.control_step_duration = (
            self.config["uct_dt"] * self.config["ground_mdp_control_hold"]
        )
        self.output_dir = self._make_output_dir()

        self.ground_mdp = get_mdp(self.config["ground_mdp_name"], self.config_path)
        self.ground_mdp.set_dt(self.config["uct_dt"])
        self.dots_mdp = build_dots(self.config, self.ground_mdp)
        self.uct = build_uct(self.config)
        self._set_active_target()

    def _load_targets(self):
        raw_targets = self.config.get("mission_targets")
        if raw_targets is None:
            raw_targets = [{"name": "target_1", "position": self.config["target_position"]}]
        return [
            MissionTarget.from_config(
                raw_target, self.config["target_success_radius"], idx
            )
            for idx, raw_target in enumerate(raw_targets)
        ]

    def _load_obstacles(self):
        raw_obstacles = self.config.get("obstacle_boxes", [])
        return [
            ObstacleBox.from_config(raw_obstacle, idx)
            for idx, raw_obstacle in enumerate(raw_obstacles)
        ]

    def _make_output_dir(self):
        output_root = Path(self.config.get("output_dir", ROOT / "results" / "plane_missile_escape"))
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"{self.config['config_name']}_seed{self.seed}_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @property
    def active_target(self):
        return self.targets[self.active_target_idx]

    def _set_active_target(self):
        self.ground_mdp.clear_targets()
        self.ground_mdp.add_target(self.active_target.position)

    def update_environment(self, curr_state):
        timestep = curr_state[self.ground_mdp.timestep_idx()]
        self.ground_mdp.clear_obstacles()
        for obstacle in self.active_obstacles(timestep):
            self.ground_mdp.add_obstacle(obstacle.as_mdp_obstacle(self.config))

        self.ground_mdp.clear_thermals()
        for thermal_bound, thermal_force_moment in util.get_thermals(self.config, timestep):
            self.ground_mdp.add_thermal(thermal_bound, thermal_force_moment)

    def active_obstacles(self, timestep):
        _ = timestep
        return self.obstacles

    def obstacle_collision(self, state):
        return any(obstacle.contains_plane(state) for obstacle in self.obstacles)

    def target_reached(self, state):
        return plane_target_distance(state, self.active_target.position) <= self.active_target.radius

    def classify_failure(self, state):
        if plane_missile_distance(state) <= self.config["missile_capture_radius"]:
            return "missile_hit"
        if self.obstacle_collision(state):
            return "obstacle_collision"
        if state_out_of_bounds(state, self.config):
            return "out_of_bounds"
        return None

    def maybe_update_min_distance(self, summary, state):
        distance = plane_missile_distance(state)
        if distance < summary["min_distance_to_missile"]:
            summary["min_distance_to_missile"] = distance
            summary["min_distance_state_idx"] = len(summary["rollout_xs"]) - 1
            summary["closest_approach_time"] = (
                summary["min_distance_state_idx"] * self.control_step_duration
            )

    def mark_target_reached(self, summary, step_idx, state):
        reached = {
            "name": self.active_target.name,
            "index": self.active_target_idx,
            "position": self.active_target.position.copy(),
            "step": step_idx,
            "time": step_idx * self.control_step_duration,
            "plane_position": plane_position(state),
        }
        summary["reached_targets"].append(reached)

        if self.active_target_idx == len(self.targets) - 1:
            summary["termination_reason"] = "all_targets_reached"
            return

        self.active_target_idx += 1
        self._set_active_target()
        summary["target_switch_steps"].append(step_idx)

    def record_planner_snapshot(self, summary, result, planner_call, curr_state):
        stride = int(self.config.get("branch_snapshot_stride", 4))
        max_snapshots = int(self.config.get("max_branch_snapshots", 5))
        if stride <= 0 or planner_call % stride != 0:
            return
        if len(summary["planner_snapshots"]) >= max_snapshots:
            return

        max_trajs = int(self.config.get("max_branch_trajs_to_store", 80))
        branch_trajs = []
        for traj in list(result.tree.trajs)[:max_trajs]:
            xs = np.array(traj.xs, dtype=float)
            if xs.ndim == 2 and xs.shape[0] > 0:
                branch_trajs.append(xs.copy())

        snapshot = {
            "planner_call": planner_call,
            "root_state": np.array(curr_state, dtype=float).copy(),
            "target_index": self.active_target_idx,
            "target_name": self.active_target.name,
            "branch_trajs": branch_trajs,
        }
        summary["planner_snapshots"].append(snapshot)

    def print_step_summary(self, step_idx, planner_call, control, reward, next_state, result):
        print(
            "step={:03d} planner_call={:03d} target={}/{} tree_rollouts={} "
            "reward={:.4f} d_target={:.2f} d_missile={:.2f} control={}".format(
                step_idx,
                planner_call,
                self.active_target_idx + 1,
                len(self.targets),
                len(result.vs),
                reward,
                plane_target_distance(next_state, self.active_target.position),
                plane_missile_distance(next_state),
                np.round(control, 4),
            )
        )

    def run(self):
        execute_steps = int(self.config.get("closed_loop_execute_steps", 1))
        max_planner_calls = int(
            self.config.get(
                "closed_loop_max_steps",
                math.ceil(self.config["ground_mdp_H"] / float(execute_steps)),
            )
        )

        curr_state = np.array(self.ground_mdp.initial_state(), dtype=float)
        summary = {
            "termination_reason": "time_limit_reached",
            "rollout_xs": [curr_state.copy()],
            "rollout_us": [],
            "rollout_rs": [],
            "active_target_indices": [self.active_target_idx],
            "planner_values": [],
            "planner_snapshots": [],
            "planner_calls": 0,
            "targets": [target.position.copy() for target in self.targets],
            "target_names": [target.name for target in self.targets],
            "target_radii": [target.radius for target in self.targets],
            "obstacles": [obstacle.bounds.copy() for obstacle in self.obstacles],
            "obstacle_names": [obstacle.name for obstacle in self.obstacles],
            "reached_targets": [],
            "target_switch_steps": [],
            "min_distance_to_missile": plane_missile_distance(curr_state),
            "min_distance_state_idx": 0,
            "closest_approach_time": 0.0,
            "output_dir": str(self.output_dir),
        }

        initial_failure = self.classify_failure(curr_state)
        if initial_failure is not None:
            summary["termination_reason"] = initial_failure
            return self.finalize(summary)

        global_step_idx = 0
        for planner_call in range(max_planner_calls):
            self.update_environment(curr_state)
            result = run_uct2(self.dots_mdp, self.uct, curr_state, self.rng)
            summary["planner_values"].append(np.array(result.vs, dtype=float))
            summary["planner_calls"] += 1
            self.record_planner_snapshot(summary, result, planner_call, curr_state)

            planned_us = choose_execution_controls(result)
            if (not result.success) or planned_us is None:
                summary["termination_reason"] = "planner_failed"
                break

            should_replan = False
            num_to_execute = min(execute_steps, planned_us.shape[0])
            for exec_idx in range(num_to_execute):
                control = np.array(planned_us[exec_idx], dtype=float)
                if not np.isfinite(control).all():
                    summary["termination_reason"] = "planner_failed"
                    break
                next_state = np.array(self.ground_mdp.F(curr_state, control), dtype=float)
                reward = float(self.ground_mdp.R(next_state, control))

                summary["rollout_us"].append(control.copy())
                summary["rollout_rs"].append(reward)
                summary["rollout_xs"].append(next_state.copy())
                summary["active_target_indices"].append(self.active_target_idx)
                self.maybe_update_min_distance(summary, next_state)

                self.print_step_summary(
                    global_step_idx,
                    planner_call,
                    control,
                    reward,
                    next_state,
                    result,
                )

                curr_state = next_state
                global_step_idx += 1

                failure = self.classify_failure(curr_state)
                if failure is not None:
                    summary["termination_reason"] = failure
                    break

                if self.target_reached(curr_state):
                    self.mark_target_reached(summary, global_step_idx, curr_state)
                    should_replan = True
                    break

            if summary["termination_reason"] != "time_limit_reached":
                break
            if should_replan:
                continue

        return self.finalize(summary)

    def finalize(self, summary):
        summary["rollout_xs"] = np.array(summary["rollout_xs"], dtype=float)
        summary["rollout_us"] = np.array(summary["rollout_us"], dtype=float)
        summary["rollout_rs"] = np.array(summary["rollout_rs"], dtype=float)
        summary["active_target_indices"] = np.array(summary["active_target_indices"], dtype=int)
        self.save_data(summary)
        if self.make_plots and self.config.get("visualize", True):
            MissionVisualizer(self.config, summary, self.output_dir).save_all()
        return summary

    def save_data(self, summary):
        np.savez_compressed(
            self.output_dir / "rollout_data.npz",
            rollout_xs=summary["rollout_xs"],
            rollout_us=summary["rollout_us"],
            rollout_rs=summary["rollout_rs"],
            active_target_indices=summary["active_target_indices"],
            targets=np.array(summary["targets"], dtype=float),
            obstacles=np.array(summary["obstacles"], dtype=float),
        )

        final_state = summary["rollout_xs"][-1]
        final_target_idx = min(
            int(summary["active_target_indices"][-1]),
            len(summary["targets"]) - 1,
        )
        summary_payload = {
            "termination_reason": summary["termination_reason"],
            "planner_calls": summary["planner_calls"],
            "executed_steps": int(len(summary["rollout_us"])),
            "targets_reached": int(len(summary["reached_targets"])),
            "num_targets": int(len(summary["targets"])),
            "final_active_target": summary["target_names"][final_target_idx],
            "final_plane_position": plane_position(final_state),
            "final_missile_position": missile_position(final_state),
            "final_distance_to_active_target": plane_target_distance(
                final_state, summary["targets"][final_target_idx]
            ),
            "final_distance_to_missile": plane_missile_distance(final_state),
            "min_distance_to_missile": summary["min_distance_to_missile"],
            "min_distance_state_idx": summary["min_distance_state_idx"],
            "closest_approach_time": summary["closest_approach_time"],
            "cumulative_reward": float(np.sum(summary["rollout_rs"])) if len(summary["rollout_rs"]) else 0.0,
            "mean_step_reward": float(np.mean(summary["rollout_rs"])) if len(summary["rollout_rs"]) else 0.0,
            "targets": [
                {
                    "name": name,
                    "position": position,
                    "radius": radius,
                }
                for name, position, radius in zip(
                    summary["target_names"],
                    summary["targets"],
                    summary["target_radii"],
                )
            ],
            "reached_targets": summary["reached_targets"],
            "obstacle_names": summary["obstacle_names"],
            "obstacles": summary["obstacles"],
            "output_dir": summary["output_dir"],
        }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(jsonable(summary_payload), handle, indent=2)


class MissionVisualizer:
    def __init__(self, config_dict, summary, output_dir):
        self.config = config_dict
        self.summary = summary
        self.output_dir = Path(output_dir)
        self.xs = np.array(summary["rollout_xs"], dtype=float)
        self.us = np.array(summary["rollout_us"], dtype=float)
        self.rs = np.array(summary["rollout_rs"], dtype=float)
        self.targets = np.array(summary["targets"], dtype=float)
        self.target_names = summary["target_names"]
        self.target_radii = np.array(summary["target_radii"], dtype=float)
        self.obstacles = np.array(summary["obstacles"], dtype=float)
        self.dt = config_dict["uct_dt"] * config_dict["ground_mdp_control_hold"]

    def save_all(self):
        os.environ.setdefault("MPLCONFIGDIR", str(self.output_dir / "matplotlib_cache"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.use_plot_style(plt)
        self.plot_trajectory_3d(plt)
        self.plot_topdown(plt)
        self.plot_distances(plt)
        self.plot_controls(plt)
        self.plot_states(plt)
        self.plot_spectral_branches(plt)

    @staticmethod
    def use_plot_style(plt):
        for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
            try:
                plt.style.use(style)
                return
            except OSError:
                continue
        plt.style.use("default")

    def state_times(self):
        return np.arange(self.xs.shape[0]) * self.dt

    def control_times(self):
        return np.arange(self.us.shape[0]) * self.dt

    def current_target_distances(self):
        target_indices = self.summary["active_target_indices"]
        distances = []
        for idx, state in enumerate(self.xs):
            target_idx = min(int(target_indices[idx]), len(self.targets) - 1)
            distances.append(plane_target_distance(state, self.targets[target_idx]))
        return np.array(distances, dtype=float)

    def plot_trajectory_3d(self, plt):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(12, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        plane = self.xs[:, 0:3]
        missile = self.xs[:, 6:9]
        ax.plot(plane[:, 0], plane[:, 1], -plane[:, 2], color="#0B3954", linewidth=2.5, label="aircraft")
        ax.plot(missile[:, 0], missile[:, 1], -missile[:, 2], color="#B23A48", linewidth=2.0, label="missile")
        ax.scatter(plane[0, 0], plane[0, 1], -plane[0, 2], color="#0B3954", s=60, marker="o", label="aircraft start")
        ax.scatter(missile[0, 0], missile[0, 1], -missile[0, 2], color="#B23A48", s=60, marker="^", label="missile start")

        for idx, target in enumerate(self.targets):
            ax.scatter(target[0], target[1], -target[2], color="#2A9D8F", s=90, marker="*", label="targets" if idx == 0 else None)
            ax.text(target[0], target[1], -target[2], f" {idx + 1}:{self.target_names[idx]}", color="#1B6F67")

        for obstacle in self.obstacles:
            verts = self.cuboid_vertices(obstacle)
            poly = Poly3DCollection(verts, alpha=0.16, facecolor="#E76F51", edgecolor="#9C2F1B", linewidth=0.7)
            ax.add_collection3d(poly)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("altitude [m]")
        ax.set_title("Plane-Missile Escape Mission")
        ax.legend(loc="upper left")
        self.equalize_3d_axes(ax, plane, missile)
        fig.tight_layout()
        fig.savefig(self.output_dir / "trajectory_3d.png")
        plt.close(fig)

    def plot_topdown(self, plt):
        fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
        plane = self.xs[:, 0:3]
        missile = self.xs[:, 6:9]
        ax.plot(plane[:, 0], plane[:, 1], color="#0B3954", linewidth=2.5, label="aircraft")
        ax.plot(missile[:, 0], missile[:, 1], color="#B23A48", linewidth=2.0, label="missile")
        ax.scatter(plane[0, 0], plane[0, 1], color="#0B3954", s=55)
        ax.scatter(missile[0, 0], missile[0, 1], color="#B23A48", s=55, marker="^")

        for idx, target in enumerate(self.targets):
            circle = plt.Circle(
                (target[0], target[1]),
                self.target_radii[idx],
                color="#2A9D8F",
                fill=False,
                linewidth=1.5,
                alpha=0.8,
            )
            ax.add_patch(circle)
            ax.scatter(target[0], target[1], color="#2A9D8F", s=85, marker="*")
            ax.text(target[0], target[1], f" {idx + 1}:{self.target_names[idx]}", color="#1B6F67")

        for obstacle in self.obstacles:
            width = obstacle[0, 1] - obstacle[0, 0]
            height = obstacle[1, 1] - obstacle[1, 0]
            rect = plt.Rectangle(
                (obstacle[0, 0], obstacle[1, 0]),
                width,
                height,
                facecolor="#E76F51",
                edgecolor="#9C2F1B",
                alpha=0.22,
                linewidth=1.2,
            )
            ax.add_patch(rect)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Top-Down Mission Geometry")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_dir / "trajectory_topdown.png")
        plt.close(fig)

    def plot_distances(self, plt):
        times = self.state_times()
        d_target = self.current_target_distances()
        d_missile = np.array([plane_missile_distance(state) for state in self.xs])

        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        ax.plot(times, d_target, color="#2A9D8F", linewidth=2.0, label="distance to active target")
        ax.plot(times, d_missile, color="#B23A48", linewidth=2.0, label="distance to missile")
        ax.axhline(self.config["missile_capture_radius"], color="#B23A48", linestyle="--", linewidth=1.0, label="missile hit radius")
        ax.axhline(self.config["target_success_radius"], color="#2A9D8F", linestyle="--", linewidth=1.0, label="target radius")
        ax.scatter(
            [self.summary["closest_approach_time"]],
            [self.summary["min_distance_to_missile"]],
            color="#B23A48",
            s=55,
            zorder=4,
            label="closest approach",
        )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance [m]")
        ax.set_title("Target Progress And Missile Separation")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_dir / "distances.png")
        plt.close(fig)

    def plot_controls(self, plt):
        if self.us.size == 0:
            return
        times = self.control_times()
        labels = self.config.get("ground_mdp_control_labels", ["alpha", "gamma", "throttle"])
        colors = ["#0B3954", "#F4A261", "#2A9D8F"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), dpi=150, sharex=True)
        for idx, ax in enumerate(axes):
            ax.step(times, self.us[:, idx], where="post", color=colors[idx], linewidth=1.8)
            ax.set_ylabel(labels[idx])
            lower, upper = self.control_limits()[idx]
            ax.axhline(lower, color="0.55", linestyle=":", linewidth=0.9)
            ax.axhline(upper, color="0.55", linestyle=":", linewidth=0.9)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle("Executed Controls")
        fig.tight_layout()
        fig.savefig(self.output_dir / "controls.png")
        plt.close(fig)

    def plot_states(self, plt):
        times = self.state_times()
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), dpi=150, sharex=True)
        panels = [
            ("aircraft speed [m/s]", self.xs[:, 3], "#0B3954"),
            ("missile speed [m/s]", self.xs[:, 9], "#B23A48"),
            ("aircraft altitude [m]", -self.xs[:, 2], "#0B3954"),
            ("missile altitude [m]", -self.xs[:, 8], "#B23A48"),
            ("aircraft theta [rad]", self.xs[:, 4], "#0B3954"),
            ("aircraft psi [rad]", self.xs[:, 5], "#0B3954"),
        ]
        for ax, (label, data, color) in zip(axes.ravel(), panels):
            ax.plot(times, data, color=color, linewidth=1.8)
            ax.set_ylabel(label)
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        fig.suptitle("State History")
        fig.tight_layout()
        fig.savefig(self.output_dir / "states.png")
        plt.close(fig)

    def plot_spectral_branches(self, plt):
        snapshots = self.summary["planner_snapshots"]
        if len(snapshots) == 0:
            return

        fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
        plane = self.xs[:, 0:3]
        ax.plot(plane[:, 0], plane[:, 1], color="#111111", linewidth=2.6, label="executed aircraft path")

        colors = ["#457B9D", "#8AB17D", "#E9C46A", "#F4A261", "#9D4EDD"]
        max_plot = int(self.config.get("max_branch_trajs_to_plot", 45))
        for snap_idx, snapshot in enumerate(snapshots):
            color = colors[snap_idx % len(colors)]
            root = snapshot["root_state"]
            ax.scatter(root[0], root[1], color=color, s=38, zorder=4)
            plotted = 0
            for branch in snapshot["branch_trajs"]:
                if plotted >= max_plot:
                    break
                if branch.shape[1] < 3:
                    continue
                alpha = 0.18 if plotted else 0.34
                label = f"SETS rollouts @ call {snapshot['planner_call']}" if plotted == 0 else None
                ax.plot(branch[:, 0], branch[:, 1], color=color, alpha=alpha, linewidth=0.9, label=label)
                plotted += 1

        for idx, target in enumerate(self.targets):
            ax.scatter(target[0], target[1], color="#2A9D8F", s=85, marker="*")
            ax.text(target[0], target[1], f" {idx + 1}:{self.target_names[idx]}", color="#1B6F67")

        for obstacle in self.obstacles:
            rect = plt.Rectangle(
                (obstacle[0, 0], obstacle[1, 0]),
                obstacle[0, 1] - obstacle[0, 0],
                obstacle[1, 1] - obstacle[1, 0],
                facecolor="#E76F51",
                edgecolor="#9C2F1B",
                alpha=0.20,
                linewidth=1.2,
            )
            ax.add_patch(rect)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Sampled SETS/UCT2 Spectral Rollout Branches")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_dir / "spectral_branches.png")
        plt.close(fig)

    def control_limits(self):
        raw = np.array(self.config["ground_mdp_U"], dtype=float).reshape((3, 2), order="F")
        return raw

    @staticmethod
    def cuboid_vertices(bounds):
        x0, x1 = bounds[0]
        y0, y1 = bounds[1]
        z0, z1 = -bounds[2, 0], -bounds[2, 1]
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
        faces = [
            [corners[idx] for idx in [0, 1, 2, 3]],
            [corners[idx] for idx in [4, 5, 6, 7]],
            [corners[idx] for idx in [0, 1, 5, 4]],
            [corners[idx] for idx in [2, 3, 7, 6]],
            [corners[idx] for idx in [1, 2, 6, 5]],
            [corners[idx] for idx in [0, 3, 7, 4]],
        ]
        return faces

    @staticmethod
    def equalize_3d_axes(ax, plane, missile):
        points = np.vstack((plane, missile))
        spans = np.ptp(points, axis=0)
        centers = np.mean(points, axis=0)
        radius = max(np.max(spans) / 2.0, 1.0)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(-centers[2] - radius, -centers[2] + radius)


def print_final_summary(summary):
    final_state = summary["rollout_xs"][-1]
    final_target_idx = min(
        int(summary["active_target_indices"][-1]),
        len(summary["targets"]) - 1,
    )
    final_target = summary["targets"][final_target_idx]
    print("")
    print("termination_reason:", summary["termination_reason"])
    print("planner_calls:", summary["planner_calls"])
    print("executed_steps:", len(summary["rollout_us"]))
    print("targets_reached:", len(summary["reached_targets"]), "/", len(summary["targets"]))
    print("final_active_target:", summary["target_names"][final_target_idx])
    print("final_plane_position:", np.round(plane_position(final_state), 3))
    print("final_missile_position:", np.round(missile_position(final_state), 3))
    print("final_distance_to_active_target:", np.round(plane_target_distance(final_state, final_target), 3))
    print("final_distance_to_missile:", np.round(plane_missile_distance(final_state), 3))
    print("min_distance_to_missile:", np.round(summary["min_distance_to_missile"], 3))
    print("min_distance_state_idx:", summary["min_distance_state_idx"])
    print("closest_approach_time:", np.round(summary["closest_approach_time"], 3))
    print("output_dir:", summary["output_dir"])
    if len(summary["rollout_rs"]) > 0:
        print("cumulative_reward:", np.round(np.sum(summary["rollout_rs"]), 4))
        print("mean_step_reward:", np.round(np.mean(summary["rollout_rs"]), 4))
    if len(summary["rollout_us"]) > 0:
        print("last_control:", np.round(summary["rollout_us"][-1], 4))


def main():
    parser = argparse.ArgumentParser(description="Closed-loop SETS/UCT2 plane-missile escape rollout.")
    parser.add_argument("--config", default="plane_missile_escape_rollout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config_path = util.get_config_path(args.config)
    config_dict = util.load_yaml(config_path)

    runner = PlaneMissileEscapeRunner(
        config_dict=config_dict,
        config_path=config_path,
        seed=args.seed,
        make_plots=not args.no_plots,
    )
    summary = runner.run()
    print_final_summary(summary)


if __name__ == "__main__":
    main()
