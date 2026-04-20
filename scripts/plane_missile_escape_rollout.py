from pathlib import Path
import numpy as np
import sys

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
        np.diag(np.array(config_dict["dots_rho"])),
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


def summarize_result(config_dict, xs, us, result):
    final_state = xs[-1]
    plane_final = final_state[0:3]
    missile_final = final_state[6:9]
    target = np.array(config_dict["target_position"])

    print("success:", result.success)
    print("tree_rollouts:", len(result.vs))
    print("planned_horizon:", len(xs))
    print("final_plane_position:", np.round(plane_final, 3))
    print("final_missile_position:", np.round(missile_final, 3))
    print("distance_plane_to_target:", np.round(np.linalg.norm(plane_final - target), 3))
    print("distance_plane_to_missile:", np.round(np.linalg.norm(plane_final - missile_final), 3))
    if len(us) > 0:
        print("first_control:", np.round(us[0], 4))


def main():
    config_name = "plane_missile_escape_rollout"
    config_path = util.get_config_path(config_name)
    config_dict = util.load_yaml(config_path)

    rng = RNG()
    rng.set_seed(0)

    ground_mdp = get_mdp(config_dict["ground_mdp_name"], config_path)
    ground_mdp.clear_obstacles()
    ground_mdp.clear_thermals()

    dots_mdp = build_dots(config_dict, ground_mdp)
    uct = build_uct(config_dict)

    curr_state = np.array(ground_mdp.initial_state())
    result = run_uct2(dots_mdp, uct, curr_state, rng)

    xs = np.array(result.planned_traj.xs)
    us = np.array(result.planned_traj.us)

    if xs.shape[0] == 0:
        print("planned trajectory is empty")
        print("success:", result.success)
        return

    summarize_result(config_dict, xs, us, result)


if __name__ == "__main__":
    main()
