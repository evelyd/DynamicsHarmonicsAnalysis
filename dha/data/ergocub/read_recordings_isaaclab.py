from pathlib import Path

import numpy as np
import pybullet
from escnn.group import Group, Representation
from morpho_symm.data.DynamicsRecording import DynamicsRecording, split_train_val_test
from morpho_symm.utils.algebra_utils import permutation_matrix
from morpho_symm.utils.rep_theory_utils import escnn_representation_form_mapping, group_rep_from_gens
from morpho_symm.utils.robot_utils import load_symmetric_system
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation

def get_kinematic_three_rep(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(4, dtype=int)}
    gens = [permutation_matrix([1, 0, 3, 2]), permutation_matrix([2, 3, 0, 1]), permutation_matrix([0, 1, 2, 3])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three


def get_Rd_signals_on_kin_subchains(G: Group, rep_kin_three: Representation):
    rep_R3 = G.representations["R3"]
    rep_F = {G.identity: np.eye(12, dtype=int)}
    gens = [np.kron(rep_kin_three(g), rep_R3(g)) for g in G.generators]
    for h, rep_h in zip(G.generators, gens):
        rep_F[h] = rep_h

    rep_F = group_rep_from_gens(G, rep_F)
    rep_F.name = "R3_on_legs"
    return rep_F

def convert_ergocub_isaaclab_recordings(data_paths: list, ideal: bool = False):
    """Convertion script for the recordings of observations from the Ergocub Robot.

    This function takes recordings stored into a single numpy array of shape (time, state_dim) where the state is
    defined as [state]:
        base_velocity,                           (3,)
        base_angular_velocity,                   (3,)
        projected_gravity,                       (3,)
        joint position,                          (26,)
        joint velocities,                        (26,)
        past actions,                            (26,)
        velocity_commands,                       (3,)
        _________________________________________________________
        TOTAL:                                    90. Dimensions
    The conversion process takes these measurements and does the following:
        1. Stores them into a DynamicsRecording format, for easy loading
        2. Defines the group representation for each observation.
        3. Changes joint position to Pinocchio convention, used by MorphoSymm.
    """
    all_data = []
    all_action_data = []
    for data_path in data_paths:
        assert data_path.exists(), f"Path {data_path.absolute()} does not exist"
        data = np.load(data_path, allow_pickle=True)
        all_data.append(np.array([traj['obs'] for traj in data]))
        all_action_data.append(np.array([traj['actions'] for traj in data]))
    print(f"Shape of all data: {all_data[0].shape}")
    print(f"Shape of all action data: {all_action_data[0].shape}")
    state_batched = np.concatenate(all_data, axis=1)
    action_batched = np.concatenate(all_action_data, axis=1)
    # Reshape the data so that the first dimension is end to end
    state = state_batched.transpose((1,0,2)).reshape(state_batched.shape[0] * state_batched.shape[1], -1)
    action = action_batched.transpose((1,0,2)).reshape(action_batched.shape[0] * action_batched.shape[1], -1)
    num_states = state.shape[-1]
    desired_num_states = 90 if ideal else 357
    assert num_states == desired_num_states, f"Expected {desired_num_states} dimensions in the state, got {state.shape[-1]}"

    dt = 0.01  # Time step of the simulation

    # Load the Ergocub robot
    robot, G = load_symmetric_system(robot_name="ergocub")
    rep_Q_js = G.representations["Q_js"]  # Representation on joint space position coordinates
    rep_TqQ_js = G.representations["TqQ_js"]  # Representation on joint space velocity coordinates
    rep_Rd = G.representations["R3"]  # Representation on vectors in R^d
    rep_Rd_pseudo = G.representations["R3_pseudo"]  # Representation on pseudo vectors in R^d
    rep_euler_xyz = G.representations["euler_xyz"]  # Representation on Euler angles
    rep_kin_three = get_kinematic_three_rep(G)  # Permutation of legs
    rep_Rd_on_limbs = get_Rd_signals_on_kin_subchains(G, rep_kin_three)  # Representation on R^3 on legs

    rep_z = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_z.name = "base_z"
    rep_xy = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[:2, :2].reshape((2, 2)) for h in G.elements if h != G.identity})
    rep_xy.name = "base_xy"
    rep_euler_z = group_rep_from_gens(G, rep_H={h: rep_euler_xyz(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_euler_z.name = "euler_z"

    # Create a mapping from current joint order to morphosymm order
    usd_joint_order = [
        "l_hip_pitch",
        "r_hip_pitch",
        "torso_roll",
        "l_hip_roll",
        "r_hip_roll",
        "torso_pitch",
        "torso_yaw",
        "l_hip_yaw",
        "r_hip_yaw",
        "l_shoulder_pitch",
        "neck_pitch",
        "r_shoulder_pitch",
        "l_knee",
        "r_knee",
        "l_shoulder_roll",
        "neck_roll",
        "r_shoulder_roll",
        "l_ankle_pitch",
        "r_ankle_pitch",
        "neck_yaw",
        "l_ankle_roll",
        "r_ankle_roll",
        "l_shoulder_yaw",
        "r_shoulder_yaw",
        "l_elbow",
        "r_elbow",
    ]
    joint_order_for_morphosymm = robot.joint_space_names
    joint_order_indices = [
        usd_joint_order.index(joint)
        for joint in joint_order_for_morphosymm
        if joint in usd_joint_order
    ] # puts isaaclab-ordered joints into morphosymm order

    # Define observation variables and their group representations

    if ideal:
        # Base body observations ___________________________________________________________________________________________
        base_vel = state[:, :3]  # Rep: rep_Rd
        base_ang_vel = state[:, 3:6]  # Rep: rep_euler_xyz
        projected_gravity = state[:, 6:9] # Rep: red Rd
        velocity_commands_xy = state[:, 87:89] # Rep: Rd for xy, euler xyz for heading
        velocity_commands_z = state[:, 89][:, np.newaxis]

        # Joint-Space observations _________________________________________________________________________________________

        # Fill in zeros for all the joint-space obs for joints that are in ms but not isaaclab
        usd_name_to_vel_map = {name: pos for name, pos in zip(usd_joint_order, state[:, 35:61].T)} # type: ignore
        joint_vel = np.stack([
            usd_name_to_vel_map.get(joint_name, np.zeros(state.shape[0]))
            for joint_name in joint_order_for_morphosymm
        ], axis=1)
        usd_name_to_pos_map = {name: pos for name, pos in zip(usd_joint_order, state[:, 9:35].T)} # type: ignore
        joint_pos = np.stack([
            usd_name_to_pos_map.get(joint_name, np.zeros(state.shape[0]))
            for joint_name in joint_order_for_morphosymm
        ], axis=1)
        usd_name_to_prev_action_map = {name: pos for name, pos in zip(usd_joint_order, state[:, 61:87].T)} # type: ignore
        prev_action = np.stack([
            usd_name_to_prev_action_map.get(joint_name, np.zeros(state.shape[0]))
            for joint_name in joint_order_for_morphosymm
        ], axis=1)

        # Subsample the data by skippig by ignoring odd frames. ============================================================
        dt_subsample = 1
        base_vel = base_vel[::dt_subsample]
    else:
        # Base body observations ___________________________________________________________________________________________
        base_ang_vel = state[:, 27:30]  # Rep: rep_euler_xyz
        projected_gravity = state[:, 57:60] # Rep: red Rd
        velocity_commands_xy = state[:, 354:356] # Rep: Rd for xy, euler xyz for heading
        velocity_commands_z = state[:, 356][:, np.newaxis]

        # Joint-Space observations _________________________________________________________________________________________

        # Fill in zeros for all the joint-space obs for joints that are in ms but not isaaclab
        usd_name_to_vel_map = {name: pos for name, pos in zip(usd_joint_order, state[:, 326:340].T)} # type: ignore
        joint_vel = np.stack([
            usd_name_to_vel_map.get(joint_name, np.zeros(state.shape[0]))
            for joint_name in joint_order_for_morphosymm
        ], axis=1)
        usd_name_to_pos_map = {name: pos for name, pos in zip(usd_joint_order, state[:, 186:200].T)} # type: ignore
        joint_pos = np.stack([
            usd_name_to_pos_map.get(joint_name, np.zeros(state.shape[0]))
            for joint_name in joint_order_for_morphosymm
        ], axis=1)
        usd_name_to_prev_action_map = {name: pos for name, pos in zip(usd_joint_order, state[:, 340:354].T)} # type: ignore
        prev_action = np.stack([
            usd_name_to_prev_action_map.get(joint_name, np.zeros(state.shape[0]))
            for joint_name in joint_order_for_morphosymm
        ], axis=1)

    # Joint-Space actions ============================================================

    usd_name_to_action_map = {name: pos for name, pos in zip(usd_joint_order, action.T)} # type: ignore
    action = np.stack([
        usd_name_to_action_map.get(joint_name, np.zeros(state.shape[0]))
        for joint_name in joint_order_for_morphosymm
    ], axis=1)

    # Subsample the data by skippig by ignoring odd frames. ============================================================
    dt_subsample = 1
    # base_vel = base_vel[::dt_subsample]
    base_ang_vel = base_ang_vel[::dt_subsample]
    projected_gravity = projected_gravity[::dt_subsample]
    joint_pos = joint_pos[::dt_subsample]
    joint_vel = joint_vel[::dt_subsample]
    prev_action = prev_action[::dt_subsample]
    action = action[::dt_subsample]
    velocity_commands_xy = velocity_commands_xy[::dt_subsample]
    velocity_commands_z = velocity_commands_z[::dt_subsample]
    # Define the dataset.
    data_recording = DynamicsRecording(
        description=f"Ergocub {data_path.parent.parent.stem}",
        info=dict(num_traj=1, trajectory_length=state.shape[0]),
        dynamics_parameters=dict(dt=dt * dt_subsample, group=dict(group_name=G.name, group_order=G.order())),
        recordings=dict(
            # base_vel=base_vel[None, ...].astype(np.float32),
            base_ang_vel=base_ang_vel[None, ...].astype(np.float32),
            projected_gravity=projected_gravity[None, ...].astype(np.float32),
            joint_pos=joint_pos[None, ...].astype(np.float32),
            joint_vel=joint_vel[None, ...].astype(np.float32),
            prev_action=prev_action[None, ...].astype(np.float32),
            velocity_commands_xy=velocity_commands_xy[None, ...].astype(np.float32),
            velocity_commands_z=velocity_commands_z[None, ...].astype(np.float32),
            action=action[None, ...].astype(np.float32),
        ),
        state_obs=(
            # 'base_vel',
            'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'prev_action', 'velocity_commands_xy', 'velocity_commands_z'
        ),
        action_obs=("action",),
        obs_representations=dict(
            joint_pos=rep_TqQ_js,  # Joint-Space observations
            joint_vel=rep_TqQ_js,
            prev_action=rep_TqQ_js,
            # Base body observations
            velocity_commands_xy=rep_xy,
            velocity_commands_z=rep_euler_z,
            # base_vel=rep_Rd,
            projected_gravity=rep_Rd,
            base_ang_vel=rep_euler_xyz,
            action=rep_TqQ_js,
        ),
    )

    # Compute the mean and variance of all observations considering symmetry constraints.
    for obs_name in data_recording.recordings.keys():
        if obs_name in data_recording.obs_moments:
            continue
        data_recording.compute_obs_moments(obs_name=obs_name)

    train_dr, val_dr, test_dr = split_train_val_test(
        data_recording, partition_sizes=(0.7, 0.15, 0.15), split_dimension="time"
    )

    for dr, p_name in zip([train_dr, val_dr, test_dr], ["train", "val", "test"]):
        file_name = f"n_trajs={dr.info['num_traj']}-frames={dr.info['trajectory_length']}-{p_name}.pkl"
        dr.save_to_file(data_path.parent.parent / file_name)
        print(f"{p_name} Dynamics Recording saved to {data_path.parent.parent / file_name}")


if __name__ == "__main__":
    tasks = ["amp_velocity"]
    # modes = ["2025-06-20_12-29-39"]
    modes = ["2025-08-06_13-51-38"]
    ideal = False
    for task in tasks:
        for mode in modes:
            data_paths = list(Path(f"data/ergocub/isaaclab_recordings/{task}/{mode}/raw_recording").glob("*.npy"))
            convert_ergocub_isaaclab_recordings(data_paths, ideal=ideal)
