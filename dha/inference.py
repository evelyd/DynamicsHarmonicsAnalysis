import os
import glob
import torch
import dha
import numpy as np
from pathlib import Path

from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.DynamicsAutoEncoder import DAE
from dha.utils.mysc import class_from_name
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
import dha.utils.isaaclab_utils as utils

import escnn
from escnn.nn import FieldType
import re

model_path = "experiments/test/S:2025-05-16_16-16-41-OS:5-G:K4xC2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=481" #711" 179 481 529
# model_path = "experiments/test/S:2025-05-16_16-22-18-OS:5-G:K4xC2-H:5-EH:5_E-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=mini_cheetah/seed=227"

terrains = ["curriculum"] #, "uneven_easy", "uneven_medium", "uneven_hard_squares"]
modes = ["2025-05-16_16-16-41"]
# modes = ["2025-05-16_16-22-18"]
for terrain in terrains:
    for mode in modes:
        data_paths = list(Path(f"data/mini_cheetah/isaaclab_recordings/{terrain}/{mode}/raw_recording").glob("*.npy"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dha_dir = os.path.dirname(dha.__file__)
model_dir = os.path.join(dha_dir, model_path)

model = utils.get_trained_dae_model(model_dir).to(device)
model.eval()  # Set the model to evaluation mode

 # Get the normalization info for the DAE model
norm_dir = os.path.join(model_dir, "state_mean_var.npy")
# Load state_mean and state_var from the npy file
norm_data = np.load(norm_dir, allow_pickle=True).item()

# Extract state_mean and state_var values
state_mean, state_std, action_mean, action_std = utils.get_stats(model_path, device)

# Get some data to test the model
all_data = []
all_action_data = []
for data_path in data_paths:
    assert data_path.exists(), f"Path {data_path.absolute()} does not exist"
    data = np.load(data_path, allow_pickle=True)
    all_data.append(np.array([traj['obs'] for traj in data]))
    all_action_data.append(np.array([traj['actions'] for traj in data]))
state_batched = np.concatenate(all_data, axis=1) # shape is (ep_length, num_envs, obs_dim)
action_batched = np.concatenate(all_action_data, axis=1)
# Reshape the data so that the first dimension is end to end
# obs = state_batched.transpose((1,0,2)).reshape(state_batched.shape[0] * state_batched.shape[1], -1)
# joint_angle_action = action_batched.transpose((1,0,2)).reshape(action_batched.shape[0] * action_batched.shape[1], -1) # this is the joint angle action
# Convert to torch tensors
obs = torch.tensor(state_batched, device=device).float()
joint_angle_action = torch.tensor(action_batched, device=device).float() # joint angle action

q0 = utils.get_pybullet_q0(device)

joint_order_indices = utils.get_joint_order_indices()

# Extract the state_obs and action_obs (action is velocity commands)
# latent_state, state, action = utils.get_latent_state(obs, model_path, model, joint_order_indices, q0, state_mean, state_std, action_mean, action_std)

# Parse the prediction horizon from the string
prediction_horizon = int(re.search(r"H:(\d+)", model_path).group(1))

# Get the state, action, and next_state tensors
state_batched, action_batched = utils.get_state_action_from_obs_batched(obs, joint_order_indices, q0)

state, action, next_state = utils.reshape_state_action(state_batched, action_batched, prediction_horizon)

# Choose an env to look at
env_idx = 140

state = state[:, env_idx].squeeze(0)
action = action[:, :, env_idx].squeeze(0)
next_state = next_state[:, :, env_idx].squeeze(0)

# Normalize before passing to the model
state_normed, action_normed, next_state_normed = utils.normalize(state_mean, state_std, action_mean, action_std, state, action, next_state)

# Predict the next state using the DAE model
with torch.no_grad():
    pred_dict = model(state_normed, action_normed, next_state_normed)

# Compare the predicted state with the actual state
predicted_state_normed = pred_dict['pred_state_traj'] # shape (batch, pred_horizon + 1, state_dim)
pred_next_state_normed = predicted_state_normed[:, 1:, :]

pred_next_state = utils.denormalize(state_mean, state_std, pred_next_state_normed)

# Compute the RMSE for this env idx
rmse = torch.sqrt(torch.mean((next_state - pred_next_state) ** 2)) #, dim=(0, 1)))
input(rmse)

# Count the unstable eigvals of A
A = model.obs_space_dynamics.transfer_op.weight.detach().cpu().numpy()
eigvals = np.linalg.eigvals(A)
unstable_eigvals = np.sum(np.abs(eigvals) > 1)
input(f"Number of unstable eigenvalues for env {env_idx}: {unstable_eigvals}")

# plot the states together
import matplotlib.pyplot as plt

# Define the state observation names and their dimensions
state_obs_names = [
    ("joint_pos_S1", 24),
    ("joint_vel", 12),
    ("base_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("a_joint_pos_S1", 24)
]

# Start index for slicing
start_idx = 0

# Create a separate plot for each state observation
colors = plt.cm.tab10.colors  # Use a colormap for consistent colors
color_idx = 0  # Initialize color index

for name, dim in state_obs_names:
    end_idx = start_idx + dim
    plt.figure(figsize=(12, 6))
    for i in range(dim):
        color = colors[i % len(colors)]  # Cycle through colors for each component
        plt.plot(
            next_state[:, -1, start_idx + i].cpu().numpy(),
            label=f'Actual {name}[{i}]',
            alpha=0.5,
            color=color
        )
        plt.plot(
            pred_next_state[:, -1, start_idx + i].cpu().numpy(),
            label=f'Predicted {name}[{i}]',
            linestyle='--',
            color=color,
            alpha=0.5
        )
    plt.title(f'Actual vs Predicted State: {name}')
    plt.xlabel('Time Step')
    plt.ylabel(f'{name} Value')
    plt.legend()
    plt.show()
    start_idx = end_idx

#TODO fix this
# /home/edelia-iit.local/miniforge3/envs/env_isaaclab/lib/python3.10/site-packages/torch/nn/init.py:511: UserWarning: Initializing zero-element tensors is a no-op
#   warnings.warn("Initializing zero-element tensors is a no-op")
# Traceback (most recent call last):
#   File "/home/edelia-iit.local/git/DynamicsHarmonicsAnalysis/dha/better_inference.py", line 34, in <module>
#     model = utils.get_trained_dae_model(model_dir)
#   File "/home/edelia-iit.local/git/DynamicsHarmonicsAnalysis/dha/utils/isaaclab_utils.py", line 234, in get_trained_dae_model
#     model.load_state_dict(remove_state_dict_prefix(state_dict, "model."))
#   File "/home/edelia-iit.local/miniforge3/envs/env_isaaclab/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2584, in load_state_dict
#     raise RuntimeError(
# RuntimeError: Error(s) in loading state_dict for ControlledDAE:
#         size mismatch for obs_fn.net.block_0.linear_0.weight: copying a param with shape torch.Size([512, 69]) from checkpoint, the shape in current model is torch.Size([0, 69]).
#         size mismatch for obs_fn.net.block_1.linear_1.weight: copying a param with shape torch.Size([512, 512]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for obs_fn.net.block_2.linear_2.weight: copying a param with shape torch.Size([512, 512]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for obs_fn.net.block_3.linear_3.weight: copying a param with shape torch.Size([512, 512]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for obs_fn.net.head.linear_4.weight: copying a param with shape torch.Size([345, 512]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for inv_obs_fn.net.block_0.linear_0.weight: copying a param with shape torch.Size([512, 345]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for inv_obs_fn.net.block_1.linear_1.weight: copying a param with shape torch.Size([512, 512]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for inv_obs_fn.net.block_2.linear_2.weight: copying a param with shape torch.Size([512, 512]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for inv_obs_fn.net.block_3.linear_3.weight: copying a param with shape torch.Size([512, 512]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for inv_obs_fn.net.head.linear_4.weight: copying a param with shape torch.Size([69, 512]) from checkpoint, the shape in current model is torch.Size([69, 0]).
#         size mismatch for obs_space_dynamics.transfer_op.weight: copying a param with shape torch.Size([345, 345]) from checkpoint, the shape in current model is torch.Size([0, 0]).
#         size mismatch for obs_space_dynamics.transfer_op.bias: copying a param with shape torch.Size([345]) from checkpoint, the shape in current model is torch.Size([0]).
#         size mismatch for obs_space_dynamics.control_op.weight: copying a param with shape torch.Size([345, 3]) from checkpoint, the shape in current model is torch.Size([0, 3]).
#         size mismatch for obs_space_dynamics.control_op.bias: copying a param with shape torch.Size([345]) from checkpoint, the shape in current model is torch.Size([0]).