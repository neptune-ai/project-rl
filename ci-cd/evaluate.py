# adapted from:
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
# date accessed: 2021.06.30

import os
import sys
from collections import namedtuple
from itertools import count

import gym
import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# (Neptune) Set environment variable
os.environ["NEPTUNE_PROJECT"] = "common/project-rl"

# (Neptune) Fetch project
project = neptune.init_project()

# (Neptune) Find latest run
runs_table_df = project.fetch_runs_table().to_pandas()
runs_table_df = runs_table_df.sort_values(by="sys/creation_time", ascending=False)
run_id = runs_table_df["sys/id"].values[0]

# (Neptune) Resume run
run = neptune.init_run(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    with_id=run_id,
    monitoring_namespace="evaluation/monitoring",
)

# (Neptune) Initialize model version created by the run
model_version = neptune.init_model_version(with_id=run["training/model/id"].fetch())

# (Neptune) Download trained agent from model repository
model_version["weights"].download("policy_net.pth")

# (Neptune) Fetch environment name and number of actions in the env
env_name = run["training/env_name"].fetch()
n_actions = model_version["n_actions"].fetch()

# (Neptune) Set number of episodes for evaluation and log in under separate namespace dedicated to evaluation
eval_episodes = 5
run["evaluation/n_episodes"] = eval_episodes

# (Neptune) Upload evaluation script as separate file
run["evaluation/script"].upload(sys.argv[0])  # sys.argv[0] returns the path of the current script

# Run evaluation logic
steps_done = 0

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.BICUBIC), T.ToTensor()])


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def _get_screen():
    screen = env.render().transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = _get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)


def _get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


env = gym.make(env_name, render_mode="rgb_array").unwrapped
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_screen = _get_screen()
_, _, screen_height, screen_width = init_screen.shape

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load("policy_net.pth"))
policy_net.eval()


def select_action(state):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


# Main training loop
for _ in range(eval_episodes):
    env.reset()

    last_screen = _get_screen()
    current_screen = _get_screen()
    state = current_screen - last_screen
    cum_reward = 0
    for _ in count():
        action = select_action(state)
        _, reward, done, _, _ = env.step(action.item())
        cum_reward += reward
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = _get_screen()
        next_state = None if done else current_screen - last_screen
        state = next_state

        if done:
            # (Neptune) Log evaluation metrics
            run["evaluation/episode/reward"].append(value=cum_reward)
            model_version["evaluation/reward"].append(value=cum_reward)
            break

env.close()

model_version.sync()

# (Neptune) Compare against production model
with neptune.init_model(with_id="PROJRL-CART") as model:
    model_versions_df = model.fetch_model_versions_table().to_pandas()

production_models = model_versions_df[model_versions_df["sys/stage"] == "production"]["sys/id"]
assert (
    len(production_models) == 1
), f"Multiple model versions found in production: {production_models.values}"

prod_model_id = production_models.values[0]
print(f"Current champion model: {prod_model_id}")

prod_model = neptune.init_model_version(with_id=prod_model_id)
prod_model_avg_reward = prod_model["evaluation/reward"].fetch_values()["value"].mean()
challenger_model_avg_reward = model_version["evaluation/reward"].fetch_values()["value"].mean()

print(
    f"Champion model average reward: {prod_model_avg_reward}\nChallenger model acerage reward: {challenger_model_avg_reward}"
)

if challenger_model_avg_reward > prod_model_avg_reward:
    print("Promoting challenger to production")
    prod_model.change_stage("archived")
    model_version.change_stage("production")
else:
    print("Archiving challenger model")
    model_version.change_stage("archived")

# (Neptune) Append tag "evaluated" to the run and model
run["sys/tags"].add("evaluated")
model_version["sys/tags"].add("evaluated")
