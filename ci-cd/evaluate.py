# adapted from:
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
# date accessed: 2021.06.30

import math
import os
import random
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

# (neptune) Fetch project
project = neptune.get_project(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    name="common/project-rl",
)

# (neptune) Find latest run
runs_table_df = project.fetch_runs_table().to_pandas()
runs_table_df = runs_table_df.sort_values(by="sys/creation_time", ascending=False)
run_id = runs_table_df["sys/id"].values[0]

# (neptune) Resume run
run = neptune.init(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    project="common/project-rl",
    run=run_id,
)

# (neptune) Download agent
run["agent/policy_net"].download("policy_net.pth")

# (neptune) Download parameters
parameters = run["training/parameters"].fetch()["training"]["parameters"]

# Run evaluation logic
steps_done = 0

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


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
    screen = env.render(mode="rgb_array").transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = _get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)


def _get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

env_name = "CartPole-v0"
rnd_seed = np.random.randint(low=1000000)

env = gym.make(env_name).unwrapped
env.seed(rnd_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env.reset()
init_screen = _get_screen()
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load("policy_net.pth"))
policy_net.eval()


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = parameters["eps_end"] + (parameters["eps_start"] - parameters["eps_end"]) * \
        math.exp(-1. * steps_done / parameters["eps_decay"])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# Main training loop
eval_episodes = 5

# (neptune) Log evaluation parameters under "evaluation" namespace
run["evaluation/n_episodes"] = eval_episodes
for i_episode in range(eval_episodes):
    env.reset()

    last_screen = _get_screen()
    current_screen = _get_screen()
    state = current_screen - last_screen
    cum_reward = 0
    frames = []
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        cum_reward += reward
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = _get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        state = next_state

        if done:
            # (neptune) Log evaluation metrics
            run["evaluation/episode/reward"].log(value=cum_reward, step=i_episode)
            break

env.close()

# (neptune) Append tag "evaluated" to the run
run["sys/tags"].add("evaluated")
