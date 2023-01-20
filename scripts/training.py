# adapted from:
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
# date accessed: 2021.06.30

import math
import os
import random
from collections import deque, namedtuple
from itertools import count

import gif
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from neptune.new.types import File
from PIL import Image

mpl.rcParams["figure.max_open_warning"] = 0
plt.switch_backend("agg")

# (Neptune) Set environment variables
os.environ["NEPTUNE_PROJECT"] = "common/project-rl"

# (Neptune) Create run
run = neptune.init_run(
    name="training",
    tags=["training", "CartPole"],
)

parameters = {
    "batch_size": 128,
    "eps_start": 0.73,
    "eps_end": 0.02,
    "eps_decay": 10,
    "gamma": 0.99,
    "num_episodes": 51,
    "target_update": 10,
}

# (Neptune) Log dict as parameters
run["training/parameters"] = parameters

# (Neptune) You can add more parameters on the way
run["training/parameters/criterion"] = "SmoothL1Loss"

gif.options.matplotlib["dpi"] = 300
steps_done = 0
episode_durations = []

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.BICUBIC), T.ToTensor()])


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


@gif.frame
def _get_screen_as_ax(screen):
    plt.figure()
    _, ax = plt.subplots(
        1,
        1,
    )
    ax.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation="none")
    ax.axis("off")


def _get_env_start_screen():
    plt.figure()
    _, ax = plt.subplots(
        1,
        1,
    )
    ax.imshow(_get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation="none")
    ax.axis("off")
    return ax.figure


# (Neptune) Log metrics. In this example episode durations
def _plot_durations():
    run["training/episode/duration"].append(
        value=episode_durations[-1], step=len(episode_durations)
    )
    avg = np.array(episode_durations).sum() / len(episode_durations)
    run["training/avg_duration"].append(value=float(avg), step=len(episode_durations))


# (Neptune) Log environment info
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="rgb_array").unwrapped
run["training/env_name"] = env_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run["training/environment/device_name"] = device

env.reset()
init_screen = _get_screen()
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

# (Neptune) Add more parameters to the "training/parameters" namespace
replay_memory = 10000
memory = ReplayMemory(replay_memory)
run["training/parameters/replay_memory_size"] = replay_memory


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = parameters["eps_end"] + (
        parameters["eps_start"] - parameters["eps_end"]
    ) * math.exp(-1.0 * steps_done / parameters["eps_decay"])
    steps_done += 1
    if sample <= eps_threshold:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


def optimize_model():
    if len(memory) < parameters["batch_size"]:
        return
    transitions = memory.sample(parameters["batch_size"])
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(parameters["batch_size"], device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * parameters["gamma"]) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # (Neptune) Log loss, have it as a chart in neptune
    run["training/loss"].append(float(loss.detach().cpu().numpy()))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Main training loop
import contextlib

for i_episode in range(parameters["num_episodes"]):
    env.reset()

    # (Neptune) Log single image
    if i_episode == 0:
        run["visualizations/start_screen"].upload(File.as_image(_get_env_start_screen()))
    last_screen = _get_screen()
    current_screen = _get_screen()
    state = current_screen - last_screen
    frames = []
    for t in count():
        frame = _get_screen_as_ax(current_screen)
        frames.append(frame)

        # (Neptune) What my agent is looking at? Log series of images.
        if i_episode % 10 == 0:
            input_screen = state.detach().cpu().numpy().squeeze()
            input_screen = (input_screen - input_screen.min()) / (
                input_screen.max() - input_screen.min() + 0.000001
            )
            input_screen = np.transpose(input_screen, (1, 2, 0))

            run[f"visualizations/episode_{i_episode}/input_screens"].append(
                File.as_image(input_screen)
            )

        action = select_action(state)
        _, reward, done, _, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = _get_screen()
        next_state = None if done else current_screen - last_screen
        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            _plot_durations()

            if i_episode % 10 == 0:
                frames_path = f"episode_{i_episode}.gif"
                gif.save(
                    frames,
                    frames_path,
                    duration=len(frames) // 10,
                    unit="s",
                    between="startend",
                )

                # (Neptune) Log gif to see episode recording
                run[f"visualizations/episode_{i_episode}/episode_recording"].upload(
                    File(frames_path)
                )
                plt.close("all")
            break
    if i_episode % parameters["target_update"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()

# (Neptune) Initialize a new model version
from neptune.new.exceptions import NeptuneModelKeyAlreadyExistsError

with contextlib.suppress(NeptuneModelKeyAlreadyExistsError):
    # Initialize a new model if it doesn not already exist
    model = neptune.init_model(name="cartpole", key="CART")

model_version = neptune.init_model_version(model="PROJRL-CART", name="cartpole")

# (Neptune) Log model weights to model version
torch.save(policy_net.state_dict(), "policy_net.pth")
model_version["weights"].upload("policy_net.pth")

# (Neptune) Associate run with model version and vice-versa
run_meta = {
    "id": run.get_structure()["sys"]["id"].fetch(),
    "name": run.get_structure()["sys"]["name"].fetch(),
    "url": run.get_url(),
}
model_version["run"] = run_meta

model_version_meta = {
    "id": model_version.get_structure()["sys"]["id"].fetch(),
    "name": model_version.get_structure()["sys"]["name"].fetch(),
    "url": model_version.get_url(),
}
run["training/model"] = model_version_meta

# (Neptune) Save parameters with model
model_version["parameters"] = run["training/parameters"].fetch()
model_version["n_actions"] = n_actions

# (Neptune) Promote model version stage to "staging"
model_version.change_stage("staging")
