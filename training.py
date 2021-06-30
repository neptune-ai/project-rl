# adapted from:
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
# date accessed: 2021.06.30

import math
import random
from collections import namedtuple, deque
from itertools import count

import gif
import gym
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

gif.options.matplotlib["dpi"] = 300

# Create run
run = neptune.init(
    project="common/project-rl",
    name="training",
    tags=["tmp"],
)

env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log device type used for training
run["environment/device"] = device

# Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


# DQN algorithm
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


# Input extraction
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
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


@gif.frame
def get_screen_as_ax(screen):
    plt.figure()
    _, ax = plt.subplots(1, 1,)
    ax.imshow(
        screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
        interpolation='none'
    )
    ax.axis("off")


def env_start_screen():
    plt.figure()
    _, ax = plt.subplots(1, 1,)
    ax.imshow(
        get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        interpolation='none'
    )
    ax.axis("off")
    run["training/visualizations/start_screen"].upload(neptune.types.File.as_image(ax.figure))
    plt.close("all")


# Training
parameters = {
    "BATCH_SIZE": 128,
    "GAMMA": 0.999,
    "EPS_START": 0.9,
    "EPS_END": 0.05,
    "EPS_DECAY": 200,
    "TARGET_UPDATE": 10,
}
run["training/parameters"] = parameters

env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

run["agent/n_actions"] = n_actions

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

replay_mem = 10000

memory = ReplayMemory(replay_mem)
run["training/parameters/replay_memory_size"] = replay_mem

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = parameters["EPS_END"] + (parameters["EPS_START"] - parameters["EPS_END"]) * \
        math.exp(-1. * steps_done / parameters["EPS_DECAY"])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    run["training/episode/duration"].log(value=episode_durations[-1], step=len(episode_durations))
    avg = np.array(episode_durations).sum() / len(episode_durations)
    run["training/episode/avg_duration"].log(value=float(avg), step=len(episode_durations))


# Training loop
def optimize_model():
    if len(memory) < parameters["BATCH_SIZE"]:
        return
    transitions = memory.sample(parameters["BATCH_SIZE"])
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(
            map(
                lambda s: s is not None,
                batch.next_state
            )
        ),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(parameters["BATCH_SIZE"], device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * parameters["GAMMA"]) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    run["training/criterion"] = "SmoothL1Loss"
    run["training/loss"].log(float(loss.detach().cpu().numpy()))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Below, you can find the main training loop.
num_episodes = 61
run["training/params/num_episodes"] = num_episodes

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    if i_episode == 0:
        env_start_screen()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    cum_reward = 0
    frames = []
    for t in count():
        # gif
        frame = get_screen_as_ax(current_screen)
        frames.append(frame)

        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        cum_reward += reward
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            run["training/episode_reward"].log(value=cum_reward, step=i_episode)
            if i_episode % 10 == 0:
                frames_path = "episode_{}.gif".format(i_episode)
                gif.save(
                    frames,
                    frames_path,
                    duration=int(len(frames)/10),
                    unit="s",
                    between="startend"
                )
                run["training/visualizations/episode_{}".format(i_episode)].upload(
                    neptune.types.File(frames_path)
                )
                plt.close("all")
            break
    if i_episode % parameters["TARGET_UPDATE"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

# Log model weights
torch.save(policy_net.state_dict(), 'agent.pth')
run['agent/model_dict'].upload('agent.pth')

print('Complete')
env.render()
env.close()
