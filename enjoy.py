import os

import torch
import numpy as np

import utils
from model import DuelingDQN
from wrapper import make_atari, wrap_atari_dqn
from arguments import argparser


def get_environ():
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    assert learner_ip != '-1'
    return learner_ip


def main():
    args = argparser()

    args.clip_rewards = False
    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    seed = args.seed + 1122
    utils.set_global_seeds(seed, use_torch=True)
    env.seed(seed)

    model = DuelingDQN(env)
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))

    episode_reward, episode_length = 0, 0
    state = env.reset()
    while True:
        if args.render:
            env.render()
        action, _ = model.act(torch.FloatTensor(np.array(state)), 0.)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            state = env.reset()
            print("Episode Length / Reward: {} / {}".format(episode_length, episode_reward))
            episode_reward = 0
            episode_length = 0


if __name__ == '__main__':
    main()
