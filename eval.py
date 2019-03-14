import _pickle as pickle
import os

import zmq
import torch
from tensorboardX import SummaryWriter
import numpy as np

import utils
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from arguments import argparser


def get_environ():
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    assert learner_ip != '-1'
    return learner_ip


def main():
    learner_ip = get_environ()
    args = argparser()

    writer = SummaryWriter(comment="-{}-eval".format(args.env))

    ctx = zmq.Context()
    param_socket = ctx.socket(zmq.SUB)
    param_socket.setsockopt(zmq.SUBSCRIBE, b'')
    param_socket.setsockopt(zmq.CONFLATE, 1)
    param_socket.connect('tcp://{}:52001'.format(learner_ip))

    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    seed = args.seed + 1122
    utils.set_global_seeds(seed, use_torch=True)
    env.seed(seed)

    model = DuelingDQN(env)

    data = param_socket.recv(copy=False)
    param = pickle.loads(data)
    model.load_state_dict(param)
    print("Loaded first parameter from learner")

    episode_reward, episode_length, episode_idx = 0, 0, 0
    state = env.reset()
    while True:
        if args.render:
            env.render()
        action, _ = model.act(torch.FloatTensor(np.array(state)), 0.01)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            state = env.reset()
            writer.add_scalar("eval/episode_reward", episode_reward, episode_idx)
            writer.add_scalar("eval/episode_length", episode_length, episode_idx)
            episode_reward = 0
            episode_length = 0
            episode_idx += 1

            if episode_idx % args.eval_update_interval == 0:
                data = param_socket.recv(copy=False)
                param = pickle.loads(data)
                model.load_state_dict(param)


if __name__ == '__main__':
    main()
