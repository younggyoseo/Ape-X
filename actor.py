import _pickle as pickle
import os
from multiprocessing import Process, Queue
import queue

import zmq
import torch
from tensorboardX import SummaryWriter
import numpy as np

import utils
from memory import BatchStorage
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from arguments import argparser


def get_environ():
    actor_id = int(os.environ.get('ACTOR_ID', '-1'))
    n_actors = int(os.environ.get('N_ACTORS', '-1'))
    replay_ip = os.environ.get('REPLAY_IP', '-1')
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    assert (actor_id != -1 and n_actors != -1)
    assert (replay_ip != '-1' and learner_ip != '-1')
    return actor_id, n_actors, replay_ip, learner_ip


def connect_param_socket(ctx, param_socket, learner_ip, actor_id):
    socket = ctx.socket(zmq.REQ)
    socket.connect("tcp://{}:52002".format(learner_ip))
    socket.send(pickle.dumps((actor_id, 1)))
    socket.recv()
    param_socket.connect('tcp://{}:52001'.format(learner_ip))
    socket.send(pickle.dumps((actor_id, 2)))
    socket.recv()
    print("Successfully connected to learner!")
    socket.close()


def recv_param(learner_ip, actor_id, param_queue):
    ctx = zmq.Context()
    param_socket = ctx.socket(zmq.SUB)
    param_socket.setsockopt(zmq.SUBSCRIBE, b'')
    param_socket.setsockopt(zmq.CONFLATE, 1)
    connect_param_socket(ctx, param_socket, learner_ip, actor_id)
    while True:
        data = param_socket.recv(copy=False)
        param = pickle.loads(data)
        param_queue.put(param)


def exploration(args, actor_id, n_actors, replay_ip, param_queue):
    ctx = zmq.Context()
    batch_socket = ctx.socket(zmq.DEALER)
    batch_socket.setsockopt(zmq.IDENTITY, pickle.dumps('actor-{}'.format(actor_id)))
    batch_socket.connect('tcp://{}:51001'.format(replay_ip))
    outstanding = 0

    writer = SummaryWriter(comment="-{}-actor{}".format(args.env, actor_id))

    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    seed = args.seed + actor_id
    utils.set_global_seeds(seed, use_torch=True)
    env.seed(seed)

    model = DuelingDQN(env)
    epsilon = args.eps_base ** (1 + actor_id / (n_actors - 1) * args.eps_alpha)
    storage = BatchStorage(args.n_steps, args.gamma)

    param = param_queue.get(block=True)
    model.load_state_dict(param)
    param = None
    print("Received First Parameter!")

    episode_reward, episode_length, episode_idx, actor_idx = 0, 0, 0, 0
    state = env.reset()
    while True:
        action, q_values = model.act(torch.FloatTensor(np.array(state)), epsilon)
        next_state, reward, done, _ = env.step(action)
        storage.add(state, reward, action, done, q_values)

        state = next_state
        episode_reward += reward
        episode_length += 1
        actor_idx += 1

        if done or episode_length == args.max_episode_length:
            state = env.reset()
            writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
            writer.add_scalar("actor/episode_length", episode_length, episode_idx)
            episode_reward = 0
            episode_length = 0
            episode_idx += 1

        if actor_idx % args.update_interval == 0:
            try:
                param = param_queue.get(block=False)
                model.load_state_dict(param)
                print("Updated Parameter..")
            except queue.Empty:
                pass

        if len(storage) == args.send_interval:
            batch, prios = storage.make_batch()
            data = pickle.dumps((batch, prios))
            batch, prios = None, None
            storage.reset()
            while outstanding >= args.max_outstanding:
                batch_socket.recv()
                outstanding -= 1
            batch_socket.send(data, copy=False)
            outstanding += 1
            print("Sending Batch..")


def main():
    actor_id, n_actors, replay_ip, learner_ip = get_environ()
    args = argparser()
    param_queue = Queue(maxsize=3)

    procs = [
        Process(target=exploration, args=(args, actor_id, n_actors, replay_ip, param_queue)),
        Process(target=recv_param, args=(learner_ip, actor_id, param_queue)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return True


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
