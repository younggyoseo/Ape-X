"""
Module for replay buffer server in Ape-X. Implemented with Asyncio.
"""
import _pickle as pickle
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import zmq
from zmq.asyncio import Context

import utils
from memory import CustomPrioritizedReplayBuffer
from arguments import argparser


def push_batch(buffer, data):
    """
    support function to push batch samples to buffer
    """
    batch, prios = pickle.loads(data)
    for sample in zip(*batch, prios):
        buffer.add(*sample)
    batch, prios = None, None


def update_prios(buffer, data):
    """
    support function to update priorities to buffer
    """
    idxes, prios = pickle.loads(data)
    buffer.update_priorities(idxes, prios)
    idxes, prios = None, None


def sample_batch(buffer, batch_size, beta):
    """
    support function to update priorities to buffer
    """
    batch = buffer.sample(batch_size, beta)
    data = pickle.dumps(batch)
    batch = None
    return data


def recv_batch_device():
    ctx = zmq.Context()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind("tcp://*:51001")
    backend = ctx.socket(zmq.DEALER)
    backend.bind("ipc:///tmp/5101.ipc")
    zmq.proxy(frontend, backend)
    return True


def recv_prios_device():
    ctx = zmq.Context()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind("tcp://*:51002")
    backend = ctx.socket(zmq.DEALER)
    backend.bind("ipc:///tmp/5102.ipc")
    zmq.proxy(frontend, backend)
    return True


def send_batch_device():
    ctx = zmq.Context()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind("tcp://*:51003")
    backend = ctx.socket(zmq.DEALER)
    backend.bind("ipc:///tmp/5103.ipc")
    zmq.proxy(frontend, backend)


async def recv_batch_worker(buffer, exe, event, lock, threshold_size):
    """
    coroutine to receive batch from actors
    """
    loop = asyncio.get_event_loop()
    ctx = Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.connect("ipc:///tmp/5101.ipc")

    start = False
    cnt = 0
    ts = time.time()

    while True:
        identity, data = await socket.recv_multipart(copy=False)
        async with lock:
            await loop.run_in_executor(exe, push_batch, buffer, data)
        await socket.send_multipart((identity, b''))
        # TODO: 1. Only one worker should print log to console.
        #       2. Hard-coded part in (50 * cnt * 4) should be fixed.
        data = None
        cnt += 1
        if cnt % 100 == 0:
            print("Buffer Size: {} / FPS: {:.2f}".format(
                len(buffer), (50 * cnt * 4) / (time.time() - ts)
            ))
            ts = time.time()
            if not start and len(buffer) >= threshold_size:
                start = True
                event.set()
    return True


async def recv_prios_worker(buffer, exe, event, lock):
    """
    coroutine to receive priorities from learner
    """
    loop = asyncio.get_event_loop()
    ctx = Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.connect("ipc:///tmp/5102.ipc")
    await event.wait()
    while True:
        identity, data = await socket.recv_multipart(copy=False)
        async with lock:
            await loop.run_in_executor(exe, update_prios, buffer, data)
        await socket.send_multipart((identity, b''))
        data = None
    return True


async def send_batch_worker(buffer, exe, event, lock, batch_size, beta):
    """
    coroutine to send training batches to learner
    """
    seed = int(str(time.time())[-4:])
    utils.set_global_seeds(seed, use_torch=False)
    loop = asyncio.get_event_loop()
    ctx = Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.connect("ipc:///tmp/5103.ipc")
    await event.wait()
    while True:
        identity, _ = await socket.recv_multipart(copy=False)
        # TODO: Is there any other greay way to support lock but make sampling faster?
        async with lock:
            batch = await loop.run_in_executor(exe, sample_batch, buffer, batch_size, beta)
        await socket.send_multipart([identity, batch], copy=False)
        batch = None
    return True


async def main():
    """
    main event loop
    """
    args = argparser()
    utils.set_global_seeds(args.seed, use_torch=False)

    procs = [
        Process(target=recv_batch_device),
        Process(target=recv_prios_device),
        Process(target=send_batch_device),
    ]
    for p in procs:
        p.start()

    buffer = CustomPrioritizedReplayBuffer(args.replay_buffer_size, args.alpha)
    exe = ThreadPoolExecutor()
    event = asyncio.Event()
    lock = asyncio.Lock()

    # TODO: How to decide the proper number of asyncio workers?
    workers = []
    for _ in range(args.n_recv_batch_worker):
        w = recv_batch_worker(buffer, exe, event, lock, args.threshold_size)
        workers.append(w)
    for _ in range(args.n_recv_prios_worker):
        w = recv_prios_worker(buffer, exe, event, lock)
        workers.append(w)
    for _ in range(args.n_send_batch_worker):
        w = send_batch_worker(buffer, exe, event, lock, args.batch_size, args.beta)
        workers.append(w)

    await asyncio.gather(*workers)
    return True


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    asyncio.run(main())
