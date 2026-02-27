# Original code Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the original license in the LICENSE file.
#
# Modified in 2025
#   - Added CAE, CAE+, CAE+ w/o U algorithms
#   - Simplified unrelated parts
#
# Additional modifications are released under CC BY-NC 4.0


from __future__ import division

import math

import torch
import typing
import gym
import threading
from torch import multiprocessing as mp
import logging
import traceback
import os
import numpy as np
import copy
import nle
import minihack
import pdb
import time
import contextlib
import termios
import tty
import gc

from nle import nethack

from src.core import prof
from src.env_utils import FrameStack, Environment


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


# computing how many objects
def num_objects(frames):
    T, B, H, W, *_ = frames.shape
    num_objects = frames[:, :, :, :, 0]
    num_objects = (num_objects == 4).long() + (num_objects == 5).long() + (num_objects == 6).long() + (num_objects == 7).long() + (num_objects == 8).long()
    return num_objects


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )




log = logging.getLogger("torchbeast")
log.propagate = False
log.setLevel(logging.INFO)

if not log.handlers:
    shandle = logging.StreamHandler()
    shandle.setFormatter(
        logging.Formatter(
            "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s"
        )
    )
    log.addHandler(shandle)


Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    if 'MiniHack' in flags.env:
        # seed = int.from_bytes(os.urandom(4), byteorder="little")
        kwargs = {
            "observation_keys": ("glyphs", "blstats", "chars", "message"),
            "savedir": None,
        }

        env = gym.make(flags.env, **kwargs)
        return env

    else:
        raise Exception("only MiniHack is supported !!!")


def get_batch(free_queue: mp.Queue,
    full_queue: mp.Queue,
    buffers: Buffers,
    initial_agent_state_buffers: typing.List[typing.List[torch.Tensor]],
    flags,
    timings,
    lock: threading.Lock = None):

    if lock is None:
        lock = threading.Lock()

    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time('dequeue')

    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }

    initial_agent_state = [
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    ]

    timings.time('batch')

    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')

    batch = {
        k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()
    }

    initial_agent_state = [t.to(device=flags.device, non_blocking=True) for t in initial_agent_state]
    timings.time('device')

    return batch, tuple(initial_agent_state)




def create_heatmap_buffers(obs_shape):
    specs = []
    for r in range(obs_shape[0]):
        for c in range(obs_shape[1]):
            specs.append(tuple([r, c]))
    buffers: Buffers = {key: torch.zeros(1).share_memory_() for key in specs}
    return buffers


def create_buffers(obs_space, num_actions, flags) -> Buffers:
    T = flags.unroll_length

    if type(obs_space) is gym.spaces.dict.Dict:
        size = (flags.unroll_length + 1,)
        # Get specimens to infer shapes and dtypes.
        samples = {k: torch.from_numpy(v) for k, v in obs_space.sample().items()}
        specs = {
            key: dict(size=size + sample.shape, dtype=sample.dtype)
            for key, sample in samples.items()
        }
        specs.update(
            policy_hiddens=dict(size=(T + 1, flags.hidden_dim), dtype=torch.float32),
            reward=dict(size=size, dtype=torch.float32),
            bonus_reward=dict(size=size, dtype=torch.float32),
            bonus_reward2=dict(size=size, dtype=torch.float32),
            done=dict(size=size, dtype=torch.bool),
            episode_return=dict(size=size, dtype=torch.float32),
            episode_step=dict(size=size, dtype=torch.int32),
            policy_logits=dict(size=size + (num_actions,), dtype=torch.float32),
            episode_state_count=dict(size=(T + 1,), dtype=torch.float32),
            global_state_count=dict(size=(T + 1,), dtype=torch.float32),
            baseline=dict(size=size, dtype=torch.float32),
            last_action=dict(size=size, dtype=torch.int64),
            action=dict(size=size, dtype=torch.int64),
            state_visits=dict(size=size, dtype=torch.int32),
        )

    else:
        obs_shape = obs_space.shape
        specs = dict(
            frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
            reward=dict(size=(T + 1,), dtype=torch.float32),
            bonus_reward=dict(size=(T + 1,), dtype=torch.float32),
            bonus_reward2=dict(size=(T + 1,), dtype=torch.float32),
            done=dict(size=(T + 1,), dtype=torch.bool),
            episode_return=dict(size=(T + 1,), dtype=torch.float32),
            episode_step=dict(size=(T + 1,), dtype=torch.int32),
            last_action=dict(size=(T + 1,), dtype=torch.int64),
            policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
            baseline=dict(size=(T + 1,), dtype=torch.float32),
            action=dict(size=(T + 1,), dtype=torch.int64),
            episode_win=dict(size=(T + 1,), dtype=torch.int32),
            carried_obj=dict(size=(T + 1,), dtype=torch.int32),
            carried_col=dict(size=(T + 1,), dtype=torch.int32),
            partial_obs=dict(size=(T + 1, 7, 7, 3), dtype=torch.uint8),
            episode_state_count=dict(size=(T + 1,), dtype=torch.float32),
            global_state_count=dict(size=(T + 1,), dtype=torch.float32),
            partial_state_count=dict(size=(T + 1,), dtype=torch.float32),
            encoded_state_count=dict(size=(T + 1,), dtype=torch.float32),
        )

    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def extract_state_key(env_output, flags):
    if flags.episodic_bonus_type != 'none':
        bonus_type = flags.episodic_bonus_type
    else:
        bonus_type = flags.global_bonus_type

    if bonus_type == 'counts-obs':
        # full observation: glyph image + stats + message
        state_key = tuple(env_output['glyphs'].view(-1).tolist() + env_output['blstats'].view(-1).tolist() + env_output['message'].view(-1).tolist())
    elif bonus_type == 'counts-msg':
        # message only
        state_key = tuple(env_output['message'].view(-1).tolist())
    elif bonus_type == 'counts-glyphs':
        # glyph image only
        state_key = tuple(env_output['glyphs'].view(-1).tolist())
    elif bonus_type == 'counts-pos':
        # (x, y) position extracted from the stats vector
        state_key = tuple(env_output['blstats'].view(-1).tolist()[:2])
    elif bonus_type == 'counts-img':
        # pixel image (for Vizdoom)
        state_key = tuple(env_output['frame'].contiguous().view(-1).tolist())
    else:
        state_key = ()

    return state_key


def act(i: int,
        free_queue: mp.Queue,
        full_queue: mp.Queue,
        model: torch.nn.Module,
        encoder: torch.nn.Module,
        buffers: Buffers,
        episode_state_count_dict: dict,
        global_state_count_dict: dict,
        initial_agent_state_buffers,
        flags):
    try:

        log.info('Actor %i started.', i)
        timings = prof.Timings()

        gym_env = create_env(flags)
        seed = i ^ flags.seed
        gym_env.seed(seed)

        if flags.num_input_frames > 1:
            gym_env = FrameStack(gym_env, flags.num_input_frames)

        env = Environment(gym_env, fix_seed=flags.fix_seed, env_seed=flags.env_seed)

        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)

        agent_output, unused_state = model(env_output, agent_state)

        rank1_update = True

        if flags.episodic_bonus_type in ['elliptical-icm', 'cae', 'caep', 'caep_wo_u', 'caep_root']:

            if flags.episodic_bonus_type in ['caep', 'caep_root']:
                hidden_dim = flags.weighted_emb_size + gym_env.action_space.n
            elif flags.episodic_bonus_type in ['cae', 'caep_wo_u']:
                hidden_dim = flags.hidden_dim + gym_env.action_space.n
            else:
                hidden_dim = flags.hidden_dim

            if rank1_update:
                cov_inverse = torch.eye(hidden_dim) * (1.0 / flags.ridge)
            else:
                cov = torch.eye(hidden_dim) * flags.ridge
            outer_product_buffer = torch.empty(hidden_dim, hidden_dim)


        step = 0


        while True:
            index = free_queue.get()
            if index is None:
                break

            # write old rollout
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                if key == 'emb':
                    continue
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # reset the episode state counts or covariance matrix when the episode is over
            # TODO: possibly redundant check as 'done' was already handled at each step ...
            if env_output['done'][0][0]:
                step = 0

                if flags.episodic_bonus_type in ['elliptical-icm', 'caep', 'caep_wo_u', 'cae', 'caep_root']:
                    if flags.episodic_bonus_type in ['caep', 'caep_root']:
                        hidden_dim = flags.weighted_emb_size + gym_env.action_space.n
                    elif flags.episodic_bonus_type in ['caep_wo_u', 'cae']:
                        hidden_dim = flags.hidden_dim + gym_env.action_space.n
                    else:
                        hidden_dim = flags.hidden_dim

                    if rank1_update:
                        cov_inverse = torch.eye(hidden_dim) * (1.0 / flags.ridge)
                    else:
                        cov = torch.eye(hidden_dim) * flags.ridge

            # do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)
                    if flags.episodic_bonus_type in ['elliptical-icm', 'caep_wo_u', 'cae']:
                        encoder_output, encoder_state = encoder(env_output, tuple())
                    if flags.episodic_bonus_type in ['caep', 'caep_root']:
                        encoder_output, encoder_state = encoder.weight_forward(env_output, tuple())

                timings.time('model')

                env_output = env.step(agent_output['action'])

                timings.time('step')

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]

                for key in agent_output:
                    if key == 'emb':
                        continue
                    buffers[key][index][t + 1, ...] = agent_output[key]


                if flags.episodic_bonus_type == 'none':
                    b = 0

                elif flags.episodic_bonus_type in ['elliptical-icm', 'caep', 'caep_wo_u', 'cae', 'caep_root']:
                    h = encoder_output.squeeze().detach()

                    if flags.episodic_bonus_type in ['caep', 'caep_wo_u', 'cae', 'caep_root']:
                        one_hot_action = torch.zeros(gym_env.action_space.n)
                        one_hot_action[agent_output['action'].item()] = 1
                        h = torch.cat((h, one_hot_action))

                    if rank1_update:
                        u = torch.mv(cov_inverse, h)
                        b = torch.dot(h, u).item()

                        torch.outer(u, u, out=outer_product_buffer)
                        torch.add(
                            cov_inverse,
                            outer_product_buffer,
                            alpha=-(1.0 / (1.0 + b)),
                            out=cov_inverse
                        )
                    else:
                        cov += torch.outer(h, h)
                        cov_inverse = torch.inverse(cov)

                        u = torch.mv(cov_inverse, h)
                        b = torch.dot(h, u).item()

                else:
                    raise Exception('bonus type not supported !!!')

                if step == 0:
                    b = 0

                if flags.episodic_bonus_type == 'caep_root':
                    buffers['bonus_reward'][index][t + 1, ...] = math.sqrt(abs(b))
                else:
                    buffers['bonus_reward'][index][t + 1, ...] = b


                step += 1


                timings.time('bonus update')
                # reset the episode covariance
                if env_output['done'][0][0]:
                    step = 0

                    if flags.episodic_bonus_type in ['elliptical-icm', 'caep', 'caep_wo_u', 'cae', 'caep_root']:
                        if flags.episodic_bonus_type in ['caep', 'caep_root']:
                            hidden_dim = flags.weighted_emb_size + gym_env.action_space.n
                        elif flags.episodic_bonus_type in ['caep_wo_u', 'cae']:
                            hidden_dim = flags.hidden_dim + gym_env.action_space.n
                        else:
                            hidden_dim = flags.hidden_dim

                        if rank1_update:
                            cov_inverse = torch.eye(hidden_dim) * (1.0 / flags.ridge)
                        else:
                            cov = torch.eye(hidden_dim) * flags.ridge

                timings.time('write')

            full_queue.put(index)

        if i == 0:
            log.info('Actor %i: %s', i, timings.summary())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
