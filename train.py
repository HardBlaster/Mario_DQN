import argparse
import datetime
import os

import gym_super_mario_bros
import torch
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from actor import Mario
from gym_wrappers import SkipFrame, GrayScaleObs, ResizeObs
from logger import MetricLogger
from model import DDQN


def train(train_params):
    save_dir = os.path.join(train_params.save_dir, datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
    logger = MetricLogger(os.path.join(save_dir, "log"))

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env = SkipFrame(env, skip=train_params.skip_frame)
    env = GrayScaleObs(env)
    env = ResizeObs(env, shape=train_params.frame_size)
    env = FrameStack(env, num_stack=train_params.stack_frame)

    model = DDQN((train_params.skip_frame, train_params.stack_frame, train_params.stack_frame), env.action_space.n)
    if train_params.weights:
        actor_state = torch.load(train_params.weights)  # map_location=torch.device('cpu')
        state_dict = actor_state['model']
        expl_rate = actor_state['exploration_rate']

        model.load_state_dict(state_dict)
        model.exploration_rate = expl_rate
    mario = Mario(model, env.action_space.n, os.path.join(save_dir, "checkpoints"),
                  cuda=torch.cuda.is_available(),
                  memory_size=train_params.memory_size,
                  batch_size=train_params.batch_size,
                  lr=train_params.lr,
                  exploration_rate=train_params.exploration_rate,
                  er_dec=train_params.er_decrease,
                  er_min=train_params.er_min,
                  random_steps=train_params.random_steps,
                  learn_freq=train_params.learn_freq,
                  sync_freq=train_params.sync_freq,
                  save_freq=train_params.save_freq)

    for e in range(1, train_params.episodes + 1):
        state = env.reset()

        for i in range(1, train_params.max_steps + 1):
            action = mario.act(state)
            
            next_state, reward, done, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)

            q, loss = mario.learn()

            logger.log_step(reward, loss, q)
            
            state = next_state

            if done or info["flag_get"]:
                break

        logger.log_episode()
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.current_step)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--heuristic', type=bool, default=False)
    arg_parser.add_argument('--memory_dir', type=str, default=None)
    

    arg_parser.add_argument('--episodes', type=int, default=int(2e4))
    arg_parser.add_argument('--save_freq', type=int, default=int(5e4))
    arg_parser.add_argument('--save_dir', type=str, default='trains')

    arg_parser.add_argument('--weights', type=str, default=None)
    arg_parser.add_argument('--memory_size', type=int, default=int(1e5))
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--exploration_rate', type=float, default=1)
    arg_parser.add_argument('--er_min', type=float, default=.1)
    arg_parser.add_argument('--er_decrease', type=float, default=1e-5)
    arg_parser.add_argument('--lr', type=float, default=25e-5)
    arg_parser.add_argument('--skip_frame', type=int, default=4)
    arg_parser.add_argument('--stack_frame', type=int, default=4)
    arg_parser.add_argument('--frame_size', type=int, default=84)
    arg_parser.add_argument('--learn_freq', type=int, default=16)
    arg_parser.add_argument('--sync_freq', type=int, default=1e4)
    arg_parser.add_argument('--random_steps', type=int, default=1e4)
    arg_parser.add_argument('--max_steps', type=int, default=int(5e4))

    params = arg_parser.parse_args()
    train(params)
