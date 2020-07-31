import gym
import envs
import argparse
import torch
import configparser


from envs.crowd_sim.utils.robot import Robot
from envs.crowd_sim.utils.human import Human
from envs.crowd_nav.utils.explorer import Explorer
from envs.crowd_nav.policy.policy_factory import policy_factory

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='envs/crowd_nav/configs/env.config')
    parser.add_argument('--policy', type=str, default='sarl')
    parser.add_argument('--policy_config', type=str, default='envs/crowd_nav/configs/policy.config')
    # parser.add_argument('--train_config', type=str, default='configs/train.config')
    # parser.add_argument('--output_dir', type=str, default='data/output')
    # parser.add_argument('--weights', type=str)
    # parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    # parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # print ('Using device: {}'.format(device))
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    policy.set_device(device)
    env = gym.make('CrowdSim-v0')


    # set up the robot
    policy.set_env(env)
    robot = Robot()
    robot.configure(args.env_config, 'robot')
    env.set_robot(robot)
    robot.set_policy(policy)
    robot.policy.set_epsilon(0.1)
    # set up the human
    humans = [Human() for _ in range(env.human_num)]
    for human in humans:
        human.configure(args.env_config, 'humans')
    env.set_humans(humans)

    
    explorer = Explorer(env, robot, policy.gamma, target_policy=policy)
    explorer.run_k_episodes(300, 'train', update_memory=False, imitation_learning=True)

if __name__ == '__main__':
    main()