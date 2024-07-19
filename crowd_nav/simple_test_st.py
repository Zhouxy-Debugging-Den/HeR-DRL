import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer_st import Explorer
from crowd_nav.policy.single.ST.policy_factory_st import policy_factory
from crowd_sim.envs.utils.robot_st import Robot
from crowd_sim.envs.policy.single.orca import ORCA
from crowd_sim.envs.utils.info import *
import sys


def main(args):
    # configure logging and device,配置logging和device
    # output_str=str(args.human_num)+'H_test_output_'+time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))+'.log'
    output_str = str(args.human_num) + 'H_test_output' + '.log'
    log_file = os.path.join(args.model_dir, output_str)
    file_handler = logging.FileHandler(log_file, mode='a')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # device = torch.device("cpu")
    logging.info('Using device: %s', device)
    logging.info('Using randomseed: %d', args.randomseed)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                print(os.listdir(args.model_dir))
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            # 加载最佳验证参数
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy,配置策略
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    # 这里定义奖励估计器，实际我们只是使用环境自带的奖励(这部分是否可以注释掉)
    # reward_estimator = Reward_Estimator()
    # env_config = config.EnvConfig(args.debug)
    # reward_estimator.configure(env_config)
    # policy.reward_estimator = reward_estimator



    # configure environment，配置环境
    env_config = config.EnvConfig(args.debug)

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    env_config.sim.random_seed_base = args.randomseed
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    logging.info('human_num: %d',args.human_num)

    # 策略加载模型
    policy.configure(policy_config, device,env_config.sim.human_num)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)
    if args.square:
        env.test_scenario = 'square_crossing'
    if args.circle:
        env.test_scenario = 'circle_crossing'
    if args.test_scenario is not None:
        env.test_scenario = args.test_scenario

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    st_frame_nums=3
    explorer = Explorer(env, robot, device, None,st_frame_nums=st_frame_nums, gamma=0.9)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)

    # set safety space for orca in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = args.safety_space
        else:
            robot.policy.safety_space = args.safety_space
        logging.info('orca agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    # 打印行人数量等环境信息


    if args.visualize:
        rewards = []
        actions = []
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        ob_start = ob
        ob_frame = []
        states = []
        while not done:
            num_discom = 0
            # 存在问题，就是其中states其实是tensor
            if len(states) >= 2:
                for i in range(2):
                    ob_frame.append(states[len(states) - 2 + i])
                ob_frame.append(ob)
            else:
                for i in range(2 - len(states)):
                    ob_frame.append(ob_start)
                for i in range(len(states)):
                    ob_frame.append(states[i])
                ob_frame.append(ob)
            action, action_index = robot.act(ob_frame)
            ob_frame = []
            ob, _, done, info = env.step(action)
            states.append(robot.policy.last_state)
            if isinstance(info, Timeout):
                _ = _ - 0.25
            rewards.append(_)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            actions.append(action)
        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
             * reward for t, reward in enumerate(rewards)])

        if args.traj:
            vedio_file = "new_pic/simple/" + str(
                args.test_case) + "_" + policy.name + ".png"
            env.render('traj', vedio_file)
        else:
            # 下面针对vedio需要进行好好捋一捋
            if args.video_dir is not None:
                if policy_config.name == 'gcn':
                    args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + policy_config.gcn.similarity_function)
                else:
                    args.video_file = os.path.join(args.video_dir, policy_config.name)
                args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            env.render('video', args.video_file)
        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        if args.plot_test_scenarios_hist:
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
            plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
            plt.close()

    positions = []
    velocity_rec = []
    rotation_rec = []
    for i in range(len(actions)):
        positions.append(i)
        action = actions[i]
        velocity_rec.append(action.v)
        rotation_rec.append(action.r)
    plt.plot(positions, velocity_rec, color='r', marker='.', linestyle='dashed')
    plt.plot(positions, rotation_rec, color='b', marker='.', linestyle='dashed')
    # plt.show()
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='graph_all_undirectional')
    parser.add_argument('-m', '--model_dir', type=str, default='data_thesis/simple/graphgnn')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=10)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--randomseed', type=int, default=7)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=5)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    main(sys_args)
