import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.single.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
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
            # if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
            #     model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            # else:
            #     print(os.listdir(args.model_dir))
            #     model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            # 加载哪一个model
            weigt_name='rl_model_'+str(args.model_pth)+'.pth'
            model_weights = os.path.join(args.model_dir, weigt_name)
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
    policy_file = os.path.join(args.model_dir, 'policy.py')
    spec_policy = importlib.util.spec_from_file_location('policy_ob', policy_file)
    if spec_policy is None:
        parser.error('policy file not found.')
    policy_ob = importlib.util.module_from_spec(spec_policy)
    spec_policy.loader.exec_module(policy_ob)
    policy = eval('policy_ob.' + policy_config.name + '()')

    # 策略加载模型
    policy.configure(policy_config, device)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    # configure environment，配置环境
    env_config = config.EnvConfig(args.debug)

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    env_config.sim.circle_radius=args.circle_radius
    env_config.env.time_limit=args.time_limit
    env_config.sim.random_seed_base = args.randomseed
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    logging.info('human_num: %d',args.human_num)

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
    explorer = Explorer(env, robot, device, None, gamma=0.9)

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
        while not done:
            action, action_index = robot.act(ob,dropout=0)
            ob, _, done, info = env.step(action)
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
        explorer.run_k_episodes(env.case_size[args.phase], args.phase,args.model_dir,print_failure=True)
        # explorer.run_k_episodes(2, args.phase,args.model_dir,print_failure=True)
        # explorer.run_k_episodes(2, args.phase, print_failure=True)
        if args.plot_test_scenarios_hist:
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
            plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
            plt.close()
    logging.info('test case: %s', args.test_case)
    positions = []
    velocity_rec = []
    rotation_rec = []
    for i in range(len(actions)):
        positions.append(i)
        action = actions[i]
        velocity_rec.append(action.v)
        rotation_rec.append(action.r)
    v_change = np.mean(abs(np.array(velocity_rec[1:]) - np.array(velocity_rec[:-1])))
    r_change = np.mean(abs(np.array(rotation_rec[1:]) - np.array(rotation_rec[:-1])))
    logging.info('avg v_change: %s', v_change)
    logging.info('avg r_change: %s', r_change)
    # plt.plot(positions, velocity_rec, color='r', marker='.', linestyle='dashed')
    # plt.plot(positions, rotation_rec, color='b', marker='.', linestyle='dashed')

    # plt.show()
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='graph_all_undirectional')
    parser.add_argument('-m', '--model_dir', type=str, default='data_thesis/simple/graphgnn')#None
    parser.add_argument('--il', default=False, action='store_true')
    # parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')   # 是否可视化
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=18)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default='new_pic')
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=5)
    parser.add_argument('--time_limit', type=int, default=30)
    parser.add_argument('--circle_radius', type=float, default=4.0)
    parser.add_argument('--rl',  default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=7)
    parser.add_argument('--model_pth', type=int, default=10)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    main(sys_args)
