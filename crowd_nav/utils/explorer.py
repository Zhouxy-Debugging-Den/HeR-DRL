import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *
import pandas as pd
from openpyxl import load_workbook
import datetime
import time

class Explorer(object):
    def __init__(self, env, robot, device, writer, memory=None, gamma=None,target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None
        self.use_noisy_net = False
        # self.state_rotated=state_rotated

    # @torch.compile
    def run_k_episodes(self, k, phase, model_dir=None,update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        returns_list = []
        collision_cases = []
        timeout_cases = []
        discomfort_nums = []
        # 添加进度条
        if phase in ['test', 'val']:
            pbar = tqdm(total=k)
        else:
            pbar = None
        # 探索过程
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            dones = []
            num_discoms =[]
            if phase in ['test', 'val']:
                dropout=False
            else:
                dropout=True
            while not done:
                num_discom = 0

                action, action_index = self.robot.act(ob,dropout)
                ob, reward, done, info = self.env.step(action)


                states.append(self.robot.policy.last_state)
                actions.append(action_index)

                if isinstance(info, Timeout):
                    dones.append(False)
                else:
                    dones.append(done)
                rewards.append(reward)
                if isinstance(info, Discomfort):
                    # 多少步是不舒服
                    discomfort += 1
                    # 单步最小HR距离
                    min_dist.append(info.min_dist)
                    # 和多少行人产生了不舒服
                    num_discom = info.num
                num_discoms.append(num_discom)
            # add the terminal state
            states.append(self.robot.get_state(ob))
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                if phase in ['test']:
                    print('collision happen %f', self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                if phase in ['test']:
                    print('timeout happen %f', self.env.global_time)
                    rewards[-1] = rewards[-1]-0.25
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                # if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                self.update_memory(states, actions, rewards, dones, imitation_learning)
            discomfort_nums.append(sum(num_discoms))
            # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
            #                                * reward for t, reward in enumerate(rewards)]))
            cumulative_rewards.append(sum(rewards))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            returns_list = returns_list + returns
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.3f}, collision rate: {:.3f}, nav time: {:.3f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate,
                                                       avg_nav_time, sum(cumulative_rewards),
                                                       average(average_returns)))
        # if phase in ['val', 'test'] or imitation_learning:
        total_time = sum(success_times + collision_times + timeout_times) / self.robot.time_step
        logging.info('Frequency of being in danger: %.3f and average min separate distance in danger: %.2f',
                    discomfort / total_time, average(min_dist))
        logging.info('discomfor nums is %.0f and return is %.04f and length is %.0f', sum(discomfort_nums),
                     average(returns_list), len(returns_list))
        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        if phase=="test":
            formatted_time=int(datetime.datetime.now().strftime("%Y%m%d%H%M"))
            data={'time':[formatted_time],'success':[success_rate],'collision':[collision_rate],'time':[avg_nav_time],
                  'danger nums':[sum(discomfort_nums)],'danger fre':[discomfort / total_time],'min dis':[average(min_dist)],
                  'avg return':[average(average_returns)],'total reward':[sum(cumulative_rewards)]
                  }
            df=pd.DataFrame(data)
            #todo: 差传达路径，以及优化记录内容
            test_doc_dir=model_dir+'/test.xlsx'
            if os.path.exists(test_doc_dir):
                wb = load_workbook(test_doc_dir)
                ws = wb.active
                list1=[]
                for w in list(data.values()):
                    list1.append(w[0])
                ws.append(list1)
                wb.save(test_doc_dir)
            else:
                with pd.ExcelWriter(test_doc_dir) as Writer:
                    df.to_excel(Writer,sheet_name='Sheet1',index=False)
        self.statistics = success_rate, collision_rate, avg_nav_time, sum(cumulative_rewards), average(average_returns), discomfort, total_time

        return self.statistics

    def update_memory(self, states, actions, rewards, dones, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]
            # VALUE UPDATE
            next_state = states[i+1]
            action = actions[i]
            if i == len(states) - 1:
                # terminal state
                value = reward
            else:
                value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)
            done = torch.Tensor([dones[i]]).to(self.device)
            # 这里要推送的state要包含几帧内容
            self.memory.push((state, value, done, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return,_,_ = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
