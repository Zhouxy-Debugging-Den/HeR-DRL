import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.single.cadrl import CADRL
from crowd_sim.envs.utils.state import JointState


class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()
    # 思考多帧部分这里该如何修改，最后到model的部分
    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        # 判断state_list中最新的state是否抵达目的地
        if self.reach_destination(state[len(state)-1]):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state[len(state)-1].robot_state.v_pref)
        if not state[len(state)-1].human_states:
            assert self.phase != 'train'
            if hasattr(self, 'attention_weights'):
                self.attention_weights = list()
            return self.select_greedy_action(state[len(state)-1].robot_state)
        max_action_index = 0
        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action_index = np.random.choice(len(self.action_space))
            max_action = self.action_space[max_action_index]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            # rotated_batch_input=None
            rewards = []
            action_index = 0
            batch_input_tensor = None
            batch_tensor = None
            for action in self.action_space:
                next_robot_state = self.propagate(state[len(state)-1].robot_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                    rewards.append(reward)
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                         for human_state in state[len(state)-1].human_states]
                    next_state = JointState(next_robot_state, next_human_states)
                    reward, _ = self.reward_estimator.estimate_reward_on_predictor(state, next_state)
                    rewards.append(reward)
                batch_next_states = torch.cat([torch.Tensor([next_robot_state + next_human_state]).to(self.device)
                                              for next_human_state in next_human_states], dim=0)

                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0).unsqueeze(0)
                if batch_tensor is None:
                    for i in range(len(state) - 1):
                        mid_states = torch.cat([torch.Tensor([state[i + 1].robot_state + human_state]).to(self.device)
                                                for human_state in state[i + 1].human_states], dim=0)
                        rotated_mid_states = self.rotate(mid_states).unsqueeze(0).unsqueeze(0)
                        if batch_tensor is None:
                            batch_tensor = rotated_mid_states
                        else:
                            batch_tensor = torch.cat([batch_tensor, rotated_mid_states], dim=1)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                rotated_batch_input=torch.cat([batch_tensor,rotated_batch_input], dim=1)
                if batch_input_tensor is None:
                    batch_input_tensor = rotated_batch_input
                else:
                    batch_input_tensor = torch.cat([batch_input_tensor, rotated_batch_input], dim=0)
            dropout=0
            next_value = self.model(batch_input_tensor,dropout).squeeze(1)
            # para_number = sum(p.numel() for p in self.model.parameters())
            rewards_tensor = torch.tensor(rewards).to(self.device)
            value = rewards_tensor + next_value * pow(self.gamma, self.time_step * state[len(state)-1].robot_state.v_pref)
            max_action_index = value.argmax()
            best_value = value[max_action_index]
            if best_value > max_value:
                max_action = self.action_space[max_action_index]

            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state_tensor = self.transform(state[len(state)-1])
        self.last_state=state[len(state)-1].human_states
        return max_action, int(max_action_index)

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.robot_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        rotated_state_tensor = self.rotate(state_tensor)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            rotated_state_tensor = torch.cat([rotated_state_tensor, occupancy_maps], dim=1)

        return rotated_state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                          for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()
