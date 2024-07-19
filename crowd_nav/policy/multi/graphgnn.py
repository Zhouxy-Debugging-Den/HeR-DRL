import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from crowd_nav.policy.multi.cadrl import mlp
from crowd_nav.policy.multi.multi_human_rl import MultiHumanRL
from torch_geometric.nn import GraphConv,GCNConv
from torch_geometric.data import Data
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state_multi import JointState



"""
本策略主要构建的是所有agent双向连接的同构图
"""
class Graphgnn(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(Graphgnn, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)

        multi_gnn=False
        if multi_gnn:
            self.conv2 = GraphConv(in_channels, hidden_channels)
            self.conv3 = GraphConv(in_channels, hidden_channels)
            self.conv4 = GraphConv(in_channels, hidden_channels)
            self.fc1 = torch.nn.Linear(32, 32)
            self.fc2 = torch.nn.Linear(32, 32)
            self.fc3=torch.nn.Linear(2 * 32, 32)
        else:
            self.conv2 = GraphConv(in_channels, out_channels)




    def forward(self, x, edge_index,dropout):
        # 这里其实应该传入edge_weight，没有传入，那么就是按照全是1进行计算的
        multi_gnn = False
        if multi_gnn:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, 0.25, training=dropout)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, 0.25, training=dropout)
            x_1=F.relu(self.fc1(x))
            x = F.relu(self.conv3(x, edge_index))
            x = F.dropout(x, 0.25, training=dropout)
            x = F.relu(self.conv4(x, edge_index))
            x = F.dropout(x, 0.25, training=dropout)
            x_2 = F.relu(self.fc2(x))
            x=torch.cat([x_1,x_2],dim=-1)
            x = F.relu(self.fc3(x))
            x = F.dropout(x,0.5,training=dropout)

        else:
            x = F.relu(self.conv1(x, edge_index))
            # x = F.dropout(x, 0.25, training=dropout)
            x = F.relu(self.conv2(x, edge_index))
            # x = F.dropout(x, 0.25, training=dropout)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim,X_dim,wr_dims,wh_dims,final_state_dim,gcn2_w1_dim,planning_dims,num_layer,other_robot_num,human_num,device):
        super().__init__()
        torch.manual_seed(1234)
        self.input_dim = input_dim
        self.self_state_dim = self_state_dim
        agent_state_dim = self.input_dim-self.self_state_dim
        self.agent_state_dim = agent_state_dim
        self.X_dim = X_dim
        self.num_layer=num_layer
        self.device=device
        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(agent_state_dim, wh_dims, last_relu=True)
        self.w_o = mlp(agent_state_dim, wh_dims, last_relu=True)
        self.human_num=human_num
        self.other_robot_num=other_robot_num
        self.conv = Graphgnn(X_dim, gcn2_w1_dim, final_state_dim)
        # self.value_net = mlp(self.self_state_dim+final_state_dim, planning_dims)
        self.value_net = mlp(final_state_dim, planning_dims)
    # edge_index这个是作为GraphConv网络的输入
    def edge_index(self,agent_number):
        d1=agent_number**2-agent_number
        edge_index = torch.zeros([2,d1],dtype=torch.long,device=self.device)
        k=0
        for i in range(agent_number):
            for j in range(agent_number):
                if i!=j:
                    edge_index[0][k]=i
                    edge_index[1][k]=j
                    k+=1
        return edge_index


    def forward(self, state_input,dropout):
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            # lengths = torch.IntTensor([state.size()[1]])

        self_state = state[:, 0, :self.self_state_dim]
        # 这里进行修改，其中不包括最后一个维度，查看一下训练结果
        human_states = state[:, 0:self.human_num, self.self_state_dim:]
        other_states=state[:,self.human_num:,self.self_state_dim:]

        # compute
        self_state_embedings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)
        other_state_embedings = self.w_o(other_states)
        X = torch.cat([self_state_embedings.unsqueeze(1), human_state_embedings,other_state_embedings], dim=1)
        # compute edge_index
        agent_number = X.size()[1]
        data=Data()
        # 初始化节点特征

        data.x=X[0,:,:]
        data.edge_index=self.edge_index(agent_number)

        h = self.conv(X, data.edge_index,dropout)
        h_final = h[:, 0, :]
        # joint_state = torch.cat([self_state, h_final], dim=1)
        # value = self.value_net(joint_state)
        value = self.value_net(h_final)
        return value


class graphgnn(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'HoR-DRL'

    def configure(self, config,device):
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        gcn2_w1_dim = config.gcn.gcn2_w1_dim
        planning_dims = config.gcn.planning_dims
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection
        other_robot_num=config.gcn.other_robot_num
        human_num=config.gcn.human_num
        self.device=device
        self.set_common_parameters(config)
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, X_dim, wr_dims, wh_dims,
                                  final_state_dim, gcn2_w1_dim, planning_dims,num_layer,other_robot_num,human_num,self.device)
        logging.info('self.model:{}'.format(self.model))
        logging.info('GCN layers: {}'.format(num_layer))
        logging.info('Policy: {}'.format(self.name))

    def get_matrix_A(self):
        return self.model.A
    def predict(self, state,droupout):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)
        if not state.agent_states:
            assert self.phase != 'train'
            if hasattr(self, 'attention_weights'):
                self.attention_weights = list()
            return self.select_greedy_action(state.robot_state)
        # max_action_index = 0
        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action_index = np.random.choice(len(self.action_space))
            max_action = self.action_space[max_action_index]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            rewards = []
            # action_index = 0
            batch_input_tensor = None
            for action in self.action_space:
                next_robot_state = self.propagate(state.robot_state, action)
                if self.query_env:
                    next_agent_states, reward, done, info = self.env.onestep_lookahead(action)
                    rewards.append(reward)
                else:
                    next_agent_states = [self.propagate(agent_state, ActionXY(agent_state.vx, agent_state.vy))
                                         for agent_state in state.agent_states]
                    next_state = JointState(next_robot_state, next_agent_states)
                    reward, _ = self.reward_estimator.estimate_reward_on_predictor(state, next_state)
                    rewards.append(reward)
                batch_next_states = torch.cat([torch.Tensor([next_robot_state + next_agent_state]).to(self.device)
                                              for next_agent_state in next_agent_states], dim=0)
                if self.state_rotated:
                    batch_input = self.rotate(batch_next_states).unsqueeze(0)
                else:
                    batch_input=batch_next_states.unsqueeze(0)
                # with_om部分
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_agent_states).unsqueeze(0)
                    batch_input = torch.cat([batch_input, occupancy_maps], dim=2)
                if batch_input_tensor is None:
                    batch_input_tensor = batch_input
                else:
                    batch_input_tensor = torch.cat([batch_input_tensor, batch_input], dim=0)
            next_value = self.model(batch_input_tensor,droupout).squeeze(1)
            # para_number = sum(p.numel() for p in self.model.parameters())
            rewards_tensor = torch.tensor(rewards).to(self.device)
            value = rewards_tensor + next_value * pow(self.gamma, self.time_step * state.robot_state.v_pref)
            max_action_index = value.argmax()
            best_value = value[max_action_index]
            if best_value > max_value:
                max_action = self.action_space[max_action_index]
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action, int(max_action_index)

