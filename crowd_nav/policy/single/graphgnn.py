import logging
import torch
import torch.nn as nn
from crowd_nav.policy.single.cadrl import mlp
from crowd_nav.policy.single.multi_human_rl import MultiHumanRL
from torch_geometric.nn import GraphConv,GCNConv
from torch_geometric.data import Data
"""
本策略主要构建的是所有agent双向连接的同构图
"""

class Graphgnn(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super(Graphgnn, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim,X_dim,wr_dims,wh_dims,final_state_dim,gcn2_w1_dim,planning_dims,num_layer,device):
        super().__init__()
        torch.manual_seed(1234)
        self.input_dim = input_dim
        self.self_state_dim = self_state_dim
        human_state_dim = self.input_dim-self.self_state_dim
        self.human_state_dim = human_state_dim
        self.X_dim = X_dim
        self.num_layer=num_layer
        self.device=device
        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)
        gcn2_w1_dim=52
        self.conv=Graphgnn(X_dim,gcn2_w1_dim,final_state_dim)
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
        human_states = state[:, :, self.self_state_dim:]

        # compute
        self_state_embedings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([self_state_embedings.unsqueeze(1), human_state_embedings], dim=1)
        # compute edge_index
        agent_number = X.size()[1]
        data=Data()
        # 初始化节点特征
        data.x=X[0,:,:]
        data.edge_index=self.edge_index(agent_number)
        h=self.conv(X,data.edge_index)
        h_final=h[:,0,:]
        # joint_state = torch.cat([self_state, h_final],dim=1)
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
        self.device=device
        self.set_common_parameters(config)
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, X_dim, wr_dims, wh_dims,
                                  final_state_dim, gcn2_w1_dim, planning_dims,num_layer,self.device)
        logging.info('self.model:{}'.format(self.model))
        logging.info('GCN layers: {}'.format(num_layer))
        logging.info('Policy: {}'.format(self.name))

    def get_matrix_A(self):
        return self.model.A


