import logging
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from crowd_nav.policy.single.cadrl import mlp
from crowd_nav.policy.single.multi_human_rl import MultiHumanRL
from torch_geometric.nn import GraphConv,to_hetero


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv((-1, -1), out_channels)
        self.conv2 = GraphConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # 最后考虑是否用relu()
        x = self.conv2(x, edge_index).relu()
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim,X_dim,wr_dims,wagent_dims,final_state_dim,gcn2_w1_dim,planning_dims,num_layer,human_num,device):
        super().__init__()
        torch.manual_seed(1234)
        self.input_dim = input_dim
        self.self_state_dim = self_state_dim
        agent_state_dim = self.input_dim-self.self_state_dim
        self.agent_state_dim = agent_state_dim
        self.X_dim = X_dim
        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(agent_state_dim, wagent_dims, last_relu=True)
        self.num_layer=num_layer
        self.device=device
        self.human_num=human_num
        data = HeteroData()
        # 初始化节点特征
        data['robot'].x = torch.zeros((1, wr_dims[-1]))
        # data['robot'].num_nodes=1
        data['human'].x = torch.zeros((self.human_num, wr_dims[-1]))
        # data['human'].num_nodes = 5

        # 基础调参-----
        hidden_channels = 50
        output_channels = 32

        # ---------
        data['robot', 'sence', 'human'].edge_index = self.robot_to_human_edge_index(self.human_num)
        data['human', 'sence', 'robot'].edge_index = self.human_to_robot_edge_index(self.human_num)
        data['human', 'affect', 'human'].edge_index = self.human_to_human_edge_index(self.human_num)

        self.GNN=to_hetero(GNN(hidden_channels,gcn2_w1_dim),data.metadata(),aggr='sum')

        # self.value_net = mlp(gcn2_w1_dim+self_state_dim, planning_dims)
        self.value_net = mlp(gcn2_w1_dim, planning_dims)
    # 机器人到行人之间的edge_index
    def robot_to_human_edge_index(self,human_num):
        edge_index = torch.zeros([2,human_num],dtype=torch.long,device=self.device)
        for i in range(human_num):
            edge_index[0][i]=0
            edge_index[1][i] = i
        return edge_index

    def human_to_robot_edge_index(self,human_num):
        edge_index = torch.zeros([2,human_num],dtype=torch.long, device=self.device)
        for i in range(human_num):
            edge_index[0][i]=i
            edge_index[1][i] = 0
        return edge_index

    # 行人到行人的edge_index
    def human_to_human_edge_index(self,human_num):
        d1 = human_num ** 2 - human_num
        edge_index = torch.zeros([2,d1],dtype=torch.long,device=self.device)
        k=0
        for i in range(human_num):
            for j in range(human_num):
                if i != j:
                    edge_index[0][k]=i
                    edge_index[1][k] =j
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
        agent_num = X.size()[1]
        human_num=agent_num-1
        data=HeteroData()
        # 初始化节点特征
        data['robot'].x=self_state_embedings.unsqueeze(1)[0,:,:]
        # data['robot'].num_nodes=1
        data['human'].x=human_state_embedings[0,:,:]
        # data['human'].num_nodes = 5

        data['robot','sence','human'].edge_index=self.robot_to_human_edge_index(human_num)
        data['human', 'sence', 'robot'].edge_index = self.human_to_robot_edge_index(human_num)
        data['human','affect','human'].edge_index=self.human_to_human_edge_index(human_num)

        data_list=[]

        for i in range(self_state_embedings.size()[0]):
            data['robot'].x = self_state_embedings.unsqueeze(1)[i, :, :]
            data['human'].x = human_state_embedings[i, :, :]
            data_list.append(data.clone())
        batch_data=Batch.from_data_list(data_list)
        # graph convolution
        h=self.GNN(batch_data.x_dict,batch_data.edge_index_dict)

        # do planning using only the final layer feature of the agent
        h_final=h['robot']
        # value = self.value_net(h_final)
        # joint_state = torch.cat([self_state, h_final], dim=1)
        value = self.value_net(h_final)
        for i in range(self_state_embedings.size()[0]):
            del data_list[self_state_embedings.size()[0]-1-i]
        return value


class HeR_DRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'HeR-DRL'

    def configure(self, config,device):
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        gcn2_w1_dim = config.gcn.gcn2_w1_dim

        human_num=config.gcn.human_num
        planning_dims = config.gcn.planning_dims
        self.device=device
        self.set_common_parameters(config)
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, X_dim, wr_dims, wh_dims,
                                  final_state_dim, gcn2_w1_dim, planning_dims,num_layer,human_num,self.device)
        logging.info('self.model:{}'.format(self.model))
        logging.info('GCN layers: {}'.format(num_layer))
        logging.info('Policy: {}'.format(self.name))

    def get_matrix_A(self):
        return self.model.A


