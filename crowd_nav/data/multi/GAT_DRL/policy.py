import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
# from crowd_nav.policy.helpers import mlp, GAT #, GraphAttentionLayer
from crowd_nav.policy.multi.multi_human_rl import MultiHumanRL


class GraphAttentionLayerSim(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayerSim, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.similarity_function = 'embedded_gaussian'
        self.W_a = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_a.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        # map input to h
        e = self.leakyrelu(self.compute_similarity_matrix(input))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = nn.functional.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, input)
        h_prime = h_prime + self.bias
        return nn.functional.elu(h_prime)

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.W_a), X.permute(0, 2, 1))
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
        elif self.similarity_function == 'cosine':
            X = torch.matmul(X, self.W_a)
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError
        return A
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # 使用Xavier初始化权重
        nn.init.constant_(m.bias, 0)        # 将偏置初始化为常数0
def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        # nn.init.orthogonal(layers[-1].weight)
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
            # layers.append(nn.LeakyReLU(negative_slope=0.1))
    net = nn.Sequential(*layers)
    net.apply(init_weights)
    return net

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.w_a = mlp(2 * self.in_features, [2 * self.in_features, 1], last_relu=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.04)

    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        A = self.compute_similarity_matrix(input)
        e = self.leakyrelu(A)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        next_H = torch.matmul(attention, input)
        return next_H, attention[0, 0, :].data.cpu().numpy()

    def compute_similarity_matrix(self, X):
        indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
        selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
        pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
        A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
        return A


class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [GraphAttentionLayerSim(in_feats, hid_feats, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = mlp(hid_feats * nheads, [out_feats], last_relu=True)
        self.add_module('out_gat', self.out_att)

    def forward(self, x, adj):
        assert len(x.shape) == 3
        assert len(adj.shape) == 3
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = self.out_att(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection
        gcn2_w1_dim=config.gcn.gcn2_w1_dim
        planning_dims=config.gcn.planning_dims
        # design choice
        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection
        self.gat0 = GraphAttentionLayer(self.X_dim, self.X_dim)
        self.gat1 = GraphAttentionLayer(self.X_dim, self.X_dim)


        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim+1, wh_dims, last_relu=True)
        # for visualization
        self.attention_weights = None
        self.value_net = mlp(gcn2_w1_dim, planning_dims, last_relu=True)

    def compute_adjectory_matrix(self, robot_state,human_state):
        robot_num = robot_state.size()[1]
        human_num = human_state.size()[1]
        Num = robot_num + human_num
        adj = torch.ones((Num, Num))
        for i in range(robot_num, robot_num+human_num):
            adj[i][0] = 0
        adj = adj.repeat(robot_state.size()[0], 1, 1)
        return adj

    def forward(self, state,dropout):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        if isinstance(state, tuple):
            state, lengths = state
        else:
            state = state
            # lengths = torch.IntTensor([state.size()[1]])

        robot_state = state[:, 0, :self.robot_state_dim]
        human_states = state[:, :, self.robot_state_dim:]

        if human_states is None:
            robot_state_embedings = self.w_r(robot_state)
            adj = torch.ones((1, 1))
            adj = adj.repeat(robot_state.size()[0], 1, 1)
            X = robot_state_embedings
            if robot_state.shape[0]==1:
                H1, self.attention_weights = self.gat0(X, adj)
            else:
                H1, _ = self.gat0(X, adj)
            H2, _ = self.gat1(H1, adj)
            if self.skip_connection:
                output = H1 + H2 + X
            else:
                output = H2
            return output
        else:
            adj = self.compute_adjectory_matrix(robot_state.unsqueeze(1),human_states)
            # compute feature matrix X
            robot_state_embedings = self.w_r(robot_state).unsqueeze(1)
            human_state_embedings = self.w_h(human_states)
            X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)
            if robot_state.shape[0]==1:
                H1, self.attention_weights = self.gat0(X, adj)
            else:
                H1, _ = self.gat0(X, adj)
            H2, _ = self.gat1(H1, adj)
            if self.skip_connection:
                output = H1 + H2 + X
            else:
                output = H2
            # 这一步发生了梯度消失
            output_value=self.value_net(output[:, 0, :])
            return output_value




class sg_dq3n(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'sg_dq3n'

    def configure(self, config,device):
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        # X_dim = config.gcn.X_dim
        # wr_dims = config.gcn.wr_dims
        # wh_dims = config.gcn.wh_dims
        # final_state_dim = config.gcn.final_state_dim
        # gcn2_w1_dim = config.gcn.gcn2_w1_dim
        #
        # human_num=config.gcn.human_num
        # planning_dims = config.gcn.planning_dims
        # with_global_state = config.hetero.with_global_state
        self.device=device
        self.set_common_parameters(config)

        self.model = ValueNetwork(config, 6, 7)
        logging.info('self.model:{}'.format(self.model))
        logging.info('GCN layers: {}'.format(num_layer))
        logging.info('Policy: {}'.format(self.name))


