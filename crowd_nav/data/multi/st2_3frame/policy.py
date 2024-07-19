import copy
import torch
import torch.nn as nn
import logging
from crowd_nav.policy.multi.ST.multi_human_rl import MultiHumanRL


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Mlp(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class Embedding(nn.Module):
    def __init__(self, joint_state=13, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(joint_state, embed_dim)
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 dropout=0.1,
                 attention_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_head_size = embed_dim // num_heads
        self.all_head_size = self.attn_head_size * num_heads

        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size * 3)
        nn.init.xavier_normal_(self.qkv.weight)
        nn.init.constant_(self.qkv.bias, 0.0)
        self.scales = self.attn_head_size ** -0.5

        self.out = nn.Linear(self.all_head_size,
                             embed_dim)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def transpose_multihead(self, x):
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)
        x = x.permute([0, 2, 1, 3])
        return x

    def forward(self, x):
        size = x.shape

        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multihead, qkv)            # [300, 8, 5, 16]
        k = torch.mean(q, 2, keepdim=True)                      # [300, 8, 1, 16]
        k = k.expand((size[0], 8, size[1], -1))                 # [300, 8, 5, 16]
        k = k.permute([0, 1, 3, 2])                             # [300, 8, 16, 5]
        attn = torch.matmul(q, k)                               # [300, 8, 5, 5]

        # print(attn.shape)
        # attn = q + k
        attn = attn * self.scales
        attn = self.softmax(attn)                               # [300, 8, 5, 5]
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)
        z = z.permute([0, 2, 1, 3])

        new_shape = list(z.shape[:-2]) + [self.all_head_size]
        z = z.reshape(new_shape)
        z = self.out(z)
        z = self.proj_dropout(z)

        return z

class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 mlp_ratio=4.,
                 dropout=0.1,
                 attention_dropout=0.1,
                 drop_path_ratio=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.attn = Attention(embed_dim,
                              num_heads,
                              dropout,
                              attention_dropout)

        self.mlp_norm = nn.LayerNorm(embed_dim)

        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x):

        # x = x + self.drop_path(self.attn(self.attn_norm(x)))
        # x = x + self.drop_path(self.mlp(self.mlp_norm(x)))

        h = x
        x = self.attn_norm(x)
        x = self.attn(x)

        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x

class Encoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                         num_heads,
                                         mlp_ratio=mlp_ratio,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout,)
            layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(layer_list)
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.encoder_norm(x)
        return out

class Spatial_Temporal_Transformer(nn.Module):
    def __init__(self,
                 joint_state=14,
                 embed_dim=128,
                 temporal_depth=1,
                 spatial_depth=1,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 attention_dropout=0.1):
        super().__init__()
        # create patch embedding with positional embedding
        self.Embedding = Embedding(joint_state,
                                   embed_dim,
                                   dropout)
        # create multi head self-attention layers
        self.temporal_encoder = Encoder(embed_dim,
                                        num_heads,
                                        temporal_depth,
                                        mlp_ratio,
                                        dropout,
                                        attention_dropout)
        self.Spatial_encoder = Encoder(embed_dim,
                                       num_heads,
                                       spatial_depth,
                                       mlp_ratio,
                                       dropout,
                                       attention_dropout)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.value_Linear_1 = nn.Linear(134, 256)
        self.act = nn.GELU()
        self.value_Linear_2 = nn.Linear(256, 128)
        self.value_Linear_3 = nn.Linear(128, 1)
        self.softmax = nn.Softmax(-1)
        self.attention_weights = None

    def forward(self, x):
        # x = x.reshape(-1, 3, 5, 13)
        b, t, h, w = x.shape
        robot_state = x[:, 2:3, 0, :6].reshape(b, 6) 
        # spatial_Transformer
        x = x.reshape(-1, h, w)
        spatial_state = self.Embedding(x)
        spatial_state = self.Spatial_encoder(spatial_state).reshape(b, t, h, -1)
        # temporal_Transformer
        temporal_state = spatial_state.permute([0, 2, 1, 3]).reshape(b*h, t, -1)
        temporal_state = self.temporal_encoder(temporal_state)
        temporal_state = temporal_state.reshape(b, h, t, -1).permute([0, 2, 1, 3])
        state = temporal_state.reshape(b, t*h, -1)

        state = state.permute([0, 2, 1])
        state = self.avgpool(state).flatten(1)
        state = torch.cat((state, robot_state), 1)
        state = self.value_Linear_1(state)
        state = self.act(state)
        state = self.value_Linear_2(state)
        state = self.act(state)
        state = self.value_Linear_3(state)
        return state

class st2(MultiHumanRL):
    def __init__(self):
        super(st2, self).__init__()
        self.name = 'st2'
        self.with_costmap = False
        self.gc = None
        self.gc_resolution = None
        self.gc_width = None
        self.gc_ox = None
        self.gc_oy = None

    def configure(self, config,device):
        self.multiagent_training = config.gcn.multiagent_training
        self.device=device
        self.set_common_parameters(config)
        self.model = Spatial_Temporal_Transformer()
        logging.info('self.model:{}'.format(self.model))
        logging.info('Policy: {}'.format(self.name))
    # def compute_cost(self, state):
    #     costs = []
    #     x = state.px
    #     y = state.py
    #     min_x = x - 0.4
    #     min_y = y - 0.4
    #     max_x = x + 0.4
    #     max_y = y + 0.4   #标定机器人的范围
    #     grid_min_x = int(round((min_x - self.gc_ox) / self.gc_resolution))
    #     grid_min_y = int(round((min_y - self.gc_oy) / self.gc_resolution))
    #     grid_max_x = int(round((max_x - self.gc_ox) / self.gc_resolution))
    #     grid_max_y = int(round((max_y - self.gc_oy) / self.gc_resolution))
    #     for i in range(grid_min_x, grid_max_x+1):
    #         for j in range(grid_min_y, grid_max_y + 1):
    #             index = i + self.gc_width * j
    #             costs.append(self.gc[index])
    #     max_cost = max(costs)
    #     return max_cost

    # def predict(self, state):
    #     """
    #     Takes pairwise joint state as input to value network and output action.
    #     The input to the value network is always of shape (batch_size, # humans, rotated joint state length).
    #     If with_costmap is True, the dangerous actions predicted by the value network will be screened out to avoid static obstacles on the map.
    #     """
    #     if self.phase is None or self.device is None:
    #         raise AttributeError('Phase, device attributes have to be set!')
    #     if self.phase == 'train' and self.epsilon is None:
    #         raise AttributeError('Epsilon attribute has to be set in training phase')
    #     if self.reach_destination(state):
    #         return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
    #     if self.action_space is None:
    #         self.build_action_space(state.self_state.v_pref)
    #
    #     occupancy_maps = None
    #     probability = np.random.random()
    #     if self.phase == 'train' and probability < self.epsilon:
    #         max_action = self.action_space[np.random.choice(len(self.action_space))]
    #     else:
    #         self.action_values = list()
    #         max_value = float('-inf')
    #         max_action = None
    #         for action in self.action_space:
    #             next_self_state = self.propagate(state.self_state, action)
    #             next_self_state_further = self.propagate_more(state.self_state, action)
    #
    #             # abort actions which will probably cause collision with static obstacles in the costmap
    #             if self.with_costmap is True:
    #                 cost = self.compute_cost(next_self_state_further)
    #                 if cost > 0:
    #                     print("********** Abort action:", action, " with cost:", cost, " that will hit the obstacles.")
    #                     continue
    #
    #             if self.query_env:  #false
    #                 next_human_states, reward, done, info = self.env.onestep_lookahead(action)
    #             else:
    #                 next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
    #                                    for human_state in state.human_states]
    #                 reward = self.compute_reward(next_self_state, next_human_states)
    #             batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
    #                                                for next_human_state in next_human_states], dim=0)
    #             rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
    #
    #             if self.with_om:  #false
    #                 if occupancy_maps is None:
    #                     occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
    #                 rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)  #[100, 5, 13]
    #             # VALUE UPDATE
    #             #
    #             # rotated_batch_input = rotated_batch_input.unsqueeze(1)  # [1, 1, 5, 13]
    #             # rotated_batch_input = torch.cat((rotated_batch_input, rotated_batch_input, rotated_batch_input), 1)  # [1, 3, 5, 13]
    #             next_state_value = self.model(rotated_batch_input).data.item()
    #             value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
    #             # value = r + gamma ** (0.25 * v_pref) * next_state_value
    #             self.action_values.append(value)
    #             if value > max_value:
    #                 max_value = value
    #                 max_action = action
    #                 # print("********** choose action:", action)
    #                 # print("********** cost:", cost)
    #
    #         if max_action is None:
    #             # if the robot is trapped, choose the turning action to escape
    #             max_action = ActionRot(0, 0.78)
    #             print("The robot is trapped. Rotate in place to escape......")
    #
    #     if self.phase == 'train' or self.phase == 'test' or self.phase == 'val':
    #         self.last_state = self.transform(state)
    #
    #     return max_action
    #
    # def compute_reward(self, nav, humans):
    #     # collision detection
    #     dmin = float('inf')
    #     collision = False
    #     if len(humans):
    #         for i, human in enumerate(humans):
    #             dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
    #             if dist < 0:
    #                 collision = True
    #                 break
    #             if dist < dmin:
    #                 dmin = dist
    #     # check if reaching the goal
    #     reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
    #     if collision:
    #         reward = self.env.collision_penalty
    #     elif reaching_goal:
    #         reward = 1
    #     elif dmin < self.env.discomfort_dist:
    #         reward = (dmin - self.env.discomfort_dist) * self.env.discomfort_penalty_factor * self.env.time_step
    #     else:
    #         reward = 0
    #
    #     return reward













