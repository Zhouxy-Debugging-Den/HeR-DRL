from crowd_nav.configs.icra_benchmark.config_multi import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'HeR_DRL'

        # gcn
        self.gcn.num_layer = 2
        self.gcn.X_dim = 32
        self.gcn.similarity_function = 'embedded_gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = True
        self.gcn.human_num=5
        self.gcn.other_robot_num=2


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
