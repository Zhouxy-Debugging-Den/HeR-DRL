from crowd_nav.configs.icra_benchmark.config_multi import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'lstm_rl'
        self.action_space.query_env = True
        self.lstm_rl.with_interaction_module = False


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
