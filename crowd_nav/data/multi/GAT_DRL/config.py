from crowd_nav.configs.icra_benchmark.config_multi import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'sg_dq3n'
        self.action_space.query_env = True
        self.sarl.with_om=True


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
