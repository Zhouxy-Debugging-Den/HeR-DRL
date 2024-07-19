from crowd_sim.envs.utils.agent_multi import Agent
from crowd_sim.envs.utils.state_multi import JointState


class Human(Agent):
    # 多了id信息
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None
        self.reach_count = 0
        self.start_pos = []

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action,_ = self.policy.predict(state)
        return action
