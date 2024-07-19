from crowd_sim.envs.utils.agent_multi import Agent
from crowd_sim.envs.utils.state_multi import JointState
from crowd_sim.envs import policy

class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.rotation_constraint = getattr(config, section).rotation_constraint

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        states = []
        for i in range(len(ob)):
            states.append(JointState(self.get_full_state(), ob[i]))
        # state = JointState(self.get_full_state(), ob)
        action, action_index = self.policy.predict(states)
        return action, action_index

    def get_state(self, ob):
        state = JointState(self.get_full_state(), ob)
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        return self.policy.transform(state)
