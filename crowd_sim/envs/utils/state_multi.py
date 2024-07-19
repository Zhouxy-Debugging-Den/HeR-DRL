import torch


class FullState(object):
    def __init__(self, px, py, vx, vy, radius,multi, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.catogory = multi   # 0是机器人，1是行人
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)


    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius,self.catogory,self.gx,
                        self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.catogory , self.gx,
                                          self.gy,self.v_pref, self.theta]])

    def to_tuple(self):
        return self.px, self.py, self.vx, self.vy, self.radius,self.catogory, self.gx, self.gy, self.v_pref,self.theta

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius,self.catogory)


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius,multi):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.multi = multi

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius,self.multi)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius,self.multi]])

    def to_tuple(self):
        return self.px, self.py, self.vx, self.vy, self.radius,self.multi


class JointState(object):
    def __init__(self, robot_state, agent_states):
        assert isinstance(robot_state, FullState)
        for agent_state in agent_states:
            assert isinstance(agent_state, ObservableState)

        self.robot_state = robot_state
        self.agent_states = agent_states

    def to_tensor(self, add_batch_size=False, device=None):
        robot_state_tensor = torch.Tensor([self.robot_state.to_tuple()])
        agent_states_tensor = torch.Tensor([agent_state.to_tuple() for agent_state in self.agent_states])

        if add_batch_size:
            robot_state_tensor = robot_state_tensor.unsqueeze(0)
            agent_states_tensor = agent_states_tensor.unsqueeze(0)

        if device == torch.device('cuda:0'):
            robot_state_tensor = robot_state_tensor.cuda()
            agent_states_tensor = agent_states_tensor.cuda()
        elif device is not None:
            robot_state_tensor.to(device)
            agent_states_tensor.to(device)

        if agent_states_tensor.shape[1]==0:
            agent_states_tensor = None
        return robot_state_tensor, agent_states_tensor


def tensor_to_joint_state(state):
    robot_state, agent_states = state

    robot_state = robot_state.cpu().squeeze().data.numpy()
    robot_state = FullState(robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4],
                            robot_state[5], robot_state[6], robot_state[7], robot_state[8],robot_state[9])
    if agent_states is None:
        agent_states = []
    else:
        agent_states = agent_states.cpu().squeeze(0).data.numpy()
        agent_states = [ObservableState(agent_state[0], agent_state[1], agent_state[2], agent_state[3],
                                        agent_state[4],agent_state[5]) for agent_state in agent_states]

    return JointState(robot_state, agent_states)
