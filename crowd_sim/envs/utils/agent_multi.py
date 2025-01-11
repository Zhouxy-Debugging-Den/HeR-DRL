import abc
import logging
import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.policy.multi.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state_multi import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = getattr(config, section).visible
        self.v_pref = getattr(config, section).v_pref
        self.radius = getattr(config, section).radius
        self.policy = policy_factory[getattr(config, section).policy]()
        self.sensor = getattr(config, section).sensor
        self.multi=getattr(config, section).multi
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None
    # rint self information (visibility, kinematic characteristics)
    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    # Set the policy, the policy kinematics corresponds to the agent kinematics
    def set_policy(self, policy):

        if self.time_step is None:
            raise ValueError('Time step is None')
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = policy.kinematics

    # Randomly sample the agent's radius and v_pref properties for a certain distribution
    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    # Setting information
    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    # Get observable status
    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius,self.multi)

    # Get the next observable state
    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_vx = action.v * np.cos(self.theta)
            next_vy = action.v * np.sin(self.theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius,self.multi)

    # Get all status
    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius,self.multi,
                         self.gx, self.gy, self.v_pref, self.theta)

    # Get location
    def get_position(self):
        return self.px, self.py

    # Setting the location
    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    # Get the target location
    def get_goal_position(self):
        return self.gx, self.gy

    # Get the starting position
    def get_start_position(self):
        return self.sx, self.sy

    # Get speed
    def get_velocity(self):
        return self.vx, self.vy

    # Setting the speed
    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    # The agent receives the observation and returns the action
    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    # Check legality
    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    # Calculate the next time step position
    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if delta_t is None:
            delta_t=0.25
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    # Execute actions and update status
    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    # Arrival at destination
    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

