from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSim-v1',
    entry_point='crowd_sim.envs:CrowdSimst2',
)
register(
    id='CrowdSim-v2',
    entry_point='crowd_sim.envs:CrowdSim_Hetero',
)