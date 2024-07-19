from crowd_sim.envs.policy.multi.linear import Linear
from crowd_sim.envs.policy.multi.orca import ORCA, CentralizedORCA
from crowd_sim.envs.policy.multi.socialforce import SocialForce, CentralizedSocialForce


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['socialforce'] = SocialForce
policy_factory['centralized_orca'] = CentralizedORCA
policy_factory['centralized_socialforce'] = CentralizedSocialForce
policy_factory['none'] = none_policy
