from crowd_sim.envs.policy.single.linear import Linear
from crowd_sim.envs.policy.single.orca import ORCA, CentralizedORCA
from crowd_sim.envs.policy.single.socialforce import SocialForce, CentralizedSocialForce


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['socialforce'] = SocialForce
policy_factory['centralized_orca'] = CentralizedORCA
policy_factory['centralized_socialforce'] = CentralizedSocialForce
policy_factory['none'] = none_policy
