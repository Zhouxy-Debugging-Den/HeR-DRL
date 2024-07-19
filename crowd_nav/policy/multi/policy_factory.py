from crowd_sim.envs.policy.multi.policy_factory import policy_factory
from crowd_nav.policy.multi.lstm_rl import lstm_rl
from crowd_nav.policy.multi.sarl import sarl
from crowd_nav.policy.multi.graphgnn import graphgnn
from crowd_nav.policy.multi.graphtohetero import graphtohetero

policy_factory['lstm-rl'] = lstm_rl
policy_factory['sarl'] = sarl
policy_factory['HoR-DRL']=graphgnn
policy_factory['HeR-DRL']=graphtohetero
