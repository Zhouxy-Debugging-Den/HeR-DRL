from crowd_sim.envs.policy.single.policy_factory import policy_factory
from crowd_nav.policy.single.cadrl import CADRL
from crowd_nav.policy.single.lstm_rl_origin import LstmRL
from crowd_nav.policy.single.sarl import SARL
from crowd_nav.policy.single.graphgnn import graphgnn
from crowd_nav.policy.single.graphtohetero import graphtohetero


policy_factory['cadrl'] = CADRL
policy_factory['lstm-rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['HoR-DRL']=graphgnn
policy_factory['HeR-DRL']=graphtohetero

