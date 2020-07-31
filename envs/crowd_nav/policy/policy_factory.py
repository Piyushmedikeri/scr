from envs.crowd_sim.policy.policy_factory import policy_factory
from envs.crowd_nav.policy.cadrl import CADRL
from envs.crowd_nav.policy.lstm_rl import LstmRL
from envs.crowd_nav.policy.sarl import SARL
from envs.crowd_nav.policy.scr import SCR
from envs.crowd_nav.policy.interactive import Interactive

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['scr'] = SCR
policy_factory['interactive'] = Interactive
