from envs.crowd_sim.utils.agent import Agent
from envs.crowd_sim.utils.state import JointState


class Robot(Agent):
    def __init__(self):
        super(Robot,self).__init__()

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
