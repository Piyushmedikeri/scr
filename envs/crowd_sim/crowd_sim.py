import gym

class CrowdSim(gym.Env):
	def __init__(self):
		print('Env initialization')

	def step(self):
		print('step initialization')

	def reset(self):
		print('Env reset')