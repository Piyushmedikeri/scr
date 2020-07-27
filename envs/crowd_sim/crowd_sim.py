import gym
import numpy as np

class CrowdSim(gym.Env):
	def __init__(self):
		print('Env initialization')
		"""
		Environment setup
		"""
		self.time_limit = 25
		self.time_step = 0.25
		self.robot = None
		self.humans = None
		self.global_time = None
		self.human_times = None


		# reward function
		self.success_reward = 1
		self.collision_penalty = 0.25
		self.discomfort_dist = 0.2
		self.discomfort_penalty_factor = 0.5


		# simulation configuration
		self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
		self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 100, 'test': 500}
		self.randomize_attributes = True
		self.train_val_sim = 'circle_crossing'
		self.test_sim = 'circle_crossing'
		self.square_width = 10
		self.circle_radius = 4
		self.human_num = 5


		# for visualization
		self.state = None
		self.action_values = None
		self.attention_weights = None
		self.case_counter = {'train': 0, 'test': 0, 'val': 0}

		print('human number: {}'.format(self.human_num))
		if self.randomize_attributes:
		    print("Randomize human's radius and preferred speed")
		else:
		    print("Not randomize human's radius and preferred speed")

		print('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

		#self.renderer = MatplotRenderer()


	def step(self):
		print('step initialization')

	def reset(self):
		print('Env reset')