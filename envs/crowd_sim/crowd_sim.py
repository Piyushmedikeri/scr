import gym
import numpy as np
from numpy.linalg import norm
from envs.crowd_sim.utils.utils import point_to_segment_dist
from envs.crowd_sim.utils.info import *


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

		#print('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

		#self.renderer = MatplotRenderer()


	def step(self):
		print('step initialization')

	def generate_static_human(self, i):
		if i == self.human_num - 1:
		    self.humans[i].set(0, 0, 0, 0, 0, 0, 0, radius=.3, v_pref=0.)
		else:
		    self.generate_circle_crossing_human(i)

	def generate_random_human_position(self, human_num, rule):
		"""
		Generate human position according to certain rule
		Rule square_crossing: generate start/goal position at two sides of y-axis
		Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

		:param human_num:
		:param rule:
		:return:
		"""
		# initial min separation distance to avoid danger penalty at beginning
		if rule == 'square_crossing':
		    for i in range(human_num):
		        self.generate_square_crossing_human(i)
		elif rule == 'circle_crossing':
		    for i in range(human_num):
		        self.generate_circle_crossing_human(i)
		elif rule == 'static':
		    for i in range(human_num):
		        self.generate_static_human(i)
		elif rule == 'mixed':
		    static = True if np.random.random() < 0.2 else False
		    if static:
		        # randomly initialize static objects in a square of (width, height)
		        width = 4
		        height = 8
		        for i in range(self.human_num):
		            if np.random.random() > 0.5:
		                sign = -1
		            else:
		                sign = 1
		            while True:
		                px = np.random.random() * width * 0.5 * sign
		                py = (np.random.random() - 0.5) * height
		                collide = False
		                for agent in [self.robot] + self.humans[:i]:
		                    if norm((px - agent.px, py - agent.py)) < self.humans[i].radius + agent.radius + self.discomfort_dist:
		                        collide = True
		                        break
		                if not collide:
		                    break
		            self.humans[i].set(px, py, px, py, 0, 0, 0)
		    else:
		        # the first 2 two humans will be in the circle crossing scenarios
		        # the rest humans will have a random starting and end position
		        for i in range(self.human_num):
		            if i < 2:
		                self.generate_circle_crossing_human(i)
		            else:
		                self.generate_square_crossing_human(i)
		else:
		    raise ValueError("Rule doesn't exist")

	def generate_circle_crossing_human(self, i):
		if self.randomize_attributes:
		    self.humans[i].sample_random_attributes()
		while True:
		    angle = np.random.random() * np.pi * 2
		    # add some noise to simulate all the possible cases robot could meet with human
		    px_noise = (np.random.random() - 0.5) * self.humans[i].v_pref
		    py_noise = (np.random.random() - 0.5) * self.humans[i].v_pref
		    px = self.circle_radius * np.cos(angle) + px_noise
		    py = self.circle_radius * np.sin(angle) + py_noise
		    collide = False
		    for agent in [self.robot] + self.humans[:i]:
		        min_dist = self.humans[i].radius + agent.radius + self.discomfort_dist
		        if norm((px - agent.px, py - agent.py)) < min_dist or \
		                norm((px - agent.gx, py - agent.gy)) < min_dist:
		            collide = True
		            break
		    if not collide:
		        break
		self.humans[i].set(px, py, -px, -py, 0, 0, 0)

	def generate_square_crossing_human(self, i):
		if self.randomize_attributes:
		    self.humans[i].sample_random_attributes()
		if np.random.random() > 0.5:
		    sign = -1
		else:
		    sign = 1
		while True:
		    px = np.random.random() * self.square_width * 0.5 * sign
		    py = (np.random.random() - 0.5) * self.square_width
		    collide = False
		    for agent in [self.robot] + self.humans[:i]:
		        if norm((px - agent.px, py - agent.py)) < self.humans[i].radius + agent.radius + self.discomfort_dist:
		            collide = True
		            break
		    if not collide:
		        break
		while True:
		    gx = np.random.random() * self.square_width * 0.5 * -sign
		    gy = (np.random.random() - 0.5) * self.square_width
		    collide = False
		    for agent in [self.robot] + self.humans[:i]:
		        if norm((gx - agent.gx, gy - agent.gy)) < self.humans[i].radius + agent.radius + self.discomfort_dist:
		            collide = True
		            break
		    if not collide:
		        break
		self.humans[i].set(px, py, gx, gy, 0, 0, 0)

	def reset(self, phase='test', test_case=None):
		"""
		Set px, py, gx, gy, vx, vy, theta for robot and humans
		:return:
		"""
		if self.robot is None:
		    raise AttributeError('robot has to be set!')
		assert phase in ['train', 'val', 'test']
		if test_case is not None:
		    self.case_counter[phase] = test_case
		self.global_time = 0
		if phase == 'test':
		    self.human_times = [0] * self.human_num
		else:
		    self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
		if not self.robot.policy.multiagent_training:
		    self.train_val_sim = 'circle_crossing'


		counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
		                  'val': 0, 'test': self.case_capacity['val']}
		self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
		if self.case_counter[phase] >= 0:
		    np.random.seed(counter_offset[phase] + self.case_counter[phase])
		    if phase in ['train', 'val']:
		        human_num = self.human_num if self.robot.policy.multiagent_training else 1
		        self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
		    else:
		        self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
		    # case_counter is always between 0 and case_size[phase]
		    self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
		else:
		    assert phase == 'test'
		    if self.case_counter[phase] == -1:
		        # for debugging purposes
		        self.human_num = 3
		        #self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
		        self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
		        self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
		        self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
		    else:
		        raise NotImplementedError

		for agent in [self.robot] + self.humans:
		    agent.time_step = self.time_step
		    agent.policy.time_step = self.time_step

		self.state = list()
		if hasattr(self.robot.policy, 'action_values'):
		    self.action_values = list()
		if hasattr(self.robot.policy, 'get_attention_weights'):
		    self.attention_weights = list()

		# get current observation
		if self.robot.sensor == 'coordinates':
		    ob = [human.get_observable_state() for human in self.humans]
		elif self.robot.sensor == 'RGB':
		    raise NotImplementedError

		return ob

	def onestep_lookahead(self, action):
		return self.step(action, update=False)

	def step(self, action, update=True):
	    """
	    Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

	    """
	    human_actions = []
	    for human in self.humans:
	        # observation for humans is always coordinates
	        ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
	        if self.robot.visible:
	            ob += [self.robot.get_observable_state()]
	        human_actions.append(human.act(ob))

	    # collision detection
	    dmin = float('inf')
	    collision = False
	    for i, human in enumerate(self.humans):
	        px = human.px - self.robot.px
	        py = human.py - self.robot.py
	        if self.robot.kinematics == 'holonomic':
	            vx = human.vx - action.vx
	            vy = human.vy - action.vy
	        elif self.robot.kinematics == 'unicycle' and action.r != 0:
	            vx = (action.v / action.r) * (
	                    np.sin(action.r * self.time_step + self.robot.theta) - np.sin(self.robot.theta))
	            vy = (action.v / action.r) * (
	                    np.cos(action.r * self.time_step + self.robot.theta) - np.cos(self.robot.theta))
	        else:
	            vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
	            vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
	        ex = px + vx * self.time_step
	        ey = py + vy * self.time_step
	        # closest distance between boundaries of two agents
	        closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
	        if closest_dist < 0:
	            collision = True
	            # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
	            break
	        elif closest_dist < dmin:
	            dmin = closest_dist

	    # collision detection between humans
	    human_num = len(self.humans)
	    for i in range(human_num):
	        for j in range(i + 1, human_num):
	            dx = self.humans[i].px - self.humans[j].px
	            dy = self.humans[i].py - self.humans[j].py
	            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
	            if dist < 0:
	                # detect collision but don't take humans' collision into account
	                logging.debug('Collision happens between humans in step()')

	    # check if reaching the goal
	    end_position = np.array(self.robot.compute_position(action, self.time_step))
	    reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

	    if self.global_time >= self.time_limit - 1:
	        reward = 0
	        done = True
	        info = Timeout()
	    elif collision:
	        reward = self.collision_penalty
	        done = True
	        info = Collision()
	    elif reaching_goal:
	        reward = self.success_reward
	        done = True
	        info = ReachGoal()
	    elif dmin < self.discomfort_dist:
	        # only penalize agent for getting too close if it's visible
	        # adjust the reward based on FPS
	        reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step * (1 - (self.robot.vx**2 + self.robot.vy**2)**.5)
	        done = False
	        info = Danger(dmin)
	    else:
	        reward = 0
	        done = False
	        info = Nothing()

	    if update:
	        # store state, action value and attention weights
	        self.state = [self.robot.get_full_state(), [human.get_full_state() for human in self.humans]]
	        if hasattr(self.robot.policy, 'action_values'):
	            self.action_values.append(self.robot.policy.action_values)
	        if hasattr(self.robot.policy, 'get_attention_weights'):
	            self.attention_weights.append(self.robot.policy.get_attention_weights())

	        # update all agents
	        self.robot.step(action)
	        for i, human_action in enumerate(human_actions):
	            self.humans[i].step(human_action)
	        self.global_time += self.time_step
	        for i, human in enumerate(self.humans):
	            # only record the first time the human reaches the goal
	            if self.human_times[i] == 0 and human.reached_destination():
	                self.human_times[i] = self.global_time

	        # compute the observation
	        ob = [human.get_observable_state() for human in self.humans]

	    else:
	        if self.robot.sensor == 'coordinates':
	            ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
	        elif self.robot.sensor == 'RGB':
	            raise NotImplementedError

	    return ob, reward, done, info

	def set_robot(self, robot):
		self.robot = robot

	def set_humans(self, humans):
		self.humans = humans