import numpy as np
from hgg.utils import generate_random_samples
from hgg.utils import euclid_distance
class StatesBuffer:
	def __init__(self, max_size=5000):
		self.__list = np.array([[0, 0]])
		self.max_size = max_size

	def insert_trajectory(self, trajectory):
		self.__list = np.vstack((trajectory, self.__list))[:self.max_size]

	def insert_state(self, state):
		self.__list = np.vstack((self.__list , np.array([state])))

	def fill_list_with_states_near_a_point(self, init_state):
		temp_list = np.tile(np.array([init_state], dtype = np.float32), (self.max_size,1))
		temp_list += np.random.normal(loc=np.zeros_like(self.__list), scale=0.5*np.ones_like(self.__list))
		self.__list = np.vstack((self.__list , temp_list))

	def fill_list_with_random_around_maze(self, limits, sample_size ):
		temp_list = generate_random_samples([limits[0][0], limits[1][0]], [limits[0][1], limits[1][1]], (2,) , sample_size)
		self.__list = np.vstack((self.__list, temp_list))


	def sample(self, batch_size=64):
		sampled_elements = self.__list[np.random.choice(self.__list.shape[0], batch_size, replace=False)]
		return sampled_elements
	
	def remove_states_near_the_trajectory(self, trajectory, threshold):
		for state in trajectory:
			self.remove_states_near_the_state(state, threshold)
	def remove_states_near_other_sample(self, sample, threshold):
		for state in sample:
			self.remove_states_near_the_state(state, threshold)
	def remove_states_near_the_state(self, state, threshold):
		delete_indices = []
		for i, s in enumerate(self.__list): #TODO rename
			if euclid_distance(s,state) < threshold:
				delete_indices.append(i)
		self.__list = np.delete(self.__list, delete_indices, axis = 0)

	def __len__(self):
		return len(self.__list)
		