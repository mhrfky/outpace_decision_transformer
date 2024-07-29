import numpy as np
import random

class TrajectoryBuffer:
    def __init__(self, max_good_parts):
        self.trajectories = {}
        self.good_parts = []
        self.acts = {}
        self.max_good_parts = max_good_parts
        self.current_index = 0  # To keep track of the order of trajectories

    def add_trajectory(self, trajectory, actions, good_parts):
        """
        Add a trajectory and its good parts to the buffer. Remove old entries if max size is reached.
        
        Parameters:
            trajectory (np.array): The full trajectory.
            good_parts (list of tuples): Each tuple contains the start and end index of a good part.
        """
        self.trajectories[self.current_index] = trajectory
        self.acts[self.current_index] = actions
        for start, end in good_parts:
            self.good_parts.append((self.current_index, start, end))
        
        # Remove old good parts if the buffer exceeds max size
        while len(self.good_parts) > self.max_good_parts:
            self._remove_oldest_good_part()
        
        self.current_index += 1

    def _remove_oldest_good_part(self):
        """
        Remove the oldest good part from the buffer and adjust the dictionary.
        """
        if self.good_parts:
            oldest_trajectory_index, _, _ = self.good_parts.pop(0)
            
            # Remove the trajectory if no more good parts are associated with it
            if self.good_parts[0][0] != oldest_trajectory_index:
                del self.trajectories[oldest_trajectory_index]

    def sample(self, n_samples):
        """
        Sample good parts randomly from the buffer.
        
        Parameters:
            n_samples (int): Number of good parts to sample.
        
        Returns:
            list of np.array: Sampled good parts.
        """
        if n_samples > len(self.good_parts):
            n_samples = len(self.good_parts)
        
        sampled_parts = random.sample(self.good_parts, n_samples)
        return sampled_parts
    def get_trajectory_by_index(self, index):
        """
        Get a trajectory by its index.
        
        Parameters:
            index (int): Index of the trajectory.
        
        Returns:
            np.array: The trajectory.
        """
        return self.trajectories[index]
    def get_acts_by_index(self, index):
        """
        Get a trajectory by its index.
        
        Parameters:
            index (int): Index of the actions.
        Returns:
            np.array: The actions.
        """
        return self.acts[index]