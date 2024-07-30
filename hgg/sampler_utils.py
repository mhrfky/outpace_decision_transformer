from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import euclidean
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
def smooth_rtg(rtg, window_size):
    return uniform_filter1d(rtg, size=window_size)

def end_start_difference(segment):
    return segment[-1] - segment[0]

def expected_difference(segment_length, total_diff):
    return total_diff / segment_length

def adaptive_threshold(segment, base_threshold, proportion=0.1):
    segment_length = len(segment)
    total_diff = end_start_difference(segment)
    expected_diff = expected_difference(segment_length, total_diff)
    return base_threshold + proportion * expected_diff

def dynamic_variance_threshold(rtg, window_size, proportion=0.5):
    variances = rolling_window_variance(rtg, window_size)
    return np.mean(variances) * proportion

def rolling_window_variance(rtg, window_size):
    variances = [np.var(rtg[i:i + window_size]) for i in range(len(rtg) - window_size + 1)]
    return variances

def filter_by_end_start_difference(rtg, window_size, base_diff_threshold):
    segments = []
    for i in range(len(rtg) - window_size + 1):
        segment = rtg[i:i + window_size]
        total_diff = end_start_difference(segment)
        if total_diff < base_diff_threshold:
            segments.append((i, i + window_size - 1))
    return segments

def filter_by_variance(rtg, segments, var_threshold):
    filtered_segments = []
    for start, end in segments:
        segment = rtg[start:end + 1]
        if np.var(segment) < var_threshold:
            filtered_segments.append((start, end))
    return filtered_segments

def direction_changes(segment):
    gradient = np.diff(segment)
    direction_changes = np.where(np.diff(np.sign(gradient)) != 0, 1, 0)
    cumulative_changes = np.cumsum(direction_changes)
    return cumulative_changes[-1]

def filter_by_direction_changes(rtg, segments, max_changes):
    filtered_segments = []
    for start, end in segments:
        segment = rtg[start:end + 1]
        if direction_changes(segment) <= max_changes:
            filtered_segments.append((start, end))
    return filtered_segments

def state_similarity(segment, threshold=0.9):
    for i in range(len(segment)):
        for j in range(i+1, len(segment)):
            if cosine_similarity(segment[i], segment[j]) > threshold:
                return True
    return False

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def filter_by_circular_trajectories(states, segments, similarity_threshold=0.9):
    filtered_segments = []
    for start, end in segments:
        segment = states[start:end + 1]
        if not state_similarity(segment, similarity_threshold):
            filtered_segments.append((start, end))
    return filtered_segments
def direction_changes_states(states):
    directions = np.diff(states, axis=0)
    changes = np.sum(np.diff(np.sign(directions), axis=0) != 0, axis=0)
    return np.sum(changes)

def filter_by_direction_changes_states(states, segments, max_changes):
    return [(start, end) for start, end in segments if direction_changes_states(states[start:end + 1]) <= max_changes]

def select_segment_indices(rtg, states, window_size, base_diff_threshold, var_threshold_proportion=0.1, max_changes=2, step_ratio=0.5, similarity_threshold=0.9):
    # smoothed_rtg = smooth_rtg(rtg, window_size)
    
    # Filter by end-start difference
    segments = filter_by_end_start_difference(rtg, window_size, base_diff_threshold)
    
    # # Calculate dynamic variance threshold
    # var_threshold = dynamic_variance_threshold(rtg, window_size, var_threshold_proportion)
    
    # # Filter by variance
    # segments = filter_by_variance(smoothed_rtg, segments, var_threshold)
    
    # # Filter by direction changes
    # segments = filter_by_direction_changes(smoothed_rtg, segments, max_changes)
    
    # Filter by circular trajectories
    # segments = filter_by_circular_trajectories(states, segments, similarity_threshold)
    
    # Ensure minimal overlap
    final_segments = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        final_segments.append((start, end))
        i += int(window_size * step_ratio)
    
    return final_segments


def create_graph(states, max_distance):
    n = len(states)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(states[i] - states[j])
            if distance <= max_distance:
                G.add_edge(i, j, weight=distance ** 2)
    return G

def concat_original_with_new(states, new_trajectory, n, rtgs, new_rtgs):
    nth_from_end = new_trajectory[-n]
    index = np.where(np.all(states == nth_from_end, axis=1))[0][0]
    result_trajectory = np.concatenate((states[:index], new_trajectory[-n:]))
    
    # Adjust RTGs based on the new trajectory
    result_rtgs = np.concatenate((rtgs[:index], new_rtgs[-n:]))
    
    return result_trajectory, result_rtgs

def shortest_path_trajectory(states, rtgs, end_index, max_distance):
    G = create_graph(states, max_distance)
    shortest_path_indices = nx.dijkstra_path(G, source=0, target=end_index, weight='weight')
    shortest_path_states = states[shortest_path_indices]
    shortest_path_rtgs = rtgs[shortest_path_indices]
    return shortest_path_states, shortest_path_rtgs

def get_shortest_path_trajectories(states, rtgs, top_n, max_distance=1, n=10):
    top_indices = np.argsort(rtgs)[-top_n:]
    trajectories = []
    adjusted_rtgs = []
    for idx in top_indices:
        new_trajectory, new_rtgs = shortest_path_trajectory(states, rtgs, idx, max_distance)
        new_trajectory, new_rtgs = concat_original_with_new(states, new_trajectory, n, rtgs, new_rtgs)
        trajectories.append(new_trajectory)
        adjusted_rtgs.append(new_rtgs)
    return np.array(trajectories), np.array(adjusted_rtgs)