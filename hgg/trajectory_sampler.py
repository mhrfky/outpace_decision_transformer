import torch
import numpy as np
from hgg.debug_utils import visualize_length_of_output
from hgg.debug_utils import update_plot

def calculate_dynamic_threshold(rtgs, window_size=5, percentile=95):
    conv_kernel = np.ones(window_size) / window_size
    smoothed_rtgs = np.convolve(rtgs, conv_kernel, mode='same')
    differences = np.abs(np.diff(smoothed_rtgs))
    threshold = np.percentile(differences, percentile)
    return threshold

@visualize_length_of_output(update_plot)
def find_decreasing_rtgs(rtgs, segment_length=10, dynamic_threshold=False, window_size=5, percentile=95):
    if dynamic_threshold:
        threshold = calculate_dynamic_threshold(rtgs, window_size, percentile)
    else:
        threshold = 0  # No dynamic threshold

    decreasing_segments = []
    start = 0
    for i in range(1, len(rtgs)):
        if rtgs[i] > rtgs[i - 1] + threshold:
            if i - start >= segment_length:
                decreasing_segments.append((start, i))
            start = i
    if start < len(rtgs) - segment_length + 1:
        decreasing_segments.append((start, len(rtgs)))

    return decreasing_segments

def fit_linear_segment(segment):
    n = segment.shape[0]
    x = torch.ones((n, 2))
    x[:, 1] = torch.arange(n)
    y = torch.tensor(segment, dtype=torch.float32)
    
    coeffs, residuals, rank, s = torch.linalg.lstsq(x, y)
    y_pred = x @ coeffs
    residual_sum_of_squares = torch.sum((y - y_pred) ** 2).item()
    
    return coeffs, residual_sum_of_squares

@visualize_length_of_output(update_plot)
def identify_good_parts(trajectory, rtgs, segment_length=10, residual_threshold=0.01, rtg_threshold=0.1, dynamic_threshold=False, window_size=5, percentile=95):
    good_parts = []
    decreasing_segments = find_decreasing_rtgs(rtgs, segment_length, dynamic_threshold, window_size, percentile)

    for start, end in decreasing_segments:
        segment = trajectory[start:end]
        rtg_segment = rtgs[start:end]
        
        # Fit linear segment and calculate residuals
        _, residuals = fit_linear_segment(rtg_segment)
        
        # Calculate mean RTG for the segment
        mean_rtg = np.mean(rtg_segment)
        
        # Normalize residuals and RTG for combining
        norm_residuals = residuals / segment_length
        norm_rtg = mean_rtg / np.max(rtgs)
        
        # Check if the segment qualifies as a good part
        if norm_residuals < residual_threshold and mean_rtg > rtg_threshold:
            good_parts.append((start, end))
    
    return good_parts


import numpy as np
from scipy.ndimage import uniform_filter1d

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

def check_variance(segment, threshold):
    return np.var(segment) < threshold

@visualize_length_of_output(update_plot)
def select_segment_indices(rtg, window_size, base_diff_threshold, var_threshold, proportion=0.1):
    smoothed_rtg = smooth_rtg(rtg, window_size)
    indices = []
    
    for i in range(2,len(rtg) - window_size - 1):
        segment = smoothed_rtg[i:i+window_size]
        total_diff = end_start_difference(segment)
        threshold = adaptive_threshold(segment, base_diff_threshold, proportion)
        
        if total_diff < threshold :
            indices.append((i, i + window_size - 1))
    
    return indices


