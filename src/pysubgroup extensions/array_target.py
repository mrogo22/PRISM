"""
array_target.py

This module introduces a new target type, ArrayTarget, which handles a single target
variable whose values are arrays of variable length. The arrays are resampled to a
fixed length for subgroup discovery. In addition, a new quality function, ArrayCosineQF,
is provided that computes subgroup quality using a weighted combination of the average
cosine similarity (shape similarity) and a sign consistency measure.
 
Author: Madalina Rogozan
Date: 19/02/2025
"""

from collections import namedtuple
from functools import total_ordering
import numpy as np
from pysubgroup.measures import (
    AbstractInterestingnessMeasure,
    BoundedInterestingnessMeasure,
    GeneralizationAwareQF_stats,
)
from .subgroup_description import EqualitySelector, get_cover_array_and_size
from .utils import BaseTarget, derive_effective_sample_size

# -------------------------------
# Helper functions
# -------------------------------

def resample_array(arr, fixed_length):
    """
    Resample a 1D numpy array to a fixed length using linear interpolation.
    """
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == fixed_length:
        return arr
    old_indices = np.linspace(0, 1, num=n)
    new_indices = np.linspace(0, 1, num=fixed_length)
    return np.interp(new_indices, old_indices, arr)

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    If a vector has zero norm it is replaced with a constant eps vector.
    """
    eps=1e-8
    if np.all(vec1 == 0):
        vec1 = np.full(vec1.shape, eps)
    if np.all(vec2 == 0):
        vec2 = np.full(vec2.shape, eps)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return np.dot(vec1, vec2) / (norm1 * norm2)

def reciprocal_magnitude(arr1, arr2):

    r_m = 1/ (1+ 3* abs(arr1-arr2)/50)
    return r_m

def compute_integral(arr, dx=1):
    """Approximate the integral of a discretized function using the trapezoidal rule."""
    return np.trapz(arr, dx=dx)

def is_negative_then_zero(array):

    seen_zero = False  # Tracks if we have encountered zeroes
    for value in array:
        if value < 0:
            if seen_zero:
                return False  # Encountered a positive number after zeroes
        elif value == 0:
            seen_zero = True  # Start tracking zeroes
        else:
            return False  # Negative values are not allowed
    # Ensure the array has at least one positive number and one zero
    return seen_zero and array[0] < 0

def is_positive_then_zero(array):
    """
    Checks if the array starts with n positive numbers followed by m zeroes (n > 0, m > 0).
    The array must start with at least one positive number and transition to zeroes without returning to positive or encountering negative values.
    """
    seen_zero = False  # Tracks if we have encountered zeroes
    for value in array:
        if value > 0:
            if seen_zero:
                return False  # Encountered a positive number after zeroes
        elif value == 0:
            seen_zero = True  # Start tracking zeroes
        else:
            return False  # Negative values are not allowed
    # Ensure the array has at least one positive number and one zero
    return seen_zero and array[0] > 0

def has_positive_to_negative_transition(array):
    """
    Checks if the array has a transition from negative to positive values
    without returning to negative.
    """
    seen_negative = False  # Tracks if we have encountered negative numbers
    for value in array:
        if value > 0:
            if seen_negative:
                return False  # Encountered a positive number after a negative
        elif value < 0:
            seen_negative = True  # Start tracking negative numbers
        else:
            return False  # Zero is not allowed
    # Ensure the array has at least one positive and one negative number
    return seen_negative and array[0] > 0

def has_negative_to_positive_transition(array):
    """
    Checks if the array has a transition from negative to positive values
    without returning to negative.
    """
    seen_positive = False  # Tracks if we have encountered positive numbers
    for value in array:
        if value < 0:
            if seen_positive:
                return False  # Encountered a negative number after a positive
        elif value > 0:
            seen_positive = True  # Start tracking positive numbers
        else:
            return False  # Zero is not allowed
    # Ensure the array has at least one negative and one positive number
    return seen_positive and array[0] < 0

def detect_trend(arr):
    if is_negative_then_zero(arr):
        return 1
    if is_positive_then_zero(arr):
        return 2
    if has_positive_to_negative_transition(arr):
        return 3
    if has_negative_to_positive_transition(arr):
        return 4
    if all(e > 0 for e in arr):
        return 5
    if all(e < 0 for e in arr):
        return 6
    if all(e == 0 for e in arr):
        return 7
    return 8  # Default for rows not matching the condition

def trend_similarity(arr1, arr2):
    trend1 = detect_trend(arr1)
    trend2 = detect_trend(arr2)
    if trend1 == trend2:
        return 1
    if trend1 == 1 and trend2 == 7:
        return 0.5
    if trend1 == 7 and trend2 == 1:
        return 0.5
    if trend1 == 2 and trend2 == 7:
        return 0.5
    if trend1 == 7 and trend2 == 2:
        return 0.5
    if trend1 == 3 and trend2 == 6:
        return 0.5
    if trend1 == 6 and trend2 == 3:
        return 0.5
    if trend1 == 4 and trend2 == 5:
        return 0.5
    if trend1 == 5 and trend2 == 4:
        return 0.5
    if trend1 ==8 and trend2 != 8:
        return 0.25
    if trend2 ==8 and trend1 != 8:
        return 0.25
    return 0
# -------------------------------
# ArrayTarget
# -------------------------------

@total_ordering
class ArrayTarget(BaseTarget):
    """
    Target type for array-valued target variables.
    
    Each instance is assumed to have a target value (from column `target_variable`)
    that is an array (or list) of numbers. Since these arrays may have variable length,
    they are resampled to a fixed length (default 10) for computation.
    
    The computed statistics include:
      - size_sg: number of instances in subgroup
      - size_dataset: total number of instances
      - centroid_sg: elementwise mean of resampled arrays in subgroup (as 1D np.array)
      - centroid_dataset: elementwise mean of resampled arrays in dataset
      - avg_cosine: average cosine similarity of each subgroup instance’s array with centroid_sg
      - sign_consistency: average fraction of positions in each instance whose sign (np.sign)
                           agrees with that of centroid_sg.
    """
    # Define the names for statistics we compute
    statistic_types = (
        "size_sg",
        "size_cover_all",
        "covered_not_in_sg",
        "size_dataset",
        "centroid_sg",
        "centroid_dataset",
        "avg_cosine",
        "sign_consistency",
        "avg_trend_match",
        "avg_rec_magnitude",
    )

    def __init__(self, target_variable, fixed_length=10, initial_data=None):
        self.target_variable = target_variable
        self.fixed_length = fixed_length
        self.initial_data = initial_data

    def __repr__(self):
        return "ArrayTarget: " + str(self.target_variable)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [self.target_variable]

    def get_base_statistics(self, subgroup, data):
        # Get subgroup cover array and its size
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        # Assume data[target_variable] entries are lists/arrays.
        # Convert each entry to a numpy array and resample it.
        # Here we assume data is a pandas DataFrame.
        all_target_series = data[self.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        # Stack to form a 2D array: shape (n_instances, fixed_length)
        resampled_all = np.vstack(resampled.to_numpy())
        cover_all = subgroup.covers(self.initial_data)
        size_cover_all = np.count_nonzero(cover_all)
        covered_not_in_sg = size_cover_all - size_sg
        # Subgroup arrays:
        subgroup_values = resampled_all[cover_arr]
        if subgroup_values.shape[0] > 0:
            centroid_sg = np.mean(subgroup_values, axis=0)
            # Compute cosine similarity for each instance in subgroup vs. centroid
            cosine_sims = np.array([cosine_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_cosine = np.mean(cosine_sims)
            # Compute sign consistency: fraction of elements with same sign as centroid
            sign_matches = np.array([np.mean(np.sign(vec) == np.sign(centroid_sg)) for vec in subgroup_values])
            sign_consistency = np.mean(sign_matches)
            trend_matches = np.array([trend_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_trend_match = np.mean(trend_matches)
            reciprocal_magnitudes_array =  np.array([reciprocal_magnitude(vec, centroid_sg) for vec in subgroup_values])
            avg_rec_magnitude = np.mean(reciprocal_magnitudes_array)
            

        else:
            centroid_sg = np.zeros(self.fixed_length)
            avg_cosine = 0.0
            sign_consistency = 0.0
            avg_trend_match = 0.0
            avg_rec_magnitude = 0.0

        centroid_dataset = np.mean(resampled_all, axis=0)
        size_dataset = len(data)
        return (size_sg, size_cover_all, covered_not_in_sg, size_dataset, centroid_sg, centroid_dataset, avg_cosine, sign_consistency, avg_trend_match, avg_rec_magnitude)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        stats = {}
        (size_sg, size_cover_all,covered_not_in_sg, size_dataset, centroid_sg, centroid_dataset, avg_cosine, sign_consistency,avg_trend_match,avg_rec_magnitude) = self.get_base_statistics(subgroup, data)
        stats["size_sg"] = size_sg
        stats["size_cover_all"] = size_cover_all
        stats["covered_not_in_sg"] = covered_not_in_sg
        stats["size_dataset"] = size_dataset
        stats["centroid_sg"] = centroid_sg
        stats["centroid_dataset"] = centroid_dataset
        stats["avg_cosine"] = avg_cosine
        stats["sign_consistency"] = sign_consistency
        stats["avg_trend_match"] = avg_trend_match
        stats["avg_rec_magnitude"] = avg_rec_magnitude

        return stats

# -------------------------------
# ArrayCosineQF
# -------------------------------

class ArrayCosineQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using cosine similarity and sign consistency.
    
    The quality is computed as:
       Q = α * (avg_cosine) + (1 - α) * (sign_consistency)
    
    where:
      - avg_cosine is the average cosine similarity of each instance's resampled array
        (in the subgroup) with the subgroup centroid.
      - sign_consistency is the average fraction of positions in the array that match
        the sign (np.sign) of the subgroup centroid.
    
    This quality function is designed to work with the gp_growth algorithm.
    """
    tpl = namedtuple("ArrayCosineQF_tpl", ["avg_cosine", "avg_trend_match", "quality", "size_sg"])

    def __init__(self, alpha=0.5, fixed_length=10, min_size_sg = 5):
        self.alpha = alpha
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
        # Compute constant (dataset-level) statistics for the target arrays.
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        centroid_dataset = np.mean(resampled_all, axis=0)
        cosine_sims = np.array([cosine_similarity(vec, centroid_dataset) for vec in resampled_all])
        avg_cosine_dataset = np.mean(cosine_sims)
        sign_matches = np.array([np.mean(np.sign(vec) == np.sign(centroid_dataset)) for vec in resampled_all])
        sign_consistency_dataset = np.mean(sign_matches)
        size_dataset = len(data)
        self.dataset_statistics = self.tpl(avg_cosine_dataset, sign_consistency_dataset, None, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]
        if subgroup_values.shape[0] > 0:
            centroid_sg = np.mean(subgroup_values, axis=0)
            cosine_sims = np.array([cosine_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_cosine = np.mean(cosine_sims)
            sign_matches = np.array([np.mean(np.sign(vec) == np.sign(centroid_sg)) for vec in subgroup_values])
            sign_consistency = np.mean(sign_matches)
            trend_matches = np.array([trend_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_trend_match = np.mean(trend_matches)
        else:
            centroid_sg = np.zeros(self.fixed_length)
            avg_cosine = 0.0
            sign_consistency = 0.0
            avg_trend_match = 0.0
            #sg quality computed here
        if size_sg < self.min_size_sg-10:
            size_deviation = (self.min_size_sg - size_sg) / self.min_size_sg
            quality = self.alpha * avg_cosine + (1 - self.alpha) * avg_trend_match - size_deviation
        else:
            quality = self.alpha * avg_cosine + (1 - self.alpha) * avg_trend_match
            size_deviation = 0.0
        return self.tpl(avg_cosine, avg_trend_match, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        # For gp_growth we assume a row corresponds to a candidate selector.
        # Here we simply return a placeholder vector. For a more refined
        # implementation one could precompute stats from the resampled arrays.
        return np.zeros(4)

    def gp_get_null_vector(self):
        return np.zeros(4)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        # For simplicity, assume v already holds our tuple components.
        return self.tpl(v[0], v[1], v[2], v[3])

    def gp_to_str(self, stats):
        return f"cosine: {stats.avg_cosine:.3f}, sign: {stats.sign_consistency:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)
    
# -------------------------------
# ArrayCosineQF
# -------------------------------

class ArraySignMagnitudeQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using reciprocal magnitude and sign consistency.
    
    The quality is computed as:
       Q = α * (sign_consistency) + (1 - α) * (reciprocal magnitude function)
    
    where:
      
      - sign_consistency is the average fraction of positions in the array that match
        the sign (np.sign) of the subgroup centroid.
    
    This quality function is designed to work with the gp_growth algorithm.
    """
    tpl = namedtuple("ArraySignMagnitudeQF_tpl", ["sign_consistency", "avg_magnitude_penalty", "quality", "size_sg"])

    def __init__(self, alpha=0.5, fixed_length=10, min_size_sg = 5, initial_data=None):
        self.alpha = alpha
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None
        self.initial_data = initial_data

    def calculate_constant_statistics(self, data, target):
        # Compute constant (dataset-level) statistics for the target arrays.
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        centroid_dataset = np.mean(resampled_all, axis=0)
        cosine_sims = np.array([cosine_similarity(vec, centroid_dataset) for vec in resampled_all])
        avg_cosine_dataset = np.mean(cosine_sims)
        sign_matches = np.array([np.mean(np.sign(vec) == np.sign(centroid_dataset)) for vec in resampled_all])
        sign_consistency_dataset = np.mean(sign_matches)
        size_dataset = len(data)
        self.dataset_statistics = self.tpl(avg_cosine_dataset, sign_consistency_dataset, None, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        cover_all = subgroup.covers(self.initial_data)
        size_cover_all = np.count_nonzero(cover_all)
        covered_not_in_sg = size_cover_all - size_sg
        subgroup_values = resampled_all[cover_arr]
        # compute cover sizes
        if subgroup_values.shape[0] > 0:
            centroid_sg = np.mean(subgroup_values, axis=0)
            cosine_sims = np.array([cosine_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_cosine = np.mean(cosine_sims)
            reciprocal_magnitudes_array =  np.array([reciprocal_magnitude(vec, centroid_sg) for vec in subgroup_values])
            avg_rec_magnitude = np.mean(reciprocal_magnitudes_array)
            sign_matches = np.array([np.mean(np.sign(vec) == np.sign(centroid_sg)) for vec in subgroup_values])
            sign_consistency = np.mean(sign_matches)

        else:
            centroid_sg = np.zeros(self.fixed_length)
            avg_cosine = 0.0
            avg_rec_magnitude = 0.0
            sign_consistency = 0.0

            #sg quality computed here
        if size_sg < self.min_size_sg-10:
            size_deviation = (self.min_size_sg - size_sg) / self.min_size_sg
            quality = self.alpha * sign_consistency + (1 - self.alpha) * avg_rec_magnitude - size_deviation
        else:
            quality = self.alpha * sign_consistency + (1 - self.alpha) * avg_rec_magnitude
            size_deviation = 0.0
        # if more than 10% not in sg quality = 0
        if covered_not_in_sg > 0:
            quality = 0
        return self.tpl(sign_consistency, avg_rec_magnitude, quality, size_sg)



    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        # For gp_growth we assume a row corresponds to a candidate selector.
        # Here we simply return a placeholder vector. For a more refined
        # implementation one could precompute stats from the resampled arrays.
        return np.zeros(4)

    def gp_get_null_vector(self):
        return np.zeros(4)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        # For simplicity, assume v already holds our tuple components.
        return self.tpl(v[0], v[1], v[2], v[3])

    def gp_to_str(self, stats):
        return f"cosine: {stats.avg_cosine:.3f}, sign: {stats.sign_consistency:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)
    
# -------------------------------
# IntegralQF
# -------------------------------
class ArrayIntegralQF(AbstractInterestingnessMeasure):
    """
        Quality function for ArrayTarget assuming PE array as a discretized function. QF uses normalized diff in integrals.

        The quality is computed as:
            Q = 1 - alpha * (abs(mean(I - I_centroid))) / (abs(I_centroid) + eps) - size_deviation

        where:
        - I is the integral of each instance's resampled array
        - I_centroid is the integral of the centroid of the subgroup
        - alpha is a parameter to control the importance of the integral difference
        - size_deviation penalizes subgroups that are too small

        This quality function is designed to work with the gp_growth algorithm.
        """   
    tpl = namedtuple("IntegralQF_tpl", ["avg_integral_diff", "quality", "size_sg"])
    
    def __init__(self, alpha=0.5, fixed_length=10, min_size_sg = 5, eps=1e-8):
        self.alpha = alpha
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.eps = eps
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None
    
    def calculate_constant_statistics(self, data, target):
        # Compute constant (dataset-level) statistics for the target arrays.
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        centroid_dataset = np.mean(resampled_all, axis=0)

        integrals = np.array([compute_integral(vec, dx=1) for vec in resampled_all])
        I_centroid_ds = compute_integral(centroid_dataset, dx=1) 
 
        avg_integral_diff = np.mean(np.abs(integrals - I_centroid_ds))
        
        size_dataset = len(data)
        quality_dataset = 1 - self.alpha * (avg_integral_diff) / (abs(I_centroid_ds) + self.eps)
        self.dataset_statistics = self.tpl(avg_integral_diff, quality_dataset, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]
        if subgroup_values.shape[0] > 0:
            centroid_sg = np.mean(subgroup_values, axis=0)
            cosine_sims = np.array([cosine_similarity(vec, centroid_sg) for vec in subgroup_values])
            # Compute the integral of the subgroup centroid.
            I_centroid = compute_integral(centroid_sg, dx=1)
            # Compute the integral for each instance in the subgroup.
            integrals = np.array([compute_integral(arr, dx=1) for arr in subgroup_values])
            # Compute the average absolute difference in integrals.
            avg_integral_diff = np.mean(np.abs(integrals - I_centroid))
            avg_cosine = np.mean(cosine_sims)
            sign_matches = np.array([np.mean(np.sign(vec) == np.sign(centroid_sg)) for vec in subgroup_values])
            sign_consistency = np.mean(sign_matches)
            trend_matches = np.array([trend_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_trend_match = np.mean(trend_matches)
        else:
            centroid_sg = np.zeros(self.fixed_length)
            avg_cosine = 0.0
            avg_integral_diff = 0.0
            I_centroid = 0.0
            sign_consistency = 0.0
            avg_trend_match = 0.0
        # Penalize subgroups that are too small.
        if size_sg < self.min_size_sg:
            size_deviation = (self.min_size_sg - size_sg) / self.min_size_sg
        else:
            size_deviation = 0.0
        
        quality = 1 - self.alpha * (avg_integral_diff) / (abs(I_centroid) + self.eps) - size_deviation
        neg_alpha = self.alpha * (-1)
        quality2 = np.exp(neg_alpha * np.abs(avg_integral_diff)) - size_deviation
        return self.tpl(avg_integral_diff, quality2, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        # Return a placeholder vector of the same length as our tpl (4 elements).
        return np.zeros(4)

    def gp_get_null_vector(self):
        return np.zeros(4)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        # Assume v already holds our tuple components.
        return tpl(v[0], v[1], v[2])

    def gp_to_str(self, stats):
        return f"avg_int_diff: {stats.avg_integral_diff:.3f}, quality: {stats.quality:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)
    
        

# -------------------------------
# ArrayEuclideanQF
# -------------------------------

class ArrayEuclideanQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using normalization and Euclidean distance.
    
    For each instance, its resampled array is first normalized (subtract its mean
    and divide by its standard deviation). Then, the subgroup centroid is computed as
    the elementwise mean of the normalized arrays. The Euclidean distance of each
    normalized array from the subgroup centroid is calculated, and the quality is defined as:
    
         Q = 1 / (1 + avg_distance)
    
    where avg_distance is the average Euclidean distance of the subgroup's normalized arrays
    from the subgroup centroid. A smaller distance (i.e. more similar shapes) yields a higher quality.
    
    This quality function is designed to work with the gp_growth algorithm.
    """
    tpl = namedtuple("ArrayEuclideanQF_tpl", ["avg_distance", "quality", "size_sg"])

    def __init__(self, fixed_length=10, min_size_sg=5):
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def normalize_array(self, arr):
        """
        Normalize a 1D array by subtracting its mean and dividing by its standard deviation.
        If the standard deviation is zero, returns an array of zeros.
        """
        arr = np.asarray(arr, dtype=float)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return arr - mean
        return (arr - mean) / std

    def calculate_constant_statistics(self, data, target):
        # Compute dataset-level statistics.
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        # Normalize each array
        normalized_all = np.vstack([self.normalize_array(arr) for arr in resampled_all])
        # Compute the dataset centroid (mean of normalized arrays)
        centroid_dataset = np.mean(normalized_all, axis=0)
        # Compute Euclidean distances for each instance from the centroid
        distances = np.linalg.norm(normalized_all - centroid_dataset, axis=1)
        avg_distance_dataset = np.mean(distances)
        size_dataset = len(data)
        quality_dataset = 1 / (1 + avg_distance_dataset)
        self.dataset_statistics = self.tpl(avg_distance_dataset, quality_dataset, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]
        if subgroup_values.shape[0] > 0:
            # Normalize each instance's array in the subgroup
            normalized_values = np.vstack([self.normalize_array(vec) for vec in subgroup_values])
            # Compute subgroup centroid (elementwise mean of normalized arrays)
            centroid_sg = np.mean(normalized_values, axis=0)
            # Compute Euclidean distance for each normalized array from the centroid
            distances = np.linalg.norm(normalized_values - centroid_sg, axis=1)
            avg_distance = np.mean(distances)
        else:
            centroid_sg = np.zeros(self.fixed_length)
            avg_distance = 0.0
        if size_sg < self.min_size_sg:
            quality = 0
        else:
            quality = 1 / (1 + avg_distance)
        return self.tpl(avg_distance, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        # Placeholder implementation for gp_growth
        return np.zeros(3)

    def gp_get_null_vector(self):
        return np.zeros(3)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        # Assume v already holds our tuple components.
        return self.tpl(v[0], v[1], v[2])

    def gp_to_str(self, stats):
        return f"avg_dist: {stats.avg_distance:.3f}, quality: {stats.quality:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)

# -------------------------------
# ArrayFirstDiffQF
# -------------------------------

class ArrayFirstDiffQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using first-difference analysis.
    
    For each instance, its resampled array (of fixed_length) is transformed into its
    first differences (i.e. differences between consecutive elements). For a subgroup,
    the centroid is computed as the elementwise mean of the first-difference vectors.
    The Euclidean distance between each instance’s first-difference vector and the subgroup
    centroid is computed, and the quality is defined as:
    
         Q = 1 / (1 + avg_diff_distance)
    
    where avg_diff_distance is the average Euclidean distance of subgroup items’ first-difference
    vectors from the subgroup’s first-difference centroid. A smaller distance indicates more similar
    temporal evolution of differences.
    
    This quality function is designed to work with the gp_growth algorithm.
    """
    tpl = namedtuple("ArrayFirstDiffQF_tpl", ["avg_diff_distance", "quality", "size_sg"])

    def __init__(self, fixed_length=10, min_size_sg=5):
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def first_differences(self, arr):
        """
        Compute the first differences of a 1D array.
        For an array of length L, returns an array of length (L-1).
        """
        return np.diff(arr)

    def calculate_constant_statistics(self, data, target):
        # Compute constant (dataset-level) statistics for first differences.
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        # Compute first differences for each instance (resulting shape: (n_instances, fixed_length-1))
        diffs_all = np.vstack([self.first_differences(arr) for arr in resampled_all])
        # Compute the dataset-level centroid for first differences:
        centroid_dataset = np.mean(diffs_all, axis=0)
        # Compute Euclidean distance from each instance's first differences to the centroid:
        distances = np.linalg.norm(diffs_all - centroid_dataset, axis=1)
        avg_distance_dataset = np.mean(distances)
        size_dataset = len(data)
        quality_dataset = 1 / (1 + avg_distance_dataset)
        self.dataset_statistics = self.tpl(avg_distance_dataset, quality_dataset, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]
        if subgroup_values.shape[0] > 0:
            # Compute first differences for each instance in the subgroup.
            diffs = np.vstack([self.first_differences(vec) for vec in subgroup_values])
            # Compute subgroup centroid (elementwise mean of first differences)
            centroid_sg = np.mean(diffs, axis=0)
            # Compute Euclidean distances of each instance's first differences from the centroid.
            distances = np.linalg.norm(diffs - centroid_sg, axis=1)
            avg_diff_distance = np.mean(distances)
        else:
            avg_diff_distance = 0.0
        if size_sg < self.min_size_sg:
            quality = 0
        else:
            quality = 1 / (1 + avg_diff_distance)
        return self.tpl(avg_diff_distance, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        # Placeholder implementation for gp_growth.
        return np.zeros(3)

    def gp_get_null_vector(self):
        return np.zeros(3)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        # Assume v already holds our tuple components.
        return self.tpl(v[0], v[1], v[2])

    def gp_to_str(self, stats):
        return f"avg_diff: {stats.avg_diff_distance:.3f}, quality: {stats.quality:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)


# -------------------------------
# Helper functions for DTW
# -------------------------------

def dtw_distance(seq1, seq2):
    """
    Compute the dynamic time warping (DTW) distance between two 1D sequences.
    Uses a simple dynamic programming approach.
    """
    n = len(seq1)
    m = len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match
    return dtw[n, m]

def compute_medoid(arrays):
    """
    Compute the medoid of a set of arrays (2D numpy array where each row is an instance).
    The medoid is defined as the array that minimizes the average DTW distance to all others.
    Returns the medoid and the average DTW distance of all arrays to the medoid.
    """
    N = arrays.shape[0]
    if N == 0:
        return None, 0.0
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance(arrays[i], arrays[j])
            distances[i, j] = d
            distances[j, i] = d
    avg_distances = distances.mean(axis=1)
    medoid_idx = np.argmin(avg_distances)
    medoid = arrays[medoid_idx]
    # Compute average DTW distance to the medoid.
    avg_dtw = np.mean([dtw_distance(arrays[i], medoid) for i in range(N)])
    return medoid, avg_dtw

# -------------------------------
# ArrayDTWQF
# -------------------------------

class ArrayDTWQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using Dynamic Time Warping (DTW).
    
    Each target array is first resampled to a fixed length. For a subgroup,
    we compute the medoid—the instance that minimizes the average DTW distance
    to all other subgroup instances—and then calculate the average DTW distance
    from every instance in the subgroup to the medoid.
    
    The quality is defined as:
    
         Q = 1 / (1 + avg_dtw)
    
    where avg_dtw is the average DTW distance. A smaller avg_dtw (i.e. more similar
    sequences) results in a higher quality score.
    
    This quality function is designed to work with the gp_growth algorithm.
    """
    tpl = namedtuple("ArrayDTWQF_tpl", ["avg_dtw", "quality", "size_sg"])

    def __init__(self, fixed_length=10, min_size_sg=5):
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
        all_target_series = data[target.target_variable]
        # Resample each array to the fixed length.
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        # Compute dataset medoid and average DTW distance.
        _, avg_dtw_dataset = compute_medoid(resampled_all)
        size_dataset = len(data)
        quality_dataset = 1 / (1 + avg_dtw_dataset)
        self.dataset_statistics = self.tpl(avg_dtw_dataset, quality_dataset, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]
        if subgroup_values.shape[0] > 0:
            # Compute the medoid and average DTW distance for the subgroup.
            _, avg_dtw = compute_medoid(subgroup_values)
        else:
            avg_dtw = 0.0
        if size_sg < self.min_size_sg:
            quality = 0
        else:
            quality = 1 / (1 + avg_dtw)
        return self.tpl(avg_dtw, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        # Placeholder for gp_growth support.
        return np.zeros(3)

    def gp_get_null_vector(self):
        return np.zeros(3)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        return self.tpl(v[0], v[1], v[2])

    def gp_to_str(self, stats):
        return f"avg_dtw: {stats.avg_dtw:.3f}, quality: {stats.quality:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)

# -------------------------------
# Helper functions for DTW with Sign Penalty
# -------------------------------

def dtw_distance_with_sign(seq1, seq2, sign_penalty=1.0):
    """
    Compute the dynamic time warping (DTW) distance between two 1D sequences,
    adding an extra cost (sign_penalty) if the signs of the compared elements differ.
    """
    n = len(seq1)
    m = len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(seq1[i-1] - seq2[j-1])
            # If the signs differ, add the penalty
            if np.sign(seq1[i-1]) != np.sign(seq2[j-1]):
                cost += sign_penalty
            dtw[i, j] = cost + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            )
    return dtw[n, m]

def compute_medoid_with_sign(arrays, sign_penalty=1.0):
    """
    Compute the medoid of a set of arrays using DTW with sign penalty.
    Returns the medoid and the average DTW distance (using dtw_distance_with_sign)
    of all arrays to the medoid.
    """
    N = arrays.shape[0]
    if N == 0:
        return None, 0.0
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = dtw_distance_with_sign(arrays[i], arrays[j], sign_penalty)
            distances[i, j] = d
            distances[j, i] = d
    avg_distances = distances.mean(axis=1)
    medoid_idx = np.argmin(avg_distances)
    medoid = arrays[medoid_idx]
    avg_dtw = np.mean([dtw_distance_with_sign(arrays[i], medoid, sign_penalty) for i in range(N)])
    return medoid, avg_dtw

# -------------------------------
# ArrayDTWWithSignQF
# -------------------------------

class ArrayDTWWithSignQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using Dynamic Time Warping (DTW) with sign penalty.
    
    Each target array is first resampled to a fixed length. For a subgroup,
    the medoid is computed using DTW distances that add a penalty (sign_penalty)
    whenever the signs of compared elements differ. The quality is defined as:
    
         Q = 1 / (1 + avg_dtw)
    
    where avg_dtw is the average DTW distance (with sign penalty) of all subgroup
    arrays to the medoid. A smaller avg_dtw indicates more similar sequences, including
    similarity in both shape and sign.
    
    This quality function is designed to work with the gp_growth algorithm.
    """
    tpl = namedtuple("ArrayDTWWithSignQF_tpl", ["avg_dtw", "quality", "size_sg"])

    def __init__(self, fixed_length=10, min_size_sg=5, sign_penalty=1.0):
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.sign_penalty = sign_penalty
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        # Compute the dataset medoid and average DTW distance with sign penalty.
        _, avg_dtw_dataset = compute_medoid_with_sign(resampled_all, self.sign_penalty)
        size_dataset = len(data)
        quality_dataset = 1 / (1 + avg_dtw_dataset)
        self.dataset_statistics = self.tpl(avg_dtw_dataset, quality_dataset, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]
        if subgroup_values.shape[0] > 0:
            # Compute the medoid and average DTW distance (with sign penalty) for the subgroup.
            _, avg_dtw = compute_medoid_with_sign(subgroup_values, self.sign_penalty)
        else:
            avg_dtw = 0.0
        if size_sg < self.min_size_sg:
            quality = 0
        else:
            quality = 1 / (1 + avg_dtw)
        return self.tpl(avg_dtw, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        return np.zeros(3)

    def gp_get_null_vector(self):
        return np.zeros(3)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        return self.tpl(v[0], v[1], v[2])

    def gp_to_str(self, stats):
        return f"avg_dtw: {stats.avg_dtw:.3f}, quality: {stats.quality:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)



def canberra_distance(vec1, vec2):
    """
    Computes the Canberra distance between two vectors.
    If both vec1[i] and vec2[i] are zero, the term is taken as zero.
    """
    distance = 0.0
    for x, y in zip(vec1, vec2):
        denom = abs(x) + abs(y)
        if denom != 0:
            distance += abs(x - y) / denom
    return distance
# -------------------------------
# ArrayCanberraQF
# -------------------------------

class ArrayCanberraQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using the Canberra distance and trend similarity.

    The quality is computed as:
       Q = alpha * (avg_canberra_similarity) + (1 - alpha) * (avg_trend_match)
    
    where:
      - avg_canberra_similarity is derived from the average Canberra distance of each
        subgroup instance’s resampled array to the subgroup centroid (smaller distance
        => higher similarity).
      - avg_trend_match is the average trend similarity (0 to 1) comparing each array
        to the centroid’s detected trend.
      - alpha is a weight parameter in [0, 1].
    
    A penalty is subtracted if the subgroup size is smaller than `min_size_sg`.
    """

    tpl = namedtuple("ArrayCanberraQF_tpl", ["avg_canberra_sim", "avg_trend_match", "quality", "size_sg"])

    def __init__(self, alpha=0.5, fixed_length=10, min_size_sg=5):
        """
        :param alpha: Weight for combining Canberra-based similarity and trend match.
        :param fixed_length: Length to which arrays are resampled.
        :param min_size_sg: Minimum subgroup size; subgroups smaller than this
                            incur a quality penalty.
        """
        self.alpha = alpha
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
        """
        Computes and stores dataset-level statistics (e.g., centroid, average distance).
        This is optional but follows the pattern used in ArrayCosineQF.
        """
        all_target_series = data[target.target_variable]
        # Resample all arrays to the fixed length
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())

        # Compute the dataset centroid
        centroid_dataset = np.mean(resampled_all, axis=0)

        # Average Canberra distance to the dataset centroid
        canberra_dists = np.array([canberra_distance(vec, centroid_dataset) for vec in resampled_all])
        avg_canberra_dist_dataset = np.mean(canberra_dists)
        # Convert distance to similarity (larger distance => smaller similarity)
        avg_canberra_sim_dataset = 1.0 / (1.0 + avg_canberra_dist_dataset)

        # Trend match with the dataset centroid
        trend_matches = np.array([trend_similarity(vec, centroid_dataset) for vec in resampled_all])
        avg_trend_match_dataset = np.mean(trend_matches)

        size_dataset = len(data)
        # Quality is None at the dataset level (not used directly)
        self.dataset_statistics = self.tpl(avg_canberra_sim_dataset, avg_trend_match_dataset, None, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        """
        Calculates the subgroup statistics needed for evaluating quality.
        Returns a namedtuple containing (avg_canberra_sim, avg_trend_match, quality, size_sg).
        """
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        # Resample all arrays to the fixed length
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]

        if subgroup_values.shape[0] > 0:
            # Compute centroid for the subgroup
            centroid_sg = np.mean(subgroup_values, axis=0)

            # Compute the average Canberra distance to the subgroup centroid
            canberra_dists = np.array([canberra_distance(vec, centroid_sg) for vec in subgroup_values])
            avg_canberra_dist = np.mean(canberra_dists)
            # Convert distance to similarity
            avg_canberra_sim = 1.0 / (1.0 + avg_canberra_dist)

            # Compute the average trend match
            trend_matches = np.array([trend_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_trend_match = np.mean(trend_matches)
        else:
            # If subgroup is empty, set all metrics to 0
            avg_canberra_sim = 0.0
            avg_trend_match = 0.0

        # Apply penalty if subgroup is too small
        if size_sg < self.min_size_sg - 10:
            size_deviation = (self.min_size_sg - size_sg) / self.min_size_sg
            quality = self.alpha * avg_canberra_sim + (1 - self.alpha) * avg_trend_match - size_deviation
        else:
            quality = self.alpha * avg_canberra_sim + (1 - self.alpha) * avg_trend_match

        return self.tpl(avg_canberra_sim, avg_trend_match, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        """
        Computes the final quality measure for a subgroup.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """
        Returns an optimistic estimate of the quality (used in branch-and-bound search).
        Here we simply return the same as evaluate, but more sophisticated bounds
        can be used.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth or similar frameworks ---

    def gp_get_stats(self, row_index):
        # For gp_growth we assume a row corresponds to a candidate selector.
        # Here we simply return a placeholder vector. One could store
        # precomputed stats for partial covers if desired.
        return np.zeros(4)

    def gp_get_null_vector(self):
        return np.zeros(4)

    def gp_merge(self, left, right):
        # Merging two partial covers in gp_growth (placeholder).
        left += right

    def gp_get_params(self, cover_arr, v):
        # Interprets the array 'v' as the statistics we need.
        return self.tpl(v[0], v[1], v[2], v[3])

    def gp_to_str(self, stats):
        return f"canberra_sim: {stats.avg_canberra_sim:.3f}, trend: {stats.avg_trend_match:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        """
        Provide dataset-level attributes (like avg_canberra_sim_dataset)
        if requested and not found on this object.
        """
        return getattr(self.dataset_statistics, name, None)
    
def sign_aware_manhattan_distance(vec1, vec2, penalty_factor=2):
    """
    Computes a sign-aware Manhattan distance between two vectors.
    For each element, if both values have the same sign (including zero),
    the difference is computed as |x - y|. If the signs differ, the difference
    is multiplied by penalty_factor.
    """
    distance = 0.0
    for x, y in zip(vec1, vec2):
        diff = abs(x - y)
        if np.sign(x) == np.sign(y):
            distance += diff
        else:
            distance += penalty_factor * diff
    return distance

class ArraySignAwareManhattanQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using a sign-aware Manhattan distance and trend similarity.

    The quality is computed as:
       Q = alpha * (avg_sign_aware_similarity) + (1 - alpha) * (avg_trend_match)
    
    where:
      - avg_sign_aware_similarity is derived from the average sign-aware Manhattan distance of each
        subgroup instance’s resampled array to the subgroup centroid, converted into similarity as:
          similarity = 1 / (1 + distance)
        (smaller distance implies higher similarity).
      - avg_trend_match is the average trend similarity (ranging from 0 to 1) comparing each array
        to the centroid’s detected trend.
      - alpha is a weight parameter in [0, 1].
    
    A penalty is subtracted if the subgroup size is smaller than `min_size_sg`.
    """

    tpl = namedtuple("ArraySignAwareManhattanQF_tpl", ["avg_sign_aware_sim", "avg_trend_match", "quality", "size_sg"])

    def __init__(self, alpha=0.5, fixed_length=10, min_size_sg=5, penalty_factor=2):
        """
        :param alpha: Weight for combining sign-aware Manhattan-based similarity and trend match.
        :param fixed_length: Length to which arrays are resampled.
        :param min_size_sg: Minimum subgroup size; subgroups smaller than this incur a quality penalty.
        :param penalty_factor: Factor to penalize differences when the signs differ.
        """
        self.alpha = alpha
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.penalty_factor = penalty_factor
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
        """
        Computes and stores dataset-level statistics (e.g., centroid, average distance).
        This is optional but follows the pattern used in ArrayCanberraQF.
        """
        all_target_series = data[target.target_variable]
        # Resample all arrays to the fixed length
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())

        # Compute the dataset centroid
        centroid_dataset = np.mean(resampled_all, axis=0)

        # Average sign-aware Manhattan distance to the dataset centroid
        distances = np.array([sign_aware_manhattan_distance(vec, centroid_dataset, self.penalty_factor) 
                              for vec in resampled_all])
        avg_distance = np.mean(distances)
        # Convert distance to similarity (larger distance => smaller similarity)
        avg_sign_aware_sim_dataset = 1.0 / (1.0 + avg_distance)

        # Trend match with the dataset centroid
        trend_matches = np.array([trend_similarity(vec, centroid_dataset) for vec in resampled_all])
        avg_trend_match_dataset = np.mean(trend_matches)

        size_dataset = len(data)
        # Quality is None at the dataset level (not used directly)
        self.dataset_statistics = self.tpl(avg_sign_aware_sim_dataset, avg_trend_match_dataset, None, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        """
        Calculates the subgroup statistics needed for evaluating quality.
        Returns a namedtuple containing (avg_sign_aware_sim, avg_trend_match, quality, size_sg).
        """
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        # Resample all arrays to the fixed length
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]

        if subgroup_values.shape[0] > 0:
            # Compute centroid for the subgroup
            centroid_sg = np.mean(subgroup_values, axis=0)

            # Compute the average sign-aware Manhattan distance to the subgroup centroid
            distances = np.array([sign_aware_manhattan_distance(vec, centroid_sg, self.penalty_factor)
                                  for vec in subgroup_values])
            avg_distance = np.mean(distances)
            # Convert distance to similarity
            avg_sign_aware_sim = 1.0 / (1.0 + avg_distance)

            # Compute the average trend match
            trend_matches = np.array([trend_similarity(vec, centroid_sg) for vec in subgroup_values])
            avg_trend_match = np.mean(trend_matches)
        else:
            # If subgroup is empty, set all metrics to 0
            avg_sign_aware_sim = 0.0
            avg_trend_match = 0.0

        # Apply penalty if subgroup is too small
        if size_sg < self.min_size_sg - 10:
            size_deviation = (self.min_size_sg - size_sg) / self.min_size_sg
            quality = self.alpha * avg_sign_aware_sim + (1 - self.alpha) * avg_trend_match - size_deviation
        else:
            quality = self.alpha * avg_sign_aware_sim + (1 - self.alpha) * avg_trend_match

        return self.tpl(avg_sign_aware_sim, avg_trend_match, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        """
        Computes the final quality measure for a subgroup.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """
        Returns an optimistic estimate of the quality (used in branch-and-bound search).
        Here we simply return the same as evaluate, but more sophisticated bounds can be used.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth or similar frameworks ---

    def gp_get_stats(self, row_index):
        # For gp_growth we assume a row corresponds to a candidate selector.
        # Here we simply return a placeholder vector. One could store
        # precomputed stats for partial covers if desired.
        return np.zeros(4)

    def gp_get_null_vector(self):
        return np.zeros(4)

    def gp_merge(self, left, right):
        # Merging two partial covers in gp_growth (placeholder).
        left += right

    def gp_get_params(self, cover_arr, v):
        # Interprets the array 'v' as the statistics we need.
        return self.tpl(v[0], v[1], v[2], v[3])

    def gp_to_str(self, stats):
        return f"sim: {stats.avg_sign_aware_sim:.3f}, trend: {stats.avg_trend_match:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        """
        Provide dataset-level attributes (like avg_sign_aware_sim_dataset)
        if requested and not found on this object.
        """
        return getattr(self.dataset_statistics, name, None)

def poly_fit_coeffs(arr, degree, fixed_length):
    """
    Fits a polynomial of given degree to the 1D array `arr` (which is assumed
    to have been resampled to `fixed_length` points) and returns the coefficients.
    The independent variable is generated as a uniform grid in [0, 1].
    """
    x = np.linspace(0, 1, fixed_length)
    coeffs = np.polyfit(x, arr, degree)
    return coeffs

class ArrayCurveFitQF(AbstractInterestingnessMeasure):
    """
    Quality function for ArrayTarget using curve fitting to represent the array shape.

    Each array (assumed to be a 1D numerical array) is first resampled to a fixed
    length, and then a polynomial of degree `degree` is fitted to it. The fitted
    polynomial coefficients represent the array's shape. For a subgroup, the centroid
    (mean) of the polynomial coefficients is computed, and the average Euclidean
    distance between each instance's coefficient vector and the centroid is calculated.
    This distance is then converted to a similarity measure via:
    
        similarity = 1 / (1 + average_distance)

    A penalty is applied if the subgroup size is below a specified minimum.

    Parameters:
       degree: Degree of the polynomial to fit (default is 2).
       fixed_length: Length to which arrays are resampled.
       min_size_sg: Minimum subgroup size; subgroups smaller than this incur a quality penalty.
    """

    tpl = namedtuple("ArrayCurveFitQF_tpl", ["avg_poly_sim", "quality", "size_sg"])

    def __init__(self, degree=2, fixed_length=10, min_size_sg=5):
        self.degree = degree
        self.fixed_length = fixed_length
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
        """
        Computes dataset-level statistics by fitting a polynomial to each array,
        then calculating the centroid of the coefficient vectors and the average
        distance to this centroid. The distance is converted to a similarity.
        """
        all_target_series = data[target.target_variable]
        # Resample arrays to fixed length.
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        # Compute polynomial coefficients for each instance.
        coeffs_list = [poly_fit_coeffs(arr, self.degree, self.fixed_length) for arr in resampled_all]
        coeffs_all = np.vstack(coeffs_list)  # shape: (n_instances, degree+1)
        centroid_coeffs = np.mean(coeffs_all, axis=0)
        # Compute Euclidean distances from each coefficient vector to the centroid.
        distances = np.linalg.norm(coeffs_all - centroid_coeffs, axis=1)
        avg_distance = np.mean(distances)
        # Convert distance to similarity (smaller distance => higher similarity)
        avg_poly_sim_dataset = 1.0 / (1.0 + avg_distance)
        size_dataset = len(data)
        self.dataset_statistics = self.tpl(avg_poly_sim_dataset, None, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        """
        Calculates subgroup-level statistics using curve fitting. Returns a namedtuple
        containing:
          - avg_poly_sim: the average similarity of the fitted polynomial coefficients
                          in the subgroup (1/(1+avg_distance)).
          - quality: the quality score (here, equal to avg_poly_sim with a penalty if subgroup is too small).
          - size_sg: the number of instances in the subgroup.
        """
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_series = data[target.target_variable]
        resampled = all_target_series.apply(lambda x: resample_array(np.array(x), self.fixed_length))
        resampled_all = np.vstack(resampled.to_numpy())
        subgroup_values = resampled_all[cover_arr]
        x = np.linspace(0, 1, self.fixed_length)
        if subgroup_values.shape[0] > 0:
            # Fit a polynomial to each subgroup instance.
            coeffs_list = [poly_fit_coeffs(arr, self.degree, self.fixed_length) for arr in subgroup_values]
            coeffs_all = np.vstack(coeffs_list)
            # Compute the centroid (mean) of the coefficient vectors.
            centroid_coeffs = np.mean(coeffs_all, axis=0)
            # Calculate the Euclidean distance for each coefficient vector.
            distances = np.linalg.norm(coeffs_all - centroid_coeffs, axis=1)
            avg_distance = np.mean(distances)
            avg_poly_sim = 1.0 / (1.0 + avg_distance)
        else:
            avg_poly_sim = 0.0
        quality = avg_poly_sim
        # Apply penalty if subgroup is smaller than minimum size.
        if size_sg < self.min_size_sg - 10:
            size_deviation = (self.min_size_sg - size_sg) / self.min_size_sg
            quality = quality - size_deviation
        return self.tpl(avg_poly_sim, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth or similar frameworks ---

    def gp_get_stats(self, row_index):
        # For gp_growth, we assume a row corresponds to a candidate selector.
        return np.zeros(3)

    def gp_get_null_vector(self):
        return np.zeros(3)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        return self.tpl(v[0], v[1], v[2])

    def gp_to_str(self, stats):
        return f"poly_sim: {stats.avg_poly_sim:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name, None)