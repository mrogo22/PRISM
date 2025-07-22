"""
correlation_target.py

This module introduces a new target type, CorrelationTarget, which handles a single target
variable whose values are numerical. This is implemented to serve as a target type for the correlation target. 
The LowVarianceQF, in particular, has been implemented to handle the correlation target. It serves the purpose of
identifying subgroups where items have similar precomputed correlation values.
 
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
# CorrelationTarget
# -------------------------------

@total_ordering
class CorrelationTarget(BaseTarget):

    # Define the names for statistics we compute
    statistic_types = (
        "size_sg",
        "size_dataset",
        "mean_sg",
        "mean_dataset",
        "std_sg",
        "std_dataset",
        "median_sg",
        "median_dataset",
        "max_sg",
        "max_dataset",
        "min_sg",
        "min_dataset",
        "mean_lift",
        "median_lift",
        "size_cover_all",
        "covered_not_in_sg",
    )

    def __init__(self, target_variable, initial_data=None):
        self.target_variable = target_variable
        self.initial_data = initial_data

    def __repr__(self):
        return "NumberTarget: " + str(self.target_variable)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [self.target_variable]

    def get_base_statistics(self, subgroup, data):
        # Get subgroup cover array and its size
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[self.target_variable].to_numpy()
    

        cover_all = subgroup.covers(self.initial_data)
        size_cover_all = np.count_nonzero(cover_all)
        covered_not_in_sg = size_cover_all - size_sg

        sg_target_values = all_target_values[cover_arr]

        instances_dataset = len(data)
        instances_subgroup = size_sg
        mean_sg = np.mean(sg_target_values)
        mean_dataset = np.mean(all_target_values)
        std_sg = np.std(sg_target_values)
        std_dataset = np.std(all_target_values)
        median_sg = np.median(sg_target_values)
        median_dataset = np.median(all_target_values)
        max_sg = np.max(sg_target_values)
        max_dataset = np.max(all_target_values)
        min_sg = np.min(sg_target_values)
        min_dataset = np.min(all_target_values)
        if mean_dataset != 0:
            mean_lift = mean_sg / mean_dataset
        else:
            mean_lift = None

        if median_dataset != 0:
            median_lift = median_sg / median_dataset
        else:
            median_lift = None

        return (instances_subgroup, instances_dataset, mean_sg, mean_dataset, std_sg, std_dataset, median_sg, median_dataset, max_sg, max_dataset, min_sg, min_dataset, mean_lift, median_lift, size_cover_all,
                covered_not_in_sg)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        stats = {}
        (instances_subgroup, instances_dataset, mean_sg, mean_dataset, std_sg, std_dataset, median_sg, median_dataset, max_sg, max_dataset, min_sg, min_dataset, mean_lift, median_lift, size_cover_all,
                covered_not_in_sg) = self.get_base_statistics(subgroup, data)
        stats["size_sg"] = instances_subgroup
        stats["size_dataset"] = instances_dataset
        stats["mean_sg"] = mean_sg
        stats["mean_dataset"] = mean_dataset
        stats["std_sg"] = std_sg
        stats["std_dataset"] = std_dataset
        stats["median_sg"] = median_sg
        stats["median_dataset"] = median_dataset
        stats["max_sg"] = max_sg
        stats["max_dataset"] = max_dataset
        stats["min_sg"] = min_sg
        stats["min_dataset"] = min_dataset
        stats["mean_lift"] = mean_lift
        stats["median_lift"] = median_lift
        stats["size_cover_all"] = size_cover_all
        stats["covered_not_in_sg"]= covered_not_in_sg

        return stats

# -------------------------------
# StandardNumberQF
# -------------------------------

class StandardNumberQF(AbstractInterestingnessMeasure):

    tpl = namedtuple("StandardNumberQF_tpl", ["mean_sg", "mean_dataset", "quality", "size_sg"])

    def __init__(self, a=1, min_size_sg=5, initial_data=None):
        self.a = a
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None
        self.initial_data = initial_data

    def calculate_constant_statistics(self, data, target):
        # Compute constant (dataset-level) statistics for the target arrays.
        all_target_values = data[target.target_variable].to_numpy()
        mean_dataset = np.mean(all_target_values)
        size_dataset = len(data)
        self.dataset_statistics = self.tpl(0, mean_dataset, 0, size_dataset)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[target.target_variable].to_numpy()
        cover_all = subgroup.covers(self.initial_data)
        size_cover_all = np.count_nonzero(cover_all)
        
        covered_not_in_sg = size_cover_all - size_sg  
        sg_target_values = all_target_values[cover_arr]
        instances_dataset = len(data)
        mean_dataset = np.mean(all_target_values)
        median_dataset = np.median(all_target_values)

        if sg_target_values.shape[0] > 0:
            mean_sg = np.mean(sg_target_values)
            median_sg = np.median(sg_target_values)
        else:
            mean_sg = 0.0
            median_sg = 0.0

        if size_sg < self.min_size_sg - 10:
            size_deviation = (self.min_size_sg - size_sg) / self.min_size_sg
            quality = size_sg**self.a * (mean_sg - mean_dataset) - size_deviation
        else:
            quality = size_sg**self.a * (mean_sg - mean_dataset)
        
        return self.tpl(mean_sg, mean_dataset, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        return np.zeros(4)

    def gp_get_null_vector(self):
        return np.zeros(4)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        return self.tpl(v[0], v[1], v[2], v[3])

    # FIXED: Use the appropriate fields for numeric target
    def gp_to_str(self, stats):
        return f"mean_sg: {stats.mean_sg:.3f}, mean_dataset: {stats.mean_dataset:.3f}, quality: {stats.quality:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)



# -------------------------------
# LowVarianceQF
# -------------------------------

class LowVarianceQF(AbstractInterestingnessMeasure):
    tpl = namedtuple("LowVarianceQF_tpl", ["mean_sg", "std_sg", "quality", "size_sg"])

    def __init__(self, a=1.0, min_size_sg=5, initial_data=None):
        self.a = a  # weight for size importance
        self.min_size_sg = min_size_sg
        self.has_constant_statistics = True
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None  # not needed here
        self.initial_data = initial_data

    def calculate_constant_statistics(self, data, target):
        # Not needed, but kept for compatibility
        pass

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        max_mad=1.0
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[target.target_variable].to_numpy()
        cover_all = subgroup.covers(self.initial_data)
        size_cover_all = np.count_nonzero(cover_all)
        
        covered_not_in_sg = size_cover_all - size_sg  

        sg_target_values = all_target_values[cover_arr]
        mean_sg = np.mean(sg_target_values)
        mad_sg = np.mean(np.abs(sg_target_values - mean_sg))

        # 1. Tightness score (0 = max deviation, 1 = perfect consistency)
        tightness_score = 1 - min(mad_sg / max_mad, 1.0)  # clamp to [0, 1]

        

        #2.  Outlier penalty
        std_sg = np.std(sg_target_values) + 1e-8
        z_scores = np.abs(sg_target_values - mean_sg) / std_sg
        n_outliers = np.sum(z_scores > 2.5)
        robustness = 1 - (n_outliers / size_sg)

        # Final quality is the product of three terms in [0, 1]
        
        quality = tightness_score * robustness
        
        if size_sg < self.min_size_sg:
            quality = 0
        
        if covered_not_in_sg > 0:
            quality = 0

        return self.tpl(mean_sg, std_sg, quality, size_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    # --- Methods to support gp_growth ---
    def gp_get_stats(self, row_index):
        return np.zeros(4)

    def gp_get_null_vector(self):
        return np.zeros(4)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, cover_arr, v):
        return self.tpl(v[0], v[1], v[2], v[3])

    # FIXED: Use the appropriate fields for numeric target
    def gp_to_str(self, stats):
        return f"mean_sg: {stats.mean_sg:.3f}, mean_dataset: {stats.mean_dataset:.3f}, quality: {stats.quality:.3f}"

    def gp_size_sg(self, stats):
        return stats.size_sg

    @property
    def gp_requires_cover_arr(self):
        return True

    def __getattr__(self, name):
        return getattr(self.dataset_statistics, name)
