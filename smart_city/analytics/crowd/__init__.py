# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Crowd analytics sub-package."""
from smart_city.analytics.crowd.density_analyzer import (
    CrowdDensityAnalyzer,
    CrowdDensityResult,
)

__all__ = ["CrowdDensityAnalyzer", "CrowdDensityResult"]
