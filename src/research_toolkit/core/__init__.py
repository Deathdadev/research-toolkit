"""
Core utilities for empirical research toolkit.

This module provides fundamental utilities for:
- Encoding-safe output (console and file)
- Scientific notation formatting
- Statistical result formatting
- Report generation
"""

from .output import SafeOutput
from .formatting import (
    ScientificNotation,
    ReportFormatter,
    format_temperature,
    format_pm25
)
from .statistics import StatisticalFormatter

__all__ = [
    # Output
    'SafeOutput',
    
    # Formatting
    'ScientificNotation',
    'ReportFormatter',
    'format_temperature',
    'format_pm25',
    
    # Statistics
    'StatisticalFormatter',
]
