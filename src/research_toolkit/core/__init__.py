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
    get_symbol,
    format_temperature,
    format_pm25,
    format_concentration,
    format_percentage
)
from .statistics import StatisticalFormatter

__all__ = [
    # Output
    'SafeOutput',
    
    # Formatting
    'ScientificNotation',
    'ReportFormatter',
    'get_symbol',
    'format_temperature',
    'format_pm25',
    'format_concentration',
    'format_percentage',
    
    # Statistics
    'StatisticalFormatter',
]
