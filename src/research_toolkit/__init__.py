"""
Research Toolkit

A comprehensive Python library for conducting research across multiple methodologies
with proper scientific notation, encoding-safe output, APA 7 referencing, and AI integration.

Supports:
- Empirical research (correlational, comparative, time series, observational, meta-analysis)
- Non-empirical research (simulation, methodological, theoretical)
- APA 7 reference management
- Statistical result formatting
- MCP server for AI integration
- Research methodology guidance

Version: 2.0.0
"""

__version__ = '2.0.0'
__author__ = 'Death'
__license__ = 'MIT'

# Import core utilities
from .core import (
    SafeOutput,
    ScientificNotation,
    ReportFormatter,
    StatisticalFormatter,
    get_symbol,
    format_temperature,
    format_pm25,
    format_concentration,
    format_percentage
)

# Import references
from .references import APA7ReferenceManager

# Import research tools
from .research import ResearchToolkitMCPServer, EmpiricalResearchMCPServer

__all__ = [
    # Version info
    '__version__',

    # Core utilities
    'SafeOutput',
    'ScientificNotation',
    'ReportFormatter',
    'StatisticalFormatter',
    'get_symbol',
    'format_temperature',
    'format_pm25',
    'format_concentration',
    'format_percentage',

    # References
    'APA7ReferenceManager',

    # Research tools (MCP Server)
    'ResearchToolkitMCPServer',
    'EmpiricalResearchMCPServer',  # Backward compatibility
]


def get_examples():
    """
    List available research examples.
    
    Returns:
        dict: Dictionary of example names and descriptions
    """
    return {
        '00_synthetic_example': 'Educational example of what NOT to do',
        '01_correlational_study': 'Correlational research (air quality)',
        '02_comparative_study': 'Group comparisons (coastal vs inland)',
        '03_time_series_analysis': 'Temporal patterns (air quality trends)',
        '04_observational_study': 'Descriptive research (GitHub repos)',
        '05_meta_analysis': 'Synthesizing studies (study hours)',
        '06_simulation_study': 'Model-based research (SIR epidemic)',
        '07_methodological_study': 'Testing methods (power analysis)',
        '08_theoretical_model': 'Theory development (information diffusion)'
    }


def get_research_types():
    """
    List all research types covered.
    
    Returns:
        dict: Dictionary of research types and their characteristics
    """
    return {
        'Empirical (Real Data Required)': [
            'Correlational Study',
            'Comparative Study',
            'Time Series Analysis',
            'Observational Study',
            'Meta-Analysis'
        ],
        'Non-Empirical (Conditional/Theoretical)': [
            'Simulation Study',
            'Methodological Study',
            'Theoretical Model'
        ]
    }


def print_info():
    """Print library information."""
    print("="*70)
    print(f"Research Toolkit v{__version__}")
    print("="*70)
    print("\nComprehensive framework for conducting research across all methodologies.")
    print("\nKey Features:")
    print("  [OK] Encoding-safe output (Unicode/ASCII)")
    print("  [OK] Scientific notation formatting")
    print("  [OK] APA 7 reference management")
    print("  [OK] Statistical result formatting")
    print("  [OK] MCP server for AI integration")
    print("  [OK] 9 complete research examples")
    print("  [OK] 8 research types covered")

    print("\nSupported Research Types:")
    types = get_research_types()
    for category, research_types in types.items():
        print(f"\n{category}:")
        for rt in research_types:
            print(f"  - {rt}")

    print("\nQuick Start:")
    print("  from research_toolkit import ReportFormatter, APA7ReferenceManager")
    print("  formatter = ReportFormatter()")
    print("  formatter.print_section('MY RESEARCH')")

    print("\nDocumentation:")
    print("  - Library Guide: docs/LIBRARY_GUIDE.md")
    print("  - MCP Integration: docs/MCP_INTEGRATION_GUIDE.md")
    print("  - Examples: examples/ directory")
    print("="*70)


if __name__ == "__main__":
    print_info()
