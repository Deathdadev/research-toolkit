"""
Scientific notation and report formatting utilities.

This module provides tools for formatting scientific notation with proper
Greek symbols and generating professional research reports.
"""

from typing import Dict, Optional
from .output import SafeOutput


class ScientificNotation:
    """
    Manages scientific notation with proper Greek symbols for reports
    and ASCII fallbacks for console output.
    """
    
    # Greek symbol mappings
    GREEK_SYMBOLS = {
        'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
        'epsilon': 'ε', 'mu': 'μ', 'sigma': 'σ', 'rho': 'ρ',
        'tau': 'τ', 'chi': 'χ', 'phi': 'φ', 'omega': 'ω'
    }
    
    # Special symbols
    SPECIAL_SYMBOLS = {
        'degree': '°', 'squared': '²', 'cubed': '³',
        'plusminus': '±', 'leq': '≤', 'geq': '≥', 'neq': '≠',
        'checkmark': '✓', 'cross': '❌'
    }
    
    @classmethod
    def get_symbol(cls, name: str, use_unicode: bool = True) -> str:
        """
        Get a scientific symbol.
        
        Args:
            name: Symbol name (e.g., 'alpha', 'mu', 'degree')
            use_unicode: If True, return Unicode; if False, return ASCII
            
        Returns:
            The symbol in requested format
        """
        # Check Greek symbols
        if name in cls.GREEK_SYMBOLS:
            symbol = cls.GREEK_SYMBOLS[name]
            return symbol if use_unicode else SafeOutput.ASCII_FALLBACKS.get(symbol, name)
        
        # Check special symbols
        if name in cls.SPECIAL_SYMBOLS:
            symbol = cls.SPECIAL_SYMBOLS[name]
            return symbol if use_unicode else SafeOutput.ASCII_FALLBACKS.get(symbol, name)
        
        return name
    
    @classmethod
    def format_units(cls, value: float, unit: str, use_unicode: bool = True, decimals: int = 2) -> str:
        """
        Format a value with scientific units.
        
        Args:
            value: Numerical value
            unit: Unit string (can contain symbol names in {})
            use_unicode: If True, use Unicode symbols
            decimals: Number of decimal places
            
        Returns:
            Formatted string
            
        Example:
            format_units(25.5, "{mu}g/m{cubed}", True) -> "25.50 μg/m³"
            format_units(25.5, "{mu}g/m{cubed}", False) -> "25.50 ug/m^3"
        """
        # Replace symbol placeholders
        formatted_unit = unit
        all_symbols = {**cls.GREEK_SYMBOLS, **cls.SPECIAL_SYMBOLS}
        
        for name in all_symbols.keys():
            placeholder = f"{{{name}}}"
            if placeholder in formatted_unit:
                symbol = cls.get_symbol(name, use_unicode)
                formatted_unit = formatted_unit.replace(placeholder, symbol)
        
        return f"{value:.{decimals}f} {formatted_unit}"


class ReportFormatter:
    """
    Formats research reports with proper scientific notation.
    Generates both console-friendly and publication-ready versions.
    """
    
    def __init__(self, use_unicode_console: Optional[bool] = None):
        """
        Initialize report formatter.
        
        Args:
            use_unicode_console: Force Unicode on/off for console (None = auto-detect)
        """
        if use_unicode_console is None:
            self.use_unicode_console = SafeOutput.can_encode_unicode()
        else:
            self.use_unicode_console = use_unicode_console
    
    def print_section(self, title: str, width: int = 70):
        """
        Print a section header.
        
        Args:
            title: Section title
            width: Width of the divider line
        """
        SafeOutput.safe_print("\n" + "=" * width)
        SafeOutput.safe_print(title)
        SafeOutput.safe_print("=" * width)
    
    def print_subsection(self, title: str):
        """
        Print a subsection header.
        
        Args:
            title: Subsection title
        """
        SafeOutput.safe_print(f"\n{title}")
    
    def print_statistical_result(
        self, 
        statistic: str, 
        value: float, 
        decimals: int = 3,
        use_greek: bool = True
    ):
        """
        Print a statistical result with proper notation.
        
        Args:
            statistic: Statistic name ('alpha', 'beta', 'r', 't', etc.)
            value: Statistic value
            decimals: Number of decimal places
            use_greek: Use Greek symbols if True
        """
        if use_greek and statistic in ScientificNotation.GREEK_SYMBOLS:
            symbol = ScientificNotation.get_symbol(statistic, self.use_unicode_console)
        else:
            symbol = statistic
        
        text = f"{symbol} = {value:.{decimals}f}"
        SafeOutput.safe_print(text)
    
    def save_report(
        self, 
        filename: str, 
        content: str, 
        use_unicode: bool = True
    ):
        """
        Save report to file with proper encoding.
        
        Args:
            filename: Output filename
            content: Report content
            use_unicode: Preserve Unicode symbols in file
        """
        encoding = 'utf-8' if use_unicode else 'ascii'
        
        if not use_unicode:
            content = SafeOutput.to_ascii(content)
        
        with SafeOutput.safe_file_output(filename, 'w', encoding) as f:
            f.write(content)


# Convenience functions
def format_temperature(temp: float, use_unicode: bool = True, decimals: int = 2) -> str:
    """Format temperature with degree symbol."""
    symbol = ScientificNotation.get_symbol('degree', use_unicode)
    return f"{temp:.{decimals}f}{symbol}C"


def format_pm25(value: float, use_unicode: bool = True, decimals: int = 1) -> str:
    """Format PM2.5 concentration."""
    return ScientificNotation.format_units(
        value, 
        "{mu}g/m{cubed}",
        use_unicode,
        decimals
    )


__all__ = [
    'ScientificNotation',
    'ReportFormatter',
    'format_temperature',
    'format_pm25'
]
