"""
Scientific notation and report formatting utilities.

This module provides tools for formatting scientific notation with proper
Greek symbols and generating professional research reports.
"""

from typing import Dict, Optional, Union
from .output import SafeOutput


class ScientificNotation:
    """
    Manages scientific notation with proper Greek symbols for reports
    and ASCII fallbacks for console output.
    """
    
    # Greek symbol mappings
    GREEK_SYMBOLS = {
        'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
        'epsilon': 'ε', 'zeta': 'ζ', 'eta': 'η', 'theta': 'θ',
        'iota': 'ι', 'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ',
        'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο', 'pi': 'π',
        'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
        'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
        # Uppercase variants
        'Alpha': 'Α', 'Beta': 'Β', 'Gamma': 'Γ', 'Delta': 'Δ',
        'Epsilon': 'Ε', 'Zeta': 'Ζ', 'Eta': 'Η', 'Theta': 'Θ',
        'Iota': 'Ι', 'Kappa': 'Κ', 'Lambda': 'Λ', 'Mu': 'Μ',
        'Nu': 'Ν', 'Xi': 'Ξ', 'Omicron': 'Ο', 'Pi': 'Π',
        'Rho': 'Ρ', 'Sigma': 'Σ', 'Tau': 'Τ', 'Upsilon': 'Υ',
        'Phi': 'Φ', 'Chi': 'Χ', 'Psi': 'Ψ', 'Omega': 'Ω',
    }
    
    # Mathematical operators and symbols
    MATH_SYMBOLS = {
        'times': '×', 'divide': '÷', 'minus': '−', 'plus': '+',
        'plusminus': '±', 'minusplus': '∓',
        'sqrt': '√', 'infinity': '∞', 'partial': '∂',
        'nabla': '∇', 'integral': '∫', 'sum': '∑', 'product': '∏',
        'proportional': '∝', 'approximately': '≈', 'equivalent': '≡',
        'identical': '≡', 'notequal': '≠', 'neq': '≠',
        'leq': '≤', 'geq': '≥', 'much_less': '≪', 'much_greater': '≫',
        'subset': '⊂', 'superset': '⊃', 'element': '∈', 'not_element': '∉',
        'intersection': '∩', 'union': '∪', 'empty_set': '∅',
        'for_all': '∀', 'exists': '∃', 'not_exists': '∄',
    }
    
    # Superscripts and subscripts
    SUPERSCRIPTS = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'n': 'ⁿ', 'i': 'ⁱ',
    }
    
    SUBSCRIPTS = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
    }
    
    # Special symbols
    SPECIAL_SYMBOLS = {
        'degree': '°', 'celsius': '℃', 'fahrenheit': '℉',
        'squared': '²', 'cubed': '³',
        'checkmark': '✓', 'cross': '✗', 'ballot_x': '✗',
        'bullet': '•', 'arrow_right': '→', 'arrow_left': '←',
        'arrow_up': '↑', 'arrow_down': '↓', 'arrow_double': '↔',
        'copyright': '©', 'registered': '®', 'trademark': '™',
        'section': '§', 'paragraph': '¶', 'dagger': '†', 'double_dagger': '‡',
    }
    
    # Common unit abbreviations
    UNIT_PREFIXES = {
        'yotta': 'Y', 'zetta': 'Z', 'exa': 'E', 'peta': 'P',
        'tera': 'T', 'giga': 'G', 'mega': 'M', 'kilo': 'k',
        'hecto': 'h', 'deca': 'da',
        'deci': 'd', 'centi': 'c', 'milli': 'm', 'micro': 'μ',
        'nano': 'n', 'pico': 'p', 'femto': 'f', 'atto': 'a',
        'zepto': 'z', 'yocto': 'y',
    }
    
    @classmethod
    def get_all_symbols(cls) -> Dict[str, str]:
        """
        Get all available symbols.
        
        Returns:
            Dictionary of all symbol names and their Unicode representations
        """
        return {
            **cls.GREEK_SYMBOLS,
            **cls.MATH_SYMBOLS,
            **cls.SPECIAL_SYMBOLS,
            **cls.UNIT_PREFIXES,
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
            
        Example:
            >>> ScientificNotation.get_symbol('alpha')
            'α'
            >>> ScientificNotation.get_symbol('alpha', use_unicode=False)
            'alpha'
        """
        all_symbols = cls.get_all_symbols()
        
        if name in all_symbols:
            symbol = all_symbols[name]
            return symbol if use_unicode else SafeOutput.ASCII_FALLBACKS.get(symbol, name)
        
        return name
    
    @classmethod
    def format_with_symbol(cls, symbol_name: str, value: Union[float, str], 
                          use_unicode: bool = True, decimals: Optional[int] = None) -> str:
        """
        Format a value with a symbol (e.g., "α = 0.05").
        
        Args:
            symbol_name: Name of the symbol
            value: Value to display
            use_unicode: Use Unicode symbols
            decimals: Number of decimal places (if value is numeric)
            
        Returns:
            Formatted string
            
        Example:
            >>> ScientificNotation.format_with_symbol('alpha', 0.05, decimals=2)
            'α = 0.05'
        """
        symbol = cls.get_symbol(symbol_name, use_unicode)
        
        if isinstance(value, (int, float)) and decimals is not None:
            return f"{symbol} = {value:.{decimals}f}"
        else:
            return f"{symbol} = {value}"
    
    @classmethod
    def format_units(cls, value: float, unit: str, use_unicode: bool = True, 
                    decimals: int = 2, separator: str = " ") -> str:
        """
        Format a value with scientific units.
        
        Args:
            value: Numerical value
            unit: Unit string (can contain symbol names in {})
            use_unicode: If True, use Unicode symbols
            decimals: Number of decimal places
            separator: Separator between value and unit
            
        Returns:
            Formatted string
            
        Example:
            >>> ScientificNotation.format_units(25.5, "{mu}g/m{cubed}")
            '25.50 μg/m³'
            >>> ScientificNotation.format_units(25.5, "{mu}g/m{cubed}", use_unicode=False)
            '25.50 ug/m^3'
        """
        formatted_unit = cls.parse_unit_string(unit, use_unicode)
        return f"{value:.{decimals}f}{separator}{formatted_unit}"
    
    @classmethod
    def parse_unit_string(cls, unit: str, use_unicode: bool = True) -> str:
        """
        Parse a unit string and replace symbol placeholders.
        
        Args:
            unit: Unit string with placeholders like {mu}, {squared}, etc.
            use_unicode: Use Unicode symbols
            
        Returns:
            Parsed unit string
            
        Example:
            >>> ScientificNotation.parse_unit_string("{mu}g/m{cubed}")
            'μg/m³'
        """
        formatted_unit = unit
        all_symbols = cls.get_all_symbols()
        
        for name, symbol in all_symbols.items():
            placeholder = f"{{{name}}}"
            if placeholder in formatted_unit:
                replacement = cls.get_symbol(name, use_unicode)
                formatted_unit = formatted_unit.replace(placeholder, replacement)
        
        return formatted_unit
    
    @classmethod
    def to_superscript(cls, text: str, use_unicode: bool = True) -> str:
        """
        Convert text to superscript notation.
        
        Args:
            text: Text to convert (digits and basic operators)
            use_unicode: Use Unicode superscripts
            
        Returns:
            Superscript text
            
        Example:
            >>> ScientificNotation.to_superscript('23')
            '²³'
            >>> ScientificNotation.to_superscript('23', use_unicode=False)
            '^23'
        """
        if not use_unicode:
            return f"^{text}"
        
        result = ""
        for char in text:
            result += cls.SUPERSCRIPTS.get(char, char)
        return result
    
    @classmethod
    def to_subscript(cls, text: str, use_unicode: bool = True) -> str:
        """
        Convert text to subscript notation.
        
        Args:
            text: Text to convert (digits and basic operators)
            use_unicode: Use Unicode subscripts
            
        Returns:
            Subscript text
            
        Example:
            >>> ScientificNotation.to_subscript('2')
            '₂'
            >>> ScientificNotation.to_subscript('2', use_unicode=False)
            '_2'
        """
        if not use_unicode:
            return f"_{text}"
        
        result = ""
        for char in text:
            result += cls.SUBSCRIPTS.get(char, char)
        return result
    
    @classmethod
    def format_chemical_formula(cls, formula: str, use_unicode: bool = True) -> str:
        """
        Format a chemical formula with proper subscripts.
        
        Args:
            formula: Chemical formula (e.g., "H2O", "CO2", "CaCO3")
            use_unicode: Use Unicode subscripts
            
        Returns:
            Formatted formula
            
        Example:
            >>> ScientificNotation.format_chemical_formula("H2O")
            'H₂O'
            >>> ScientificNotation.format_chemical_formula("CO2", use_unicode=False)
            'CO_2'
        """
        result = ""
        i = 0
        while i < len(formula):
            char = formula[i]
            if char.isdigit():
                result += cls.to_subscript(char, use_unicode)
            else:
                result += char
            i += 1
        return result


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
    
    def format_header(self, title: str, width: int = 70, char: str = '=') -> str:
        """
        Format a section header.
        
        Args:
            title: Section title
            width: Width of the divider line
            char: Character for the divider
            
        Returns:
            Formatted header string
        """
        return f"\n{char * width}\n{title}\n{char * width}"
    
    def format_subheader(self, title: str, char: str = '-') -> str:
        """
        Format a subsection header.
        
        Args:
            title: Subsection title
            char: Character for underline
            
        Returns:
            Formatted subheader string
        """
        return f"\n{title}\n{char * len(title)}"
    
    def print_section(self, title: str, width: int = 70, char: str = '='):
        """
        Print a section header.
        
        Args:
            title: Section title
            width: Width of the divider line
            char: Character for the divider
        """
        header = self.format_header(title, width, char)
        SafeOutput.safe_print(header)
    
    def print_subsection(self, title: str, char: str = '-'):
        """
        Print a subsection header.
        
        Args:
            title: Subsection title
            char: Character for underline
        """
        subheader = self.format_subheader(title, char)
        SafeOutput.safe_print(subheader)
    
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
        text = ScientificNotation.format_with_symbol(
            statistic, value, self.use_unicode_console, decimals
        )
        SafeOutput.safe_print(text)
    
    def format_bullet_list(self, items: list, bullet: str = None, indent: int = 2) -> str:
        """
        Format a bullet list.
        
        Args:
            items: List of items
            bullet: Bullet character (default: auto-detect)
            indent: Indentation spaces
            
        Returns:
            Formatted bullet list
        """
        if bullet is None:
            bullet = ScientificNotation.get_symbol('bullet', self.use_unicode_console)
        
        lines = []
        for item in items:
            lines.append(f"{' ' * indent}{bullet} {item}")
        return '\n'.join(lines)
    
    def format_numbered_list(self, items: list, indent: int = 2) -> str:
        """
        Format a numbered list.
        
        Args:
            items: List of items
            indent: Indentation spaces
            
        Returns:
            Formatted numbered list
        """
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"{' ' * indent}{i}. {item}")
        return '\n'.join(lines)
    
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


# Convenience functions for common use cases
def get_symbol(name: str, use_unicode: bool = True) -> str:
    """
    Get a scientific symbol (convenience wrapper).
    
    Args:
        name: Symbol name (e.g., 'alpha', 'mu', 'degree')
        use_unicode: If True, return Unicode; if False, return ASCII
        
    Returns:
        The symbol in requested format
        
    Example:
        >>> get_symbol('alpha')
        'α'
        >>> get_symbol('alpha', use_unicode=False)
        'alpha'
    """
    return ScientificNotation.get_symbol(name, use_unicode)


def format_temperature(temp: float, use_unicode: bool = True, decimals: int = 2,
                      scale: str = 'C') -> str:
    """
    Format temperature with degree symbol.
    
    Args:
        temp: Temperature value
        use_unicode: Use Unicode symbols
        decimals: Number of decimal places
        scale: Temperature scale ('C', 'F', 'K')
        
    Returns:
        Formatted temperature string
    """
    if scale == 'K':
        return f"{temp:.{decimals}f} K"
    
    degree = ScientificNotation.get_symbol('degree', use_unicode)
    return f"{temp:.{decimals}f}{degree}{scale}"


def format_pm25(value: float, use_unicode: bool = True, decimals: int = 1) -> str:
    """
    Format PM2.5 concentration.
    
    Args:
        value: PM2.5 concentration
        use_unicode: Use Unicode symbols
        decimals: Number of decimal places
        
    Returns:
        Formatted PM2.5 string
    """
    return ScientificNotation.format_units(
        value, 
        "{mu}g/m{cubed}",
        use_unicode,
        decimals
    )


def format_concentration(value: float, unit: str = 'M', use_unicode: bool = True,
                        decimals: int = 2) -> str:
    """
    Format chemical concentration.
    
    Args:
        value: Concentration value
        unit: Concentration unit ('M', 'mM', 'μM', 'nM', etc.)
        use_unicode: Use Unicode symbols
        decimals: Number of decimal places
        
    Returns:
        Formatted concentration string
    """
    parsed_unit = ScientificNotation.parse_unit_string(unit, use_unicode)
    return f"{value:.{decimals}f} {parsed_unit}"


def format_percentage(value: float, decimals: int = 1, include_symbol: bool = True) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage value (0-100)
        decimals: Number of decimal places
        include_symbol: Include % symbol
        
    Returns:
        Formatted percentage string
    """
    if include_symbol:
        return f"{value:.{decimals}f}%"
    return f"{value:.{decimals}f}"


__all__ = [
    'ScientificNotation',
    'ReportFormatter',
    'get_symbol',
    'format_temperature',
    'format_pm25',
    'format_concentration',
    'format_percentage',
]
