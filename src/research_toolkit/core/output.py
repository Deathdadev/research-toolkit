"""
Core output utilities for encoding-safe console and file operations.

This module provides safe output mechanisms that automatically handle
Unicode/ASCII conversion based on console capabilities.
"""

import sys
from contextlib import contextmanager
from typing import Optional


class SafeOutput:
    """
    Provides encoding-safe output for console and file operations.
    Automatically detects encoding capabilities and uses appropriate fallbacks.
    """
    
    # ASCII fallbacks for Unicode characters
    ASCII_FALLBACKS = {
        # Greek letters (lowercase)
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
        'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
        'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
        # Greek letters (uppercase)
        'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta',
        'Ε': 'Epsilon', 'Ζ': 'Zeta', 'Η': 'Eta', 'Θ': 'Theta',
        'Ι': 'Iota', 'Κ': 'Kappa', 'Λ': 'Lambda', 'Μ': 'Mu',
        'Ν': 'Nu', 'Ξ': 'Xi', 'Ο': 'Omicron', 'Π': 'Pi',
        'Ρ': 'Rho', 'Σ': 'Sigma', 'Τ': 'Tau', 'Υ': 'Upsilon',
        'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega',
        # Mathematical operators
        '×': 'x', '÷': '/', '−': '-', '±': '+/-', '∓': '-/+',
        '√': 'sqrt', '∞': 'inf', '∂': 'd', '∇': 'nabla',
        '∫': 'integral', '∑': 'sum', '∏': 'product',
        '∝': 'proportional_to', '≈': '~=', '≡': '===', '≠': '!=',
        '≤': '<=', '≥': '>=', '≪': '<<', '≫': '>>',
        '⊂': 'subset', '⊃': 'superset', '∈': 'in', '∉': 'not_in',
        '∩': 'intersect', '∪': 'union', '∅': 'empty',
        '∀': 'forall', '∃': 'exists', '∄': 'not_exists',
        # Superscripts and subscripts
        '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
        '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        '⁺': '^+', '⁻': '^-', '⁼': '^=', '⁽': '^(', '⁾': '^)',
        'ⁿ': '^n', 'ⁱ': '^i',
        '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4',
        '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
        '₊': '_+', '₋': '_-', '₌': '_=', '₍': '_(', '₎': '_)',
        # Special symbols
        '°': 'deg', '℃': 'degC', '℉': 'degF',
        '✓': '[OK]', '✗': '[X]', '•': '*',
        '→': '->', '←': '<-', '↑': '^', '↓': 'v', '↔': '<->',
        '©': '(c)', '®': '(R)', '™': '(TM)',
        '§': 'S', '¶': 'P', '†': '+', '‡': '++',
    }
    
    @staticmethod
    def can_encode_unicode() -> bool:
        """
        Check if stdout can encode Unicode characters.
        
        Returns:
            True if Unicode is supported, False otherwise
        """
        try:
            test_string = "α β μ ° ²"
            encoding = sys.stdout.encoding or 'utf-8'
            test_string.encode(encoding)
            return True
        except (UnicodeEncodeError, AttributeError, LookupError):
            return False
    
    @staticmethod
    def safe_print(text: str, fallback_text: Optional[str] = None, end: str = '\n', file=None):
        """
        Print text with automatic ASCII fallback if Unicode fails.
        
        Args:
            text: Text to print (may contain Unicode)
            fallback_text: Optional pre-computed ASCII fallback
            end: Line ending (default newline)
            file: File object (default stdout)
        """
        output = file or sys.stdout
        
        try:
            print(text, end=end, file=output)
        except UnicodeEncodeError:
            if fallback_text:
                print(fallback_text, end=end, file=output)
            else:
                ascii_text = SafeOutput.to_ascii(text)
                print(ascii_text, end=end, file=output)
    
    @staticmethod
    def to_ascii(text: str) -> str:
        """
        Convert Unicode text to ASCII using fallbacks.
        
        Args:
            text: Text with potential Unicode characters
            
        Returns:
            ASCII-safe text
        """
        result = text
        for unicode_char, ascii_char in SafeOutput.ASCII_FALLBACKS.items():
            result = result.replace(unicode_char, ascii_char)
        return result
    
    @staticmethod
    @contextmanager
    def safe_file_output(filename: str, mode: str = 'w', encoding: str = 'utf-8'):
        """
        Context manager for safe file output with proper encoding.
        
        Args:
            filename: Output file path
            mode: File mode (default 'w')
            encoding: File encoding (default 'utf-8')
            
        Yields:
            File handle
            
        Example:
            with SafeOutput.safe_file_output('report.txt') as f:
                f.write("Temperature: 25 °C\\n")
        """
        with open(filename, mode, encoding=encoding, errors='replace') as f:
            yield f
    
    @staticmethod
    def format_for_output(text: str, target: str = 'console') -> str:
        """
        Format text for specific output target.
        
        Args:
            text: Text to format
            target: 'console' (ASCII fallback) or 'file' (preserve Unicode)
            
        Returns:
            Formatted text
        """
        if target == 'console' and not SafeOutput.can_encode_unicode():
            return SafeOutput.to_ascii(text)
        return text


__all__ = ['SafeOutput']
