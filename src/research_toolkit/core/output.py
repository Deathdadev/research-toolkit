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
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'μ': 'mu', 'σ': 'sigma', 'ρ': 'rho',
        'τ': 'tau', 'χ': 'chi', 'φ': 'phi', 'ω': 'omega',
        '°': 'deg', '²': '^2', '³': '^3', '±': '+/-',
        '≤': '<=', '≥': '>=', '≠': '!=',
        '✓': '[OK]', '❌': '[X]'
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
