"""
Statistical result formatting utilities.

This module provides standardized formatting for statistical results
following APA 7 guidelines.
"""

from typing import Tuple


class StatisticalFormatter:
    """
    Formats statistical results with proper notation and APA 7 style.
    """
    
    @staticmethod
    def format_p_value(p: float, threshold: float = 0.001, decimals: int = 3) -> str:
        """
        Format p-value with appropriate precision.
        
        Args:
            p: p-value
            threshold: Threshold for "< threshold" notation
            decimals: Number of decimal places for regular p-values
            
        Returns:
            Formatted p-value string
            
        Example:
            >>> format_p_value(0.0234)
            'p = .023'
            >>> format_p_value(0.0001)
            'p < .001'
        """
        if p < threshold:
            return f"p < {threshold:.3f}"
        elif p < 1:
            # APA style: omit leading zero for p-values
            return f"p = {p:.{decimals}f}"[0:2] + f"p = {p:.{decimals}f}"[3:]
        else:
            return "p = 1.000"
    
    @staticmethod
    def format_ci(
        lower: float, 
        upper: float, 
        decimals: int = 2,
        percentage: bool = False,
        ci_level: int = 95
    ) -> str:
        """
        Format confidence interval.
        
        Args:
            lower: Lower bound
            upper: Upper bound
            decimals: Number of decimal places
            percentage: If True, format as percentage
            ci_level: Confidence level (default 95)
            
        Returns:
            Formatted CI string
            
        Example:
            >>> format_ci(1.23, 4.56)
            '95% CI [1.23, 4.56]'
        """
        if percentage:
            return f"{ci_level}% CI [{lower:.{decimals}f}%, {upper:.{decimals}f}%]"
        return f"{ci_level}% CI [{lower:.{decimals}f}, {upper:.{decimals}f}]"
    
    @staticmethod
    def format_mean_sd(mean: float, sd: float, decimals: int = 2, n: int = None) -> str:
        """
        Format mean with standard deviation.
        
        Args:
            mean: Mean value
            sd: Standard deviation
            decimals: Number of decimal places
            n: Optional sample size
            
        Returns:
            Formatted string "M = X.XX, SD = Y.YY" or "M = X.XX, SD = Y.YY, n = Z"
            
        Example:
            >>> format_mean_sd(10.5, 2.3)
            'M = 10.50, SD = 2.30'
        """
        result = f"M = {mean:.{decimals}f}, SD = {sd:.{decimals}f}"
        if n is not None:
            result += f", n = {n}"
        return result
    
    @staticmethod
    def format_correlation(r: float, p: float, n: int = None, decimals: int = 2) -> str:
        """
        Format correlation result.
        
        Args:
            r: Correlation coefficient
            p: p-value
            n: Optional sample size
            decimals: Number of decimal places
            
        Returns:
            Formatted correlation string
            
        Example:
            >>> format_correlation(0.456, 0.012, 100)
            'r(98) = .46, p = .012'
        """
        # Remove leading zero from r
        r_str = f"{r:.{decimals}f}"
        if r >= 0:
            r_str = r_str[1:]  # Remove "0."
        else:
            r_str = "-" + r_str[2:]  # Keep minus, remove "0."
        
        if n is not None:
            df = n - 2
            result = f"r({df}) = {r_str}"
        else:
            result = f"r = {r_str}"
        
        result += f", {StatisticalFormatter.format_p_value(p)}"
        return result
    
    @staticmethod
    def format_t_test(t: float, df: int, p: float, decimals: int = 2) -> str:
        """
        Format t-test result.
        
        Args:
            t: t-statistic
            df: Degrees of freedom
            p: p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted t-test string
            
        Example:
            >>> format_t_test(2.34, 48, 0.023)
            't(48) = 2.34, p = .023'
        """
        return f"t({df}) = {t:.{decimals}f}, {StatisticalFormatter.format_p_value(p)}"
    
    @staticmethod
    def format_f_test(f: float, df1: int, df2: int, p: float, decimals: int = 2) -> str:
        """
        Format F-test result.
        
        Args:
            f: F-statistic
            df1: Numerator degrees of freedom
            df2: Denominator degrees of freedom
            p: p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted F-test string
            
        Example:
            >>> format_f_test(3.45, 2, 47, 0.041)
            'F(2, 47) = 3.45, p = .041'
        """
        return f"F({df1}, {df2}) = {f:.{decimals}f}, {StatisticalFormatter.format_p_value(p)}"
    
    @staticmethod
    def format_chi_square(chi2: float, df: int, p: float, decimals: int = 2) -> str:
        """
        Format chi-square test result.
        
        Args:
            chi2: Chi-square statistic
            df: Degrees of freedom
            p: p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted chi-square string
            
        Example:
            >>> format_chi_square(5.67, 2, 0.059)
            'χ²(2) = 5.67, p = .059'
        """
        return f"χ²({df}) = {chi2:.{decimals}f}, {StatisticalFormatter.format_p_value(p)}"
    
    @staticmethod
    def interpret_effect_size(effect_size: float, measure: str = 'cohens_d') -> str:
        """
        Interpret effect size magnitude.
        
        Args:
            effect_size: Effect size value
            measure: Type of effect size measure ('cohens_d', 'correlation', 'eta_squared')
            
        Returns:
            Interpretation string ('negligible', 'small', 'medium', 'large')
            
        Example:
            >>> interpret_effect_size(0.6, 'cohens_d')
            'medium'
        """
        abs_effect = abs(effect_size)
        
        if measure == 'cohens_d':
            if abs_effect < 0.2:
                return 'negligible'
            elif abs_effect < 0.5:
                return 'small'
            elif abs_effect < 0.8:
                return 'medium'
            else:
                return 'large'
        
        elif measure == 'correlation':
            if abs_effect < 0.1:
                return 'negligible'
            elif abs_effect < 0.3:
                return 'small'
            elif abs_effect < 0.5:
                return 'medium'
            else:
                return 'large'
        
        elif measure == 'eta_squared':
            if abs_effect < 0.01:
                return 'negligible'
            elif abs_effect < 0.06:
                return 'small'
            elif abs_effect < 0.14:
                return 'medium'
            else:
                return 'large'
        
        return 'unknown'
    
    @staticmethod
    def format_effect_size(
        effect_size: float,
        measure: str = 'cohens_d',
        decimals: int = 2,
        include_interpretation: bool = True
    ) -> str:
        """
        Format effect size with optional interpretation.
        
        Args:
            effect_size: Effect size value
            measure: Type of effect size ('cohens_d', 'r', 'eta_squared')
            decimals: Number of decimal places
            include_interpretation: Include interpretation in output
            
        Returns:
            Formatted effect size string
            
        Example:
            >>> format_effect_size(0.6, 'cohens_d')
            'd = 0.60 (medium effect)'
        """
        # Symbol mapping
        symbols = {
            'cohens_d': 'd',
            'correlation': 'r',
            'eta_squared': 'η²',
            'partial_eta_squared': 'ηp²',
            'omega_squared': 'ω²'
        }
        
        symbol = symbols.get(measure, measure)
        result = f"{symbol} = {effect_size:.{decimals}f}"
        
        if include_interpretation:
            interp = StatisticalFormatter.interpret_effect_size(effect_size, measure)
            result += f" ({interp} effect)"
        
        return result


__all__ = ['StatisticalFormatter']
