"""
Statistical result formatting utilities.

This module provides standardized formatting for statistical results
following APA 7 guidelines.
"""



class StatisticalFormatter:
    """
    Formats statistical results with proper notation and APA 7 style.
    """

    @staticmethod
    def format_p_value(p: float, threshold: float = 0.001, decimals: int = 3) -> str:
        """
        Format p-value with appropriate precision (APA 7 style).
        
        Args:
            p: p-value
            threshold: Threshold for "< threshold" notation
            decimals: Number of decimal places for regular p-values
            
        Returns:
            Formatted p-value string
            
        Example:
            >>> StatisticalFormatter.format_p_value(0.0234)
            'p = .023'
            >>> StatisticalFormatter.format_p_value(0.0001)
            'p < .001'
        """
        if p < threshold:
            # Format threshold without leading zero
            threshold_str = f"{threshold:.{decimals}f}"
            if threshold < 1:
                threshold_str = threshold_str[1:]  # Remove '0.'
            return f"p < {threshold_str}"
        elif p < 1:
            # APA style: omit leading zero for p-values
            p_str = f"{p:.{decimals}f}"
            if p_str.startswith('0.'):
                p_str = p_str[1:]  # Remove leading '0'
            return f"p = {p_str}"
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


    @staticmethod
    def format_regression(
        r_squared: float,
        adj_r_squared: float = None,
        f_stat: float = None,
        df1: int = None,
        df2: int = None,
        p: float = None,
        decimals: int = 3
    ) -> str:
        """
        Format regression results.
        
        Args:
            r_squared: R-squared value
            adj_r_squared: Optional adjusted R-squared
            f_stat: Optional F-statistic
            df1: Optional numerator degrees of freedom
            df2: Optional denominator degrees of freedom
            p: Optional p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted regression string
            
        Example:
            >>> StatisticalFormatter.format_regression(0.456, 0.442, 25.6, 2, 47, 0.001)
            'R² = .456 (adj. R² = .442), F(2, 47) = 25.600, p < .001'
        """
        # Format R-squared (without leading zero)
        r2_str = f"{r_squared:.{decimals}f}"
        if r2_str.startswith('0.'):
            r2_str = r2_str[1:]

        result = f"R² = {r2_str}"

        if adj_r_squared is not None:
            adj_str = f"{adj_r_squared:.{decimals}f}"
            if adj_str.startswith('0.'):
                adj_str = adj_str[1:]
            result += f" (adj. R² = {adj_str})"

        if f_stat is not None and df1 is not None and df2 is not None:
            result += f", {StatisticalFormatter.format_f_test(f_stat, df1, df2, p or 0, decimals)}"

        return result

    @staticmethod
    def format_anova_oneway(
        f: float,
        df_between: int,
        df_within: int,
        p: float,
        decimals: int = 2
    ) -> str:
        """
        Format one-way ANOVA result.
        
        Args:
            f: F-statistic
            df_between: Between-groups degrees of freedom
            df_within: Within-groups degrees of freedom
            p: p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted ANOVA string
        """
        return StatisticalFormatter.format_f_test(f, df_between, df_within, p, decimals)

    @staticmethod
    def format_mann_whitney(
        u: float,
        n1: int,
        n2: int,
        p: float,
        decimals: int = 2
    ) -> str:
        """
        Format Mann-Whitney U test result.
        
        Args:
            u: U-statistic
            n1: Sample size 1
            n2: Sample size 2
            p: p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted Mann-Whitney string
            
        Example:
            >>> StatisticalFormatter.format_mann_whitney(145.5, 20, 20, 0.032)
            'U = 145.50 (n₁ = 20, n₂ = 20), p = .032'
        """
        return f"U = {u:.{decimals}f} (n₁ = {n1}, n₂ = {n2}), {StatisticalFormatter.format_p_value(p)}"

    @staticmethod
    def format_wilcoxon(
        w: float,
        n: int,
        p: float,
        decimals: int = 2
    ) -> str:
        """
        Format Wilcoxon signed-rank test result.
        
        Args:
            w: W-statistic
            n: Sample size
            p: p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted Wilcoxon string
            
        Example:
            >>> StatisticalFormatter.format_wilcoxon(234.5, 30, 0.012)
            'W = 234.50 (n = 30), p = .012'
        """
        return f"W = {w:.{decimals}f} (n = {n}), {StatisticalFormatter.format_p_value(p)}"

    @staticmethod
    def format_kruskal_wallis(
        h: float,
        df: int,
        p: float,
        decimals: int = 2
    ) -> str:
        """
        Format Kruskal-Wallis H test result.
        
        Args:
            h: H-statistic
            df: Degrees of freedom
            p: p-value
            decimals: Number of decimal places
            
        Returns:
            Formatted Kruskal-Wallis string
            
        Example:
            >>> StatisticalFormatter.format_kruskal_wallis(12.45, 2, 0.002)
            'H(2) = 12.45, p = .002'
        """
        return f"H({df}) = {h:.{decimals}f}, {StatisticalFormatter.format_p_value(p)}"


__all__ = ['StatisticalFormatter']
