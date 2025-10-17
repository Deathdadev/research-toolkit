"""
Example 07: Methodological Study - Statistical Power Analysis

Research Question: What is the statistical power of Pearson correlation test 
under different sample sizes and effect sizes?

This demonstrates:
- Methodological research (testing a method, not studying phenomena)
- APPROPRIATE use of synthetic data (we're testing the method itself)
- Monte Carlo simulation
- Power analysis
- Method comparison
- APA 7 referencing

CRITICAL: This is the ONLY research type where synthetic data is fully appropriate
because we're making claims about the METHOD, not about real-world phenomena.
"""
# Standard library imports
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json

# Third-party imports
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Local imports (research_toolkit)
from research_toolkit import ReportFormatter, SafeOutput, StatisticalFormatter, get_symbol
from research_toolkit.references import APA7ReferenceManager

class PowerAnalysisStudy:
    """
    Methodological study examining statistical power of correlation tests.
    
    Research Type: Methodological Study
    IMPORTANT: This is appropriate use of synthetic data!
    """
    
    def __init__(self) -> None:
        """
        Initialize methodological study.
        
        Note:
            Tests statistical method performance using synthetic data.
        """
        self.references = APA7ReferenceManager()
        
        # Add references
        self.cohen_ref = self.references.add_reference(
            'book',
            author='Cohen, J.',
            year='1988',
            title='Statistical power analysis for the behavioral sciences',
            publisher='Lawrence Erlbaum Associates'
        )
        
        self.scipy_ref = self.references.add_reference(
            'journal',
            author='Virtanen, P.; Gommers, R.; Oliphant, T. E.',
            year='2020',
            title='SciPy 1.0: Fundamental algorithms for scientific computing in Python',
            journal='Nature Methods',
            volume='17',
            issue='3',
            pages='261-272',
            doi='10.1038/s41592-019-0686-2'
        )
        
        
        self.formatter = ReportFormatter()
        self.stat_formatter = StatisticalFormatter()
        self.metadata = {
            'research_type': 'Methodological Study',
            'study_date': datetime.now().isoformat(),
            'title': 'Statistical Power of Pearson Correlation: A Monte Carlo Simulation Study',
            'research_question': 'What is the statistical power of Pearson correlation test under varying conditions?',
            'purpose': 'Test METHOD performance (NOT studying real-world phenomena)',
            'data_type': 'SYNTHETIC (appropriate for methodological research)',
            'why_synthetic_ok': 'We are testing the statistical METHOD, not making claims about reality',
            'design': 'Monte Carlo simulation',
            'factors_varied': [
                'Sample size (n = 20, 50, 100, 200)',
                'True population correlation (r = 0.1, 0.3, 0.5, 0.7)',
                'Number of simulations per condition: 1,000'
            ],
            'dependent_variable': 'Statistical power (proportion of significant results)',
            'significance_level': 0.05,
            'statistical_methods': [
                f'Pearson correlation test ({self.scipy_ref})',
                'Monte Carlo simulation',
                f'Power analysis ({self.cohen_ref})'
            ],
            'claims_about': 'METHOD performance, NOT real-world phenomena',
            'limitations': [
                'Assumes bivariate normal distribution',
                'Results specific to Pearson correlation',
                'Does not address Type I error inflation',
                'Assumes accurate significance level'
            ]
        }
        
        self.results = None
    
    def state_methodological_purpose(self) -> None:
        """
        State that this tests METHODS, not real-world phenomena.
        """
        self.formatter.print_section("METHODOLOGICAL STUDY - IMPORTANT DISTINCTION")
        SafeOutput.safe_print("\n{get_symbol('checkmark')} THIS STUDY IS ABOUT:")
        SafeOutput.safe_print("  - Testing statistical METHOD performance")
        SafeOutput.safe_print("  - Evaluating Pearson correlation power")
        SafeOutput.safe_print("  - Providing guidance for sample size planning")
        
        SafeOutput.safe_print("\n{get_symbol('checkmark')} SYNTHETIC DATA IS APPROPRIATE BECAUSE:")
        SafeOutput.safe_print("  - We are testing the METHOD, not studying the world")
        SafeOutput.safe_print("  - We NEED known ground truth to evaluate the method")
        SafeOutput.safe_print("  - Claims are about METHOD properties, not phenomena")
        
        SafeOutput.safe_print("\n{get_symbol('cross')} THIS STUDY IS NOT ABOUT:")
        SafeOutput.safe_print("  - Real-world phenomena or relationships")
        SafeOutput.safe_print("  - Empirical observations")
        SafeOutput.safe_print("  - Substantive claims about any real variables")
        
        SafeOutput.safe_print("\n{get_symbol('checkmark')} VALID CONCLUSIONS:")
        SafeOutput.safe_print("  - 'Pearson correlation has X% power under these conditions'")
        SafeOutput.safe_print("  - 'Sample size of N is needed to detect effect r'")
        SafeOutput.safe_print("  - 'The method performs well/poorly under condition Y'")
        
        SafeOutput.safe_print("\n{get_symbol('cross')} INVALID CONCLUSIONS:")
        SafeOutput.safe_print("  - Any claims about real-world relationships")
        SafeOutput.safe_print("  - Any substantive empirical findings")
        SafeOutput.safe_print("="*70 + "\n")
    
    def generate_test_data(self, n: int, true_correlation: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data with known correlation.
        
        This is APPROPRIATE for methodological research!
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate bivariate normal data with specified correlation
        mean = [0, 0]
        cov = [[1, true_correlation], [true_correlation, 1]]
        
        data = np.random.multivariate_normal(mean, cov, n)
        x, y = data[:, 0], data[:, 1]
        
        return x, y
    
    def conduct_power_analysis(self) -> pd.DataFrame:
        """
        Conduct comprehensive power analysis via Monte Carlo simulation
        """
        self.formatter.print_section("POWER ANALYSIS VIA MONTE CARLO SIMULATION")
        
        sample_sizes = [20, 50, 100, 200]
        true_correlations = [0.1, 0.3, 0.5, 0.7]
        n_simulations = 1000
        alpha = 0.05
        
        SafeOutput.safe_print(f"\nSimulation Parameters:")
        SafeOutput.safe_print(f"  Number of simulations per condition: {n_simulations}")
        SafeOutput.safe_print(f"  Significance level: alpha = {alpha}")
        SafeOutput.safe_print(f"  Sample sizes tested: {sample_sizes}")
        SafeOutput.safe_print(f"  Effect sizes tested: {true_correlations}")
        
        results = []
        
        SafeOutput.safe_print(f"\nRunning simulations...")
        
        for true_r in true_correlations:
            for n in sample_sizes:
                significant_count = 0
                
                # Run simulations
                for sim in range(n_simulations):
                    x, y = self.generate_test_data(n, true_r, seed=sim)
                    r, p = stats.pearsonr(x, y)
                    
                    if p < alpha:
                        significant_count += 1
                
                power = significant_count / n_simulations
                
                results.append({
                    'true_correlation': true_r,
                    'sample_size': n,
                    'power': power,
                    'n_simulations': n_simulations
                })
                
                SafeOutput.safe_print(f"  r = {true_r:.1f}, n = {n:3d}: Power = {power:.3f} ({power*100:.1f}%)")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def test_method_bias(self) -> None:
        """
        Test if Pearson correlation produces biased estimates
        """
        self.formatter.print_section("BIAS ASSESSMENT")
        
        true_r = 0.5
        sample_sizes = [20, 50, 100]
        n_simulations = 1000
        
        SafeOutput.safe_print(f"\nTesting for bias with true r = {true_r}")
        SafeOutput.safe_print(f"Number of simulations: {n_simulations}")
        
        for n in sample_sizes:
            estimates = []
            
            for sim in range(n_simulations):
                x, y = self.generate_test_data(n, true_r, seed=sim)
                r, _ = stats.pearsonr(x, y)
                estimates.append(r)
            
            estimates = np.array(estimates)
            bias = estimates.mean() - true_r
            rmse = np.sqrt(np.mean((estimates - true_r)**2))
            
            SafeOutput.safe_print(f"\nSample size n = {n}:")
            SafeOutput.safe_print(f"  True r = {true_r:.4f}")
            SafeOutput.safe_print(f"  Mean estimate = {estimates.mean():.4f}")
            SafeOutput.safe_print(f"  Bias = {bias:.4f} ({abs(bias/true_r)*100:.2f}%)")
            SafeOutput.safe_print(f"  RMSE = {rmse:.4f}")
            SafeOutput.safe_print(f"  95% CI = [{np.percentile(estimates, 2.5):.3f}, {np.percentile(estimates, 97.5):.3f}]")
    
    def visualize_power_analysis(self) -> None:
        """Create power analysis visualizations"""
        self.formatter.print_section("VISUALIZATIONS")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Methodological Study: Power Analysis of Pearson Correlation', 
                     fontsize=13, fontweight='bold', y=0.98)
        
        # Power curves
        for true_r in self.results['true_correlation'].unique():
            subset = self.results[self.results['true_correlation'] == true_r]
            axes[0].plot(subset['sample_size'], subset['power'], 
                        'o-', linewidth=2, markersize=8, 
                        label=f'r = {true_r:.1f}')
        
        axes[0].axhline(y=0.80, color='red', linestyle='--', 
                       linewidth=2, label='80% Power Threshold')
        axes[0].set_xlabel('Sample Size', fontsize=11)
        axes[0].set_ylabel('Statistical Power', fontsize=11)
        axes[0].set_title('Power as Function of Sample Size and Effect Size', fontsize=11)
        axes[0].legend(fontsize=9, loc='lower right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Heatmap
        pivot = self.results.pivot(index='true_correlation', 
                                   columns='sample_size', 
                                   values='power')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Power'},
                   annot_kws={'fontsize': 8})
        axes[1].set_xlabel('Sample Size', fontsize=11)
        axes[1].set_ylabel('True Correlation (r)', fontsize=11)
        axes[1].set_title('Power Analysis Heatmap', fontsize=11)
        axes[1].tick_params(axis='both', labelsize=9)
        
        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10, wspace=0.30)
        plt.savefig('07_methodological_power_analysis.png', dpi=300, bbox_inches='tight')
        SafeOutput.safe_print("\n{get_symbol('checkmark')} Power analysis visualization saved")
        plt.close()
        
        # Sample size recommendations
        fig, ax = plt.subplots(figsize=(11, 6))
        
        for target_power in [0.70, 0.80, 0.90]:
            required_ns = []
            effect_sizes = []
            
            for true_r in sorted(self.results['true_correlation'].unique()):
                subset = self.results[self.results['true_correlation'] == true_r]
                subset = subset.sort_values('sample_size')
                
                # Find first n where power >= target
                adequate = subset[subset['power'] >= target_power]
                if len(adequate) > 0:
                    required_n = adequate.iloc[0]['sample_size']
                    required_ns.append(required_n)
                    effect_sizes.append(true_r)
            
            ax.plot(effect_sizes, required_ns, 'o-', linewidth=2, 
                   markersize=8, label=f'{int(target_power*100)}% Power')
        
        ax.set_xlabel('Effect Size (r)', fontsize=12)
        ax.set_ylabel('Required Sample Size', fontsize=12)
        ax.set_title('Sample Size Requirements for Adequate Power', fontsize=13, pad=15)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)
        plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.10)
        plt.savefig('07_methodological_sample_size_guide.png', dpi=300, bbox_inches='tight')
        SafeOutput.safe_print("{get_symbol('checkmark')} Sample size guide saved")
        plt.close()
    
    def generate_report(self) -> None:
        """Generate methodological research report"""
        self.formatter.print_section("METHODOLOGICAL RESEARCH REPORT")
        
        SafeOutput.safe_print(f"\nTitle: {self.metadata['title']}")
        SafeOutput.safe_print(f"\nResearch Question: {self.metadata['research_question']}")
        
        SafeOutput.safe_print("\n--- ABSTRACT ---")
        SafeOutput.safe_print("\nThis methodological study evaluated the statistical power of ")
        SafeOutput.safe_print(f"Pearson correlation test ({self.scipy_ref}) under various conditions ")
        SafeOutput.safe_print(f"using Monte Carlo simulation ({self.cohen_ref}). ")
        SafeOutput.safe_print(f"Power was assessed across {len(self.results['sample_size'].unique())} ")
        SafeOutput.safe_print(f"sample sizes and {len(self.results['true_correlation'].unique())} ")
        SafeOutput.safe_print(f"effect sizes using {self.results['n_simulations'].iloc[0]} simulations ")
        SafeOutput.safe_print(f"per condition.")
        
        SafeOutput.safe_print("\n--- METHODOLOGY ---")
        SafeOutput.safe_print(f"\nDesign: {self.metadata['design']}")
        SafeOutput.safe_print(f"Purpose: {self.metadata['purpose']}")
        SafeOutput.safe_print(f"Data Type: {self.metadata['data_type']}")
        SafeOutput.safe_print(f"Why synthetic data is appropriate: {self.metadata['why_synthetic_ok']}")
        
        SafeOutput.safe_print("\n--- RESULTS ---")
        SafeOutput.safe_print("\nPower analysis revealed that sample size requirements ")
        SafeOutput.safe_print("varied substantially based on the magnitude of the true correlation:")
        
        for true_r in sorted(self.results['true_correlation'].unique()):
            SafeOutput.safe_print(f"\n  For r = {true_r:.1f}:")
            subset = self.results[self.results['true_correlation'] == true_r]
            
            for _, row in subset.iterrows():
                power_pct = row['power'] * 100
                SafeOutput.safe_print(f"    n = {int(row['sample_size']):3d}: Power = {power_pct:5.1f}%")
            
            # Find minimum n for 80% power
            adequate = subset[subset['power'] >= 0.80]
            if len(adequate) > 0:
                min_n = adequate['sample_size'].min()
                SafeOutput.safe_print(f"    -> Minimum n for 80% power: {int(min_n)}")
            else:
                SafeOutput.safe_print(f"    -> Requires n > {int(subset['sample_size'].max())} for 80% power")
        
        SafeOutput.safe_print("\n--- INTERPRETATION ---")
        SafeOutput.safe_print("\n{get_symbol('checkmark')} APPROPRIATE CLAIMS (About the METHOD):")
        SafeOutput.safe_print("  - 'Pearson correlation requires n = 85 for 80% power at r = 0.3'")
        SafeOutput.safe_print("  - 'Power increases with sample size and effect size'")
        SafeOutput.safe_print("  - 'Small effects (r = 0.1) require large samples for adequate power'")
        SafeOutput.safe_print("  - 'The method is sensitive to sample size'")
        
        SafeOutput.safe_print("\n{get_symbol('cross')} INAPPROPRIATE CLAIMS (About phenomena):")
        SafeOutput.safe_print("  - 'X and Y correlate at r = 0.3' (NO! Synthetic data)")
        SafeOutput.safe_print("  - Any claim about real-world relationships")
        SafeOutput.safe_print("  - Any substantive findings about phenomena")
        
        SafeOutput.safe_print("\n{get_symbol('checkmark')} PRACTICAL IMPLICATIONS:")
        SafeOutput.safe_print("  - Researchers should use these results for sample size planning")
        SafeOutput.safe_print("  - Small pilot studies may be underpowered")
        SafeOutput.safe_print("  - Effect size expectations should inform design")
        
        SafeOutput.safe_print("\n--- LIMITATIONS ---")
        for limitation in self.metadata['limitations']:
            SafeOutput.safe_print(f"  - {limitation}")
        
        SafeOutput.safe_print("\n--- CONCLUSION ---")
        SafeOutput.safe_print("\nThis methodological study provides power estimates for ")
        SafeOutput.safe_print(f"Pearson correlation under various conditions ({self.cohen_ref}). ")
        SafeOutput.safe_print("Results can guide sample size planning in applied research. ")
        SafeOutput.safe_print("However, actual power in specific studies depends on ")
        SafeOutput.safe_print("data characteristics and adherence to test assumptions.")
    
    def save_results(self, filename: str = '07_methodological_power_results.csv') -> None:
        """Save power analysis results"""
        if self.results is not None:
            self.results.to_csv(filename, index=False)
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Power analysis results saved to: {filename}")
            SafeOutput.safe_print(f"{get_symbol('checkmark')} Metadata saved to: {metadata_file}")
    
    def compare_parametric_nonparametric(self) -> None:
        """
        Bonus: Compare Pearson vs Spearman power
        """
        self.formatter.print_section("METHOD COMPARISON: PEARSON VS SPEARMAN")
        
        n = 100
        true_r = 0.5
        n_simulations = 1000
        
        pearson_power = 0
        spearman_power = 0
        
        SafeOutput.safe_print(f"\nComparing methods with n = {n}, r = {true_r}")
        
        for sim in range(n_simulations):
            x, y = self.generate_test_data(n, true_r, seed=sim)
            
            # Pearson
            r_pearson, p_pearson = stats.pearsonr(x, y)
            if p_pearson < 0.05:
                pearson_power += 1
            
            # Spearman
            r_spearman, p_spearman = stats.spearmanr(x, y)
            if p_spearman < 0.05:
                spearman_power += 1
        
        pearson_power /= n_simulations
        spearman_power /= n_simulations
        
        SafeOutput.safe_print(f"\nResults:")
        SafeOutput.safe_print(f"  Pearson correlation power: {pearson_power:.3f} ({pearson_power*100:.1f}%)")
        SafeOutput.safe_print(f"  Spearman correlation power: {spearman_power:.3f} ({spearman_power*100:.1f}%)")
        SafeOutput.safe_print(f"  Difference: {abs(pearson_power - spearman_power):.3f}")
        
        if pearson_power > spearman_power:
            SafeOutput.safe_print(f"\n  -> Pearson more powerful (for normal data)")
        else:
            SafeOutput.safe_print(f"\n  -> Spearman competitive or better")
    
    def generate_references(self) -> None:
        """Generate APA 7 reference list."""
        self.formatter.print_section("REFERENCES")
        SafeOutput.safe_print("")
        SafeOutput.safe_print(self.references.generate_reference_list())
    
    def run_full_study(self) -> None:
        """Execute complete methodological study"""
        self.formatter.print_section("METHODOLOGICAL STUDY: STATISTICAL POWER ANALYSIS")
        SafeOutput.safe_print(f"Study Date: {datetime.now().strftime('%Y-%m-%d')}")
        SafeOutput.safe_print(f"Research Type: {self.metadata['research_type']}")
        
        # State purpose upfront
        self.state_methodological_purpose()
        
        # Conduct analysis
        self.conduct_power_analysis()
        self.test_method_bias()
        self.compare_parametric_nonparametric()
        
        # Save and visualize
        self.save_results()
        self.visualize_power_analysis()
        
        # Report
        self.generate_report()
        self.generate_references()
        
        self.formatter.print_section("METHODOLOGICAL STUDY COMPLETE")
        SafeOutput.safe_print("\nThis study provides guidance for researchers planning correlational studies.")
        SafeOutput.safe_print("Results show how sample size and effect size affect statistical power.")
        SafeOutput.safe_print("\nIMPORTANT: These findings are about the METHOD, not about any real phenomena.")


if __name__ == "__main__":
    formatter = ReportFormatter()
    formatter.print_section("EXAMPLE 07: METHODOLOGICAL STUDY")
    SafeOutput.safe_print("\nThis example demonstrates APPROPRIATE use of synthetic data:")
    SafeOutput.safe_print("  - Testing statistical METHOD (not studying phenomena)")
    SafeOutput.safe_print("  - Monte Carlo power analysis")
    SafeOutput.safe_print("  - Method bias assessment")
    SafeOutput.safe_print("  - Parametric vs non-parametric comparison")
    SafeOutput.safe_print("  - Claims about METHOD, not about world")
    SafeOutput.safe_print("  - APA 7 style reporting")
    SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Synthetic data is APPROPRIATE here because we're testing the method!")
    SafeOutput.safe_print("="*70 + "\n")
    
    study = PowerAnalysisStudy()
    study.run_full_study()
    
    SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Example complete. This shows when synthetic data IS valid in research.")
