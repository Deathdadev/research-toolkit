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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import json

# Import from research_toolkit library
from research_toolkit.core import SafeOutput, ReportFormatter, StatisticalFormatter
from research_toolkit.references import APA7ReferenceManager


class PowerAnalysisStudy:
    """
    Methodological study examining statistical power of correlation tests.
    
    Research Type: Methodological Study
    IMPORTANT: This is appropriate use of synthetic data!
    """
    
    def __init__(self):
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
    
    def state_methodological_purpose(self):
        """
        CRITICAL: Clearly state this is methodological research
        """
        print("\n" + "="*70)
        print("METHODOLOGICAL STUDY - IMPORTANT DISTINCTION")
        print("="*70)
        print("\n[OK] THIS STUDY IS ABOUT:")
        print("  - Testing statistical METHOD performance")
        print("  - Evaluating Pearson correlation power")
        print("  - Providing guidance for sample size planning")
        
        print("\n[OK] SYNTHETIC DATA IS APPROPRIATE BECAUSE:")
        print("  - We are testing the METHOD, not studying the world")
        print("  - We NEED known ground truth to evaluate the method")
        print("  - Claims are about METHOD properties, not phenomena")
        
        print("\n[X] THIS STUDY IS NOT ABOUT:")
        print("  - Real-world phenomena or relationships")
        print("  - Empirical observations")
        print("  - Substantive claims about any real variables")
        
        print("\n[OK] VALID CONCLUSIONS:")
        print("  - 'Pearson correlation has X% power under these conditions'")
        print("  - 'Sample size of N is needed to detect effect r'")
        print("  - 'The method performs well/poorly under condition Y'")
        
        print("\n[X] INVALID CONCLUSIONS:")
        print("  - Any claims about real-world relationships")
        print("  - Any substantive empirical findings")
        print("="*70 + "\n")
    
    def generate_test_data(self, n, true_correlation, seed=None):
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
    
    def conduct_power_analysis(self):
        """
        Conduct comprehensive power analysis via Monte Carlo simulation
        """
        print("\n" + "="*70)
        print("POWER ANALYSIS VIA MONTE CARLO SIMULATION")
        print("="*70)
        
        sample_sizes = [20, 50, 100, 200]
        true_correlations = [0.1, 0.3, 0.5, 0.7]
        n_simulations = 1000
        alpha = 0.05
        
        print(f"\nSimulation Parameters:")
        print(f"  Number of simulations per condition: {n_simulations}")
        print(f"  Significance level: alpha = {alpha}")
        print(f"  Sample sizes tested: {sample_sizes}")
        print(f"  Effect sizes tested: {true_correlations}")
        
        results = []
        
        print(f"\nRunning simulations...")
        
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
                
                print(f"  r = {true_r:.1f}, n = {n:3d}: Power = {power:.3f} ({power*100:.1f}%)")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def test_method_bias(self):
        """
        Test if Pearson correlation produces biased estimates
        """
        print("\n" + "="*70)
        print("BIAS ASSESSMENT")
        print("="*70)
        
        true_r = 0.5
        sample_sizes = [20, 50, 100]
        n_simulations = 1000
        
        print(f"\nTesting for bias with true r = {true_r}")
        print(f"Number of simulations: {n_simulations}")
        
        for n in sample_sizes:
            estimates = []
            
            for sim in range(n_simulations):
                x, y = self.generate_test_data(n, true_r, seed=sim)
                r, _ = stats.pearsonr(x, y)
                estimates.append(r)
            
            estimates = np.array(estimates)
            bias = estimates.mean() - true_r
            rmse = np.sqrt(np.mean((estimates - true_r)**2))
            
            print(f"\nSample size n = {n}:")
            print(f"  True r = {true_r:.4f}")
            print(f"  Mean estimate = {estimates.mean():.4f}")
            print(f"  Bias = {bias:.4f} ({abs(bias/true_r)*100:.2f}%)")
            print(f"  RMSE = {rmse:.4f}")
            print(f"  95% CI = [{np.percentile(estimates, 2.5):.3f}, {np.percentile(estimates, 97.5):.3f}]")
    
    def visualize_power_analysis(self):
        """Create power analysis visualizations"""
        print("\n" + "="*70)
        print("VISUALIZATIONS")
        print("="*70)
        
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
        print("\n[OK] Power analysis visualization saved")
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
        print("[OK] Sample size guide saved")
        plt.close()
    
    def generate_report(self):
        """Generate methodological research report"""
        print("\n" + "="*70)
        print("METHODOLOGICAL RESEARCH REPORT")
        print("="*70)
        
        print(f"\nTitle: {self.metadata['title']}")
        print(f"\nResearch Question: {self.metadata['research_question']}")
        
        print("\n--- ABSTRACT ---")
        print("\nThis methodological study evaluated the statistical power of ")
        print(f"Pearson correlation test ({self.scipy_ref}) under various conditions ")
        print(f"using Monte Carlo simulation ({self.cohen_ref}). ")
        print(f"Power was assessed across {len(self.results['sample_size'].unique())} ")
        print(f"sample sizes and {len(self.results['true_correlation'].unique())} ")
        print(f"effect sizes using {self.results['n_simulations'].iloc[0]} simulations ")
        print(f"per condition.")
        
        print("\n--- METHODOLOGY ---")
        print(f"\nDesign: {self.metadata['design']}")
        print(f"Purpose: {self.metadata['purpose']}")
        print(f"Data Type: {self.metadata['data_type']}")
        print(f"Why synthetic data is appropriate: {self.metadata['why_synthetic_ok']}")
        
        print("\n--- RESULTS ---")
        print("\nPower analysis revealed that sample size requirements ")
        print("varied substantially based on the magnitude of the true correlation:")
        
        for true_r in sorted(self.results['true_correlation'].unique()):
            print(f"\n  For r = {true_r:.1f}:")
            subset = self.results[self.results['true_correlation'] == true_r]
            
            for _, row in subset.iterrows():
                power_pct = row['power'] * 100
                print(f"    n = {int(row['sample_size']):3d}: Power = {power_pct:5.1f}%")
            
            # Find minimum n for 80% power
            adequate = subset[subset['power'] >= 0.80]
            if len(adequate) > 0:
                min_n = adequate['sample_size'].min()
                print(f"    -> Minimum n for 80% power: {int(min_n)}")
            else:
                print(f"    -> Requires n > {int(subset['sample_size'].max())} for 80% power")
        
        print("\n--- INTERPRETATION ---")
        print("\n[OK] APPROPRIATE CLAIMS (About the METHOD):")
        print("  - 'Pearson correlation requires n = 85 for 80% power at r = 0.3'")
        print("  - 'Power increases with sample size and effect size'")
        print("  - 'Small effects (r = 0.1) require large samples for adequate power'")
        print("  - 'The method is sensitive to sample size'")
        
        print("\n[X] INAPPROPRIATE CLAIMS (About phenomena):")
        print("  - 'X and Y correlate at r = 0.3' (NO! Synthetic data)")
        print("  - Any claim about real-world relationships")
        print("  - Any substantive findings about phenomena")
        
        print("\n[OK] PRACTICAL IMPLICATIONS:")
        print("  - Researchers should use these results for sample size planning")
        print("  - Small pilot studies may be underpowered")
        print("  - Effect size expectations should inform design")
        
        print("\n--- LIMITATIONS ---")
        for limitation in self.metadata['limitations']:
            print(f"  - {limitation}")
        
        print("\n--- CONCLUSION ---")
        print("\nThis methodological study provides power estimates for ")
        print(f"Pearson correlation under various conditions ({self.cohen_ref}). ")
        print("Results can guide sample size planning in applied research. ")
        print("However, actual power in specific studies depends on ")
        print("data characteristics and adherence to test assumptions.")
    
    def save_results(self, filename='07_methodological_power_results.csv'):
        """Save power analysis results"""
        if self.results is not None:
            self.results.to_csv(filename, index=False)
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"\n[OK] Power analysis results saved to: {filename}")
            print(f"[OK] Metadata saved to: {metadata_file}")
    
    def compare_parametric_nonparametric(self):
        """
        Bonus: Compare Pearson vs Spearman power
        """
        print("\n" + "="*70)
        print("METHOD COMPARISON: PEARSON VS SPEARMAN")
        print("="*70)
        
        n = 100
        true_r = 0.5
        n_simulations = 1000
        
        pearson_power = 0
        spearman_power = 0
        
        print(f"\nComparing methods with n = {n}, r = {true_r}")
        
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
        
        print(f"\nResults:")
        print(f"  Pearson correlation power: {pearson_power:.3f} ({pearson_power*100:.1f}%)")
        print(f"  Spearman correlation power: {spearman_power:.3f} ({spearman_power*100:.1f}%)")
        print(f"  Difference: {abs(pearson_power - spearman_power):.3f}")
        
        if pearson_power > spearman_power:
            print(f"\n  -> Pearson more powerful (for normal data)")
        else:
            print(f"\n  -> Spearman competitive or better")
    
    def generate_references(self):
        """Generate APA 7 reference list"""
        print("\n" + "="*70)
        print("REFERENCES")
        print("="*70)
        print()
        print(self.references.generate_reference_list())
    
    def run_full_study(self):
        """Execute complete methodological study"""
        print("\n" + "="*70)
        print("METHODOLOGICAL STUDY: STATISTICAL POWER ANALYSIS")
        print("="*70)
        print(f"Study Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Research Type: {self.metadata['research_type']}")
        
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
        
        print("\n" + "="*70)
        print("METHODOLOGICAL STUDY COMPLETE")
        print("="*70)
        print("\nThis study provides guidance for researchers planning correlational studies.")
        print("Results show how sample size and effect size affect statistical power.")
        print("\nIMPORTANT: These findings are about the METHOD, not about any real phenomena.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXAMPLE 07: METHODOLOGICAL STUDY")
    print("="*70)
    print("\nThis example demonstrates APPROPRIATE use of synthetic data:")
    print("  - Testing statistical METHOD (not studying phenomena)")
    print("  - Monte Carlo power analysis")
    print("  - Method bias assessment")
    print("  - Parametric vs non-parametric comparison")
    print("  - Claims about METHOD, not about world")
    print("  - APA 7 style reporting")
    print("\n[OK] Synthetic data is APPROPRIATE here because we're testing the method!")
    print("="*70 + "\n")
    
    study = PowerAnalysisStudy()
    study.run_full_study()
    
    print("\n[OK] Example complete. This shows when synthetic data IS valid in research.")
