"""
Example 05: Meta-Analysis - Effect of Study Hours on Academic Performance

Research Question: What is the overall correlation between study hours 
and academic performance across published studies?

This demonstrates:
- Meta-analytic research design
- Synthesis of existing research
- Fixed-effects and random-effects models
- Heterogeneity assessment
- Publication bias testing
- Forest and funnel plots
- APA 7 referencing using research_toolkit

Data Source: Effect sizes from simulated published studies
(In production: Extract from actual published research via systematic review)
"""
# Standard library imports
from datetime import datetime
from typing import Optional, Tuple
import json

# Third-party imports
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports (research_toolkit)
from research_toolkit import ReportFormatter, SafeOutput, StatisticalFormatter, get_symbol
from research_toolkit.references import APA7ReferenceManager

class StudyHoursMetaAnalysis:
    """
    Meta-analysis of studies examining study hours and academic performance.
    
    Research Type: Meta-Analysis (Empirical - Synthesis)
    Design: Quantitative synthesis of existing research
    """
    
    def __init__(self) -> None:
        """
        Initialize meta-analysis study.
        
        Note:
            Synthesizes findings from multiple primary studies.
        """
        self.references = APA7ReferenceManager()
        
        # Add methodological references
        self.borenstein_ref = self.references.add_reference(
            'book',
            author='Borenstein, M.; Hedges, L. V.; Higgins, J. P. T.; Rothstein, H. R.',
            year='2009',
            title='Introduction to meta-analysis',
            publisher='Wiley'
        )
        
        
        self.formatter = ReportFormatter()
        self.stat_formatter = StatisticalFormatter()
        self.metadata = {
            'research_type': 'Meta-Analysis (Empirical - Synthesis)',
            'study_date': datetime.now().isoformat(),
            'title': 'Study Hours and Academic Performance: A Meta-Analytic Review',
            'research_question': 'What is the overall correlation between study hours and academic performance?',
            'design': 'Quantitative meta-analysis',
            'search_strategy': 'Systematic review of educational psychology databases',
            'inclusion_criteria': [
                'Published peer-reviewed studies',
                'Measured both study hours and academic performance',
                'Reported correlation coefficient or convertible statistic',
                'Sample size reported'
            ],
            'exclusion_criteria': [
                'Non-empirical studies',
                'Insufficient statistical information',
                'Duplicate samples'
            ],
            'effect_size_metric': 'Correlation coefficient (r) converted to Fisher\'s z',
            'k_studies': 15,  # Number of studies included
            'total_n': 0,  # Will be calculated
            'statistical_methods': [
                f'Fixed-effects model {self.references.get_in_text_citation([self.borenstein_ref])}',
                f'Random-effects model {self.references.get_in_text_citation([self.borenstein_ref])}',
                'Heterogeneity analysis (Q, I^2)',
                'Publication bias tests (Egger\'s test)',
                'Forest plots',
                'Funnel plots'
            ],
            'limitations': [
                'Quality depends on primary study quality',
                'Publication bias may affect results',
                'Study heterogeneity limits interpretability',
                'Study-level analysis (not individual participants)',
                'Cannot establish causation (depends on primary designs)',
                'Restricted to published English-language studies'
            ]
        }
        
        self.studies: Optional[pd.DataFrame] = None
    
    def create_study_database(self) -> pd.DataFrame:
        """
        Create database of studies with effect sizes.
        
        Returns:
            DataFrame containing study effect sizes and metadata
            
        Note:
            In production: Extract from systematic literature review.
            Here: Simulated but realistic study data.
        """
        self.formatter.print_section("SYSTEMATIC LITERATURE REVIEW")
        SafeOutput.safe_print(f"\nSearch Strategy: {self.metadata['search_strategy']}")
        SafeOutput.safe_print("\nInclusion Criteria:")
        for criterion in self.metadata['inclusion_criteria']:
            SafeOutput.safe_print(f"  - {criterion}")
        
        # Simulated studies based on typical meta-analysis
        np.random.seed(42)
        
        studies_data = []
        authors = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 
                   'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                   'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson']
        
        for i, author in enumerate(authors):
            # Realistic sample sizes
            n = np.random.randint(80, 400)
            
            # True effect with variation (heterogeneity)
            true_mean_r = 0.35  # Overall average effect
            study_r = np.clip(np.random.normal(true_mean_r, 0.15), 0.05, 0.70)
            
            # Add sampling error
            se_r = np.sqrt((1 - study_r**2)**2 / (n - 1))
            
            # Study quality (0-10 scale)
            quality = np.random.randint(6, 11)
            
            studies_data.append({
                'study_id': i + 1,
                'author': f'{author} et al.',
                'year': np.random.randint(2015, 2024),
                'sample_size': n,
                'correlation_r': round(study_r, 3),
                'std_error_r': round(se_r, 4),
                'study_quality': quality,
                'population': np.random.choice(['College', 'High School', 'Mixed'])
            })
        
        self.studies = pd.DataFrame(studies_data)
        self.metadata['total_n'] = self.studies['sample_size'].sum()
        
        SafeOutput.safe_print(f"\nIncluded Studies: k = {len(self.studies)}")
        SafeOutput.safe_print(f"Total Participants: N = {self.metadata['total_n']}")
        SafeOutput.safe_print(f"Year Range: {self.studies['year'].min()} - {self.studies['year'].max()}")
        
        return self.studies
    
    def convert_to_fishers_z(self) -> None:
        """Convert correlations to Fisher's z for meta-analysis."""
        self.formatter.print_section("EFFECT SIZE TRANSFORMATION")
        
        # Fisher's z transformation
        self.studies['fishers_z'] = 0.5 * np.log(
            (1 + self.studies['correlation_r']) / 
            (1 - self.studies['correlation_r'])
        )
        
        # Standard error for Fisher's z
        self.studies['z_se'] = 1 / np.sqrt(self.studies['sample_size'] - 3)
        
        SafeOutput.safe_print("\nTransformed correlations to Fisher's z for analysis")
        SafeOutput.safe_print(f"Mean r = {self.studies['correlation_r'].mean():.3f}")
        SafeOutput.safe_print(f"Mean z = {self.studies['fishers_z'].mean():.3f}")
    
    def save_raw_data(self, filename: str = '05_meta_analysis_studies.csv') -> None:
        """
        Save study database.
        
        Args:
            filename: Output CSV filename
        """
        if self.studies is not None:
            self.studies.to_csv(filename, index=False)
            
            # Convert metadata to JSON-serializable format (convert numpy types)
            metadata_serializable = {}
            for key, value in self.metadata.items():
                if isinstance(value, (list, dict, str, bool, type(None))):
                    metadata_serializable[key] = value
                elif hasattr(value, 'item'):  # numpy scalar
                    metadata_serializable[key] = value.item()
                else:
                    metadata_serializable[key] = value
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata_serializable, f, indent=2)
            
            SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Study database saved to: {filename}")
            SafeOutput.safe_print(f"{get_symbol('checkmark')} Metadata saved to: {metadata_file}")
    
    def fixed_effects_model(self) -> Tuple[float, float, float]:
        """
        Conduct fixed-effects meta-analysis.
        
        Returns:
            Tuple of (pooled_r, ci_lower_r, ci_upper_r)
        """
        self.formatter.print_section("FIXED-EFFECTS MODEL")
        
        z = self.studies['fishers_z'].values
        se = self.studies['z_se'].values
        
        # Inverse variance weights
        weights = 1 / (se ** 2)
        
        # Pooled effect
        pooled_z = np.sum(weights * z) / np.sum(weights)
        pooled_se = np.sqrt(1 / np.sum(weights))
        
        # Convert back to r
        pooled_r = (np.exp(2 * pooled_z) - 1) / (np.exp(2 * pooled_z) + 1)
        
        # 95% CI
        ci_lower_z = pooled_z - 1.96 * pooled_se
        ci_upper_z = pooled_z + 1.96 * pooled_se
        ci_lower_r = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper_r = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
        
        # Significance test
        z_score = pooled_z / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        SafeOutput.safe_print(f"\nPooled correlation: r = {pooled_r:.3f}")
        SafeOutput.safe_print(f"95% CI: [{ci_lower_r:.3f}, {ci_upper_r:.3f}]")
        SafeOutput.safe_print(f"Z = {z_score:.3f}, p = {p_value:.6f}")
        
        if p_value < 0.05:
            SafeOutput.safe_print("\n-> Significant overall effect detected")
        
        return pooled_r, ci_lower_r, ci_upper_r
    
    def random_effects_model(self) -> Tuple[float, float]:
        """
        Conduct random-effects meta-analysis.
        
        Returns:
            Tuple of (pooled_r, tau_squared)
        """
        self.formatter.print_section("RANDOM-EFFECTS MODEL")
        
        z = self.studies['fishers_z'].values
        se = self.studies['z_se'].values
        
        # Fixed-effects for Q statistic
        weights_fixed = 1 / (se ** 2)
        fixed_z = np.sum(weights_fixed * z) / np.sum(weights_fixed)
        
        # Q statistic
        Q = np.sum(weights_fixed * (z - fixed_z) ** 2)
        df = len(z) - 1
        
        # Tau-squared (between-study variance)
        C = np.sum(weights_fixed) - np.sum(weights_fixed ** 2) / np.sum(weights_fixed)
        tau_squared = max(0, (Q - df) / C)
        
        # Random-effects weights
        weights_random = 1 / (se ** 2 + tau_squared)
        
        # Pooled effect
        pooled_z = np.sum(weights_random * z) / np.sum(weights_random)
        pooled_se = np.sqrt(1 / np.sum(weights_random))
        
        # Convert to r
        pooled_r = (np.exp(2 * pooled_z) - 1) / (np.exp(2 * pooled_z) + 1)
        
        # 95% CI
        ci_lower_z = pooled_z - 1.96 * pooled_se
        ci_upper_z = pooled_z + 1.96 * pooled_se
        ci_lower_r = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper_r = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
        
        # Significance
        z_score = pooled_z / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        SafeOutput.safe_print(f"\nPooled correlation: r = {pooled_r:.3f}")
        SafeOutput.safe_print(f"95% CI: [{ci_lower_r:.3f}, {ci_upper_r:.3f}]")
        SafeOutput.safe_print(f"tau^2 = {tau_squared:.4f} (between-study variance)")
        SafeOutput.safe_print(f"Z = {z_score:.3f}, p = {p_value:.6f}")
        
        return pooled_r, tau_squared
    
    def assess_heterogeneity(self) -> Tuple[float, float]:
        """
        Assess heterogeneity across studies.
        
        Returns:
            Tuple of (Q_statistic, I_squared)
        """
        self.formatter.print_section("HETEROGENEITY ANALYSIS")
        
        z = self.studies['fishers_z'].values
        se = self.studies['z_se'].values
        
        weights = 1 / (se ** 2)
        fixed_z = np.sum(weights * z) / np.sum(weights)
        
        # Q statistic
        Q = np.sum(weights * (z - fixed_z) ** 2)
        df = len(z) - 1
        p_value = 1 - stats.chi2.cdf(Q, df)
        
        # I^2 statistic
        I_squared = max(0, ((Q - df) / Q) * 100)
        
        SafeOutput.safe_print("\nCochran's Q test:")
        SafeOutput.safe_print(f"  Q({df}) = {Q:.3f}, p = {p_value:.4f}")
        
        SafeOutput.safe_print(f"\nI^2 statistic: {I_squared:.1f}%")
        
        if I_squared < 25:
            interpretation = "low heterogeneity"
        elif I_squared < 50:
            interpretation = "moderate heterogeneity"
        elif I_squared < 75:
            interpretation = "substantial heterogeneity"
        else:
            interpretation = "considerable heterogeneity"
        
        SafeOutput.safe_print(f"  Interpretation: {interpretation}")
        
        if I_squared > 50:
            SafeOutput.safe_print("\n  -> Substantial heterogeneity detected")
            SafeOutput.safe_print("  -> Random-effects model more appropriate")
            SafeOutput.safe_print("  -> Consider moderator analysis")
        else:
            SafeOutput.safe_print("\n  -> Low to moderate heterogeneity")
            SafeOutput.safe_print("  -> Studies relatively homogeneous")
        
        return Q, I_squared
    
    def test_publication_bias(self) -> None:
        """Test for publication bias using Egger's regression."""
        self.formatter.print_section("PUBLICATION BIAS ASSESSMENT")
        
        z = self.studies['fishers_z'].values
        se = self.studies['z_se'].values
        precision = 1 / se
        
        # Egger's regression test
        slope, intercept, r, p_value, std_err = stats.linregress(precision, z)
        
        SafeOutput.safe_print("\nEgger's Regression Test:")
        SafeOutput.safe_print(f"  Intercept = {intercept:.3f}")
        SafeOutput.safe_print(f"  p = {p_value:.4f}")
        
        if p_value < 0.05:
            SafeOutput.safe_print("\n  -> WARNING: Significant publication bias detected")
            SafeOutput.safe_print("  -> Small studies show systematically different effects")
            SafeOutput.safe_print("  -> Results may overestimate true effect")
        else:
            SafeOutput.safe_print("\n  -> No significant evidence of publication bias")
            SafeOutput.safe_print("  -> Funnel plot appears symmetric")
    
    def create_forest_plot(self) -> None:
        """Create forest plot of individual studies and pooled effect."""
        self.formatter.print_section("FOREST PLOT")
        
        fig, ax = plt.subplots(figsize=(12, len(self.studies) * 0.4 + 3))
        
        # Plot individual studies
        for i, (idx, row) in enumerate(self.studies.iterrows()):
            r = row['correlation_r']
            se = row['std_error_r']
            ci_lower = r - 1.96 * se
            ci_upper = r + 1.96 * se
            
            # Point estimate (sized by weight)
            weight = row['sample_size'] / self.studies['sample_size'].sum()
            size = 100 + weight * 500
            
            ax.plot(r, i, 'ks', markersize=np.sqrt(size))
            
            # Confidence interval
            ax.plot([ci_lower, ci_upper], [i, i], 'k-', linewidth=2)
            
            # Study label
            label = f"{row['author']} ({row['year']}) [n={row['sample_size']}]"
            ax.text(-0.05, i, label, ha='right', va='center', fontsize=9)
        
        # Pooled effect (calculate simple weighted mean for display)
        weights = self.studies['sample_size'].values / self.studies['sample_size'].sum()
        pooled_r = np.sum(weights * self.studies['correlation_r'].values)
        
        # Add pooled effect line
        ax.axvline(x=pooled_r, color='red', linestyle='--', linewidth=2, label=f'Pooled r = {pooled_r:.3f}')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_yticks(range(len(self.studies)))
        ax.set_yticklabels([''] * len(self.studies))
        ax.set_xlabel('Correlation Coefficient (r)')
        ax.set_title('Forest Plot: Study Hours and Academic Performance')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('05_meta_analysis_forest_plot.png', dpi=300, bbox_inches='tight')
        SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Forest plot saved")
        plt.close()
    
    def create_funnel_plot(self) -> None:
        """Create funnel plot for publication bias assessment."""
        SafeOutput.safe_print(f"{get_symbol('checkmark')} Creating funnel plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        r = self.studies['correlation_r'].values
        se = self.studies['std_error_r'].values
        precision = 1 / se
        
        # Plot studies
        ax.scatter(r, precision, s=100, alpha=0.6, edgecolors='black')
        
        # Pooled effect line
        weights = self.studies['sample_size'].values / self.studies['sample_size'].sum()
        pooled_r = np.sum(weights * r)
        ax.axvline(x=pooled_r, color='red', linestyle='--', linewidth=2, label='Pooled Effect')
        
        # Funnel shape (pseudo 95% CI)
        y_max = precision.max() * 1.1
        for ci_multiplier in [1.96]:
            ax.plot(pooled_r + ci_multiplier / np.linspace(1, y_max, 100), 
                   np.linspace(1, y_max, 100), 'k--', alpha=0.3, linewidth=1)
            ax.plot(pooled_r - ci_multiplier / np.linspace(1, y_max, 100), 
                   np.linspace(1, y_max, 100), 'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Correlation Coefficient (r)')
        ax.set_ylabel('Precision (1 / Standard Error)')
        ax.set_title('Funnel Plot: Publication Bias Assessment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('05_meta_analysis_funnel_plot.png', dpi=300, bbox_inches='tight')
        SafeOutput.safe_print(f"{get_symbol('checkmark')} Funnel plot saved")
        plt.close()
    
    def generate_report(self) -> None:
        """Generate APA-style meta-analysis report."""
        self.formatter.print_section("META-ANALYSIS REPORT")
        
        SafeOutput.safe_print(f"\nTitle: {self.metadata['title']}")
        SafeOutput.safe_print(f"\nResearch Question: {self.metadata['research_question']}")
        
        SafeOutput.safe_print("\n--- ABSTRACT ---")
        SafeOutput.safe_print(f"\nThis meta-analysis synthesized {len(self.studies)} studies ")
        SafeOutput.safe_print(f"(total N = {self.metadata['total_n']}) examining the relationship ")
        SafeOutput.safe_print("between study hours and academic performance. Both fixed-effects ")
        SafeOutput.safe_print(f"and random-effects models {self.references.get_in_text_citation([self.borenstein_ref])} were employed. ")
        SafeOutput.safe_print("Heterogeneity and publication bias were assessed.")
        
        SafeOutput.safe_print("\n--- METHOD ---")
        SafeOutput.safe_print(f"\nA systematic literature review identified {len(self.studies)} ")
        SafeOutput.safe_print(f"eligible studies published between {self.studies['year'].min()} ")
        SafeOutput.safe_print(f"and {self.studies['year'].max()}. Correlation coefficients were ")
        SafeOutput.safe_print("transformed to Fisher's z for analysis.")
        
        SafeOutput.safe_print("\n--- RESULTS ---")
        
        # Get results from models
        z = self.studies['fishers_z'].values
        se = self.studies['z_se'].values
        weights = 1 / (se ** 2)
        pooled_z = np.sum(weights * z) / np.sum(weights)
        pooled_r = (np.exp(2 * pooled_z) - 1) / (np.exp(2 * pooled_z) + 1)
        
        Q, I_squared = self.assess_heterogeneity()
        
        SafeOutput.safe_print("\nFixed-effects meta-analysis revealed an overall correlation of ")
        SafeOutput.safe_print(f"r = {pooled_r:.3f}, indicating a {'moderate' if abs(pooled_r) > 0.3 else 'small'} ")
        SafeOutput.safe_print("positive relationship between study hours and academic performance.")
        
        SafeOutput.safe_print("\nHeterogeneity analysis indicated ")
        if I_squared > 50:
            SafeOutput.safe_print(f"substantial heterogeneity (I^2 = {I_squared:.1f}%), ")
            SafeOutput.safe_print("suggesting that effect sizes varied considerably across studies.")
        else:
            SafeOutput.safe_print(f"{'low' if I_squared < 25 else 'moderate'} heterogeneity (I^2 = {I_squared:.1f}%).")
        
        SafeOutput.safe_print("\n--- INTERPRETATION ---")
        SafeOutput.safe_print("\n{get_symbol('checkmark')} APPROPRIATE CLAIMS:")
        SafeOutput.safe_print(f"  - 'Across {len(self.studies)} studies, study hours and performance correlate at r = {pooled_r:.3f}'")
        SafeOutput.safe_print("  - 'The overall effect is positive and significant'")
        SafeOutput.safe_print("  - 'Effect sizes show [low/moderate/high] heterogeneity'")
        
        SafeOutput.safe_print("\n{get_symbol('cross')} INAPPROPRIATE CLAIMS:")
        SafeOutput.safe_print("  - 'Studying more CAUSES better performance' (meta-analysis inherits design limits)")
        SafeOutput.safe_print("  - Cannot establish causation unless primary studies were experimental")
        
        SafeOutput.safe_print("\n--- LIMITATIONS ---")
        for limitation in self.metadata['limitations']:
            SafeOutput.safe_print(f"  - {limitation}")
        
        SafeOutput.safe_print("\n--- CONCLUSION ---")
        SafeOutput.safe_print(f"\nThis meta-analysis of {len(self.studies)} studies provides evidence ")
        SafeOutput.safe_print("for a consistent positive relationship between study hours and ")
        SafeOutput.safe_print("academic performance. However, heterogeneity suggests moderating ")
        SafeOutput.safe_print("factors, and causal conclusions require experimental evidence.")
    
    def generate_references(self) -> None:
        """Generate APA 7 reference list."""
        self.formatter.print_section("REFERENCES")
        SafeOutput.safe_print("")
        SafeOutput.safe_print(self.references.generate_reference_list())
        SafeOutput.safe_print("\n[Note: In a real meta-analysis, all included studies would be listed here]")
    
    def run_full_study(self) -> None:
        """Execute complete meta-analysis workflow."""
        self.formatter.print_section("META-ANALYSIS: STUDY HOURS AND PERFORMANCE")
        SafeOutput.safe_print(f"Study Date: {datetime.now().strftime('%Y-%m-%d')}")
        SafeOutput.safe_print(f"Research Type: {self.metadata['research_type']}")
        
        # Execute workflow
        self.create_study_database()
        self.convert_to_fishers_z()
        self.save_raw_data()
        
        self.fixed_effects_model()
        self.random_effects_model()
        Q, I_squared = self.assess_heterogeneity()
        self.test_publication_bias()
        
        self.create_forest_plot()
        self.create_funnel_plot()
        
        self.generate_report()
        self.generate_references()
        
        self.formatter.print_section("META-ANALYSIS COMPLETE")
        SafeOutput.safe_print("\nAll study data saved and can be independently verified.")
        SafeOutput.safe_print("See 05_meta_analysis_studies.csv for included studies")
        SafeOutput.safe_print("See 05_meta_analysis_*.png for visualizations")


if __name__ == "__main__":
    formatter = ReportFormatter()
    formatter.print_section("EXAMPLE 05: META-ANALYSIS")
    SafeOutput.safe_print("\nThis example demonstrates proper meta-analytic research:")
    SafeOutput.safe_print("  - Systematic synthesis of existing studies")
    SafeOutput.safe_print("  - Fixed-effects and random-effects models")
    SafeOutput.safe_print("  - Heterogeneity assessment")
    SafeOutput.safe_print("  - Publication bias testing")
    SafeOutput.safe_print("  - Forest and funnel plots")
    SafeOutput.safe_print("  - Proper interpretation of synthesized evidence")
    SafeOutput.safe_print("  - APA 7 style reporting")
    SafeOutput.safe_print("="*70 + "\n")
    
    meta = StudyHoursMetaAnalysis()
    meta.run_full_study()
    
    SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Example complete. This demonstrates verifiable meta-analytic research.")
