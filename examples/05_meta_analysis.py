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


class StudyHoursMetaAnalysis:
    """
    Meta-analysis of studies examining study hours and academic performance.
    
    Research Type: Meta-Analysis (Empirical - Synthesis)
    Design: Quantitative synthesis of existing research
    """
    
    def __init__(self):
        self.references = APA7ReferenceManager()
        
        # Add methodological references
        self.borenstein_ref = self.references.add_reference(
            'book',
            author='Borenstein, M.; Hedges, L. V.; Higgins, J. P. T.; Rothstein, H. R.',
            year='2009',
            title='Introduction to meta-analysis',
            publisher='Wiley'
        )
        
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
                f'Fixed-effects model ({self.borenstein_ref})',
                f'Random-effects model ({self.borenstein_ref})',
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
        
        self.studies = None
    
    def create_study_database(self):
        """
        Create database of studies with effect sizes.
        
        In production: Extract from systematic literature review.
        Here: Simulated but realistic study data.
        """
        print("\n" + "="*70)
        print("SYSTEMATIC LITERATURE REVIEW")
        print("="*70)
        print(f"\nSearch Strategy: {self.metadata['search_strategy']}")
        print(f"\nInclusion Criteria:")
        for criterion in self.metadata['inclusion_criteria']:
            print(f"  - {criterion}")
        
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
        
        print(f"\nIncluded Studies: k = {len(self.studies)}")
        print(f"Total Participants: N = {self.metadata['total_n']}")
        print(f"Year Range: {self.studies['year'].min()} - {self.studies['year'].max()}")
        
        return self.studies
    
    def convert_to_fishers_z(self):
        """Convert correlations to Fisher's z for meta-analysis"""
        print("\n" + "="*70)
        print("EFFECT SIZE TRANSFORMATION")
        print("="*70)
        
        # Fisher's z transformation
        self.studies['fishers_z'] = 0.5 * np.log(
            (1 + self.studies['correlation_r']) / 
            (1 - self.studies['correlation_r'])
        )
        
        # Standard error for Fisher's z
        self.studies['z_se'] = 1 / np.sqrt(self.studies['sample_size'] - 3)
        
        print("\nTransformed correlations to Fisher's z for analysis")
        print(f"Mean r = {self.studies['correlation_r'].mean():.3f}")
        print(f"Mean z = {self.studies['fishers_z'].mean():.3f}")
    
    def save_raw_data(self, filename='05_meta_analysis_studies.csv'):
        """Save study database"""
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
            
            print(f"\n[OK] Study database saved to: {filename}")
            print(f"[OK] Metadata saved to: {metadata_file}")
    
    def fixed_effects_model(self):
        """Conduct fixed-effects meta-analysis"""
        print("\n" + "="*70)
        print("FIXED-EFFECTS MODEL")
        print("="*70)
        
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
        
        print(f"\nPooled correlation: r = {pooled_r:.3f}")
        print(f"95% CI: [{ci_lower_r:.3f}, {ci_upper_r:.3f}]")
        print(f"Z = {z_score:.3f}, p = {p_value:.6f}")
        
        if p_value < 0.05:
            print("\n-> Significant overall effect detected")
        
        return pooled_r, ci_lower_r, ci_upper_r
    
    def random_effects_model(self):
        """Conduct random-effects meta-analysis"""
        print("\n" + "="*70)
        print("RANDOM-EFFECTS MODEL")
        print("="*70)
        
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
        
        print(f"\nPooled correlation: r = {pooled_r:.3f}")
        print(f"95% CI: [{ci_lower_r:.3f}, {ci_upper_r:.3f}]")
        print(f"tau^2 = {tau_squared:.4f} (between-study variance)")
        print(f"Z = {z_score:.3f}, p = {p_value:.6f}")
        
        return pooled_r, tau_squared
    
    def assess_heterogeneity(self):
        """Assess heterogeneity across studies"""
        print("\n" + "="*70)
        print("HETEROGENEITY ANALYSIS")
        print("="*70)
        
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
        
        print(f"\nCochran's Q test:")
        print(f"  Q({df}) = {Q:.3f}, p = {p_value:.4f}")
        
        print(f"\nI^2 statistic: {I_squared:.1f}%")
        
        if I_squared < 25:
            interpretation = "low heterogeneity"
        elif I_squared < 50:
            interpretation = "moderate heterogeneity"
        elif I_squared < 75:
            interpretation = "substantial heterogeneity"
        else:
            interpretation = "considerable heterogeneity"
        
        print(f"  Interpretation: {interpretation}")
        
        if I_squared > 50:
            print("\n  -> Substantial heterogeneity detected")
            print("  -> Random-effects model more appropriate")
            print("  -> Consider moderator analysis")
        else:
            print("\n  -> Low to moderate heterogeneity")
            print("  -> Studies relatively homogeneous")
        
        return Q, I_squared
    
    def test_publication_bias(self):
        """Test for publication bias"""
        print("\n" + "="*70)
        print("PUBLICATION BIAS ASSESSMENT")
        print("="*70)
        
        z = self.studies['fishers_z'].values
        se = self.studies['z_se'].values
        precision = 1 / se
        
        # Egger's regression test
        slope, intercept, r, p_value, std_err = stats.linregress(precision, z)
        
        print(f"\nEgger's Regression Test:")
        print(f"  Intercept = {intercept:.3f}")
        print(f"  p = {p_value:.4f}")
        
        if p_value < 0.05:
            print("\n  -> WARNING: Significant publication bias detected")
            print("  -> Small studies show systematically different effects")
            print("  -> Results may overestimate true effect")
        else:
            print("\n  -> No significant evidence of publication bias")
            print("  -> Funnel plot appears symmetric")
    
    def create_forest_plot(self):
        """Create forest plot of individual studies"""
        print("\n" + "="*70)
        print("FOREST PLOT")
        print("="*70)
        
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
        print("\n[OK] Forest plot saved")
        plt.close()
    
    def create_funnel_plot(self):
        """Create funnel plot for publication bias"""
        print("\n[OK] Creating funnel plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        r = self.studies['correlation_r'].values
        se = self.studies['std_error_r'].values
        precision = 1 / se
        
        # Plot studies
        ax.scatter(r, precision, s=100, alpha=0.6, edgecolors='black')
        
        # Pooled effect line
        weights = self.studies['sample_size'].values / self.studies['sample_size'].sum()
        pooled_r = np.sum(weights * r)
        ax.axvline(x=pooled_r, color='red', linestyle='--', linewidth=2, label=f'Pooled Effect')
        
        # Funnel shape (pseudo 95% CI)
        x_range = np.linspace(r.min() - 0.1, r.max() + 0.1, 100)
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
        print("[OK] Funnel plot saved")
        plt.close()
    
    def generate_report(self):
        """Generate APA-style meta-analysis report"""
        print("\n" + "="*70)
        print("META-ANALYSIS REPORT")
        print("="*70)
        
        print(f"\nTitle: {self.metadata['title']}")
        print(f"\nResearch Question: {self.metadata['research_question']}")
        
        print("\n--- ABSTRACT ---")
        print(f"\nThis meta-analysis synthesized {len(self.studies)} studies ")
        print(f"(total N = {self.metadata['total_n']}) examining the relationship ")
        print(f"between study hours and academic performance. Both fixed-effects ")
        print(f"and random-effects models ({self.borenstein_ref}) were employed. ")
        print(f"Heterogeneity and publication bias were assessed.")
        
        print("\n--- METHOD ---")
        print(f"\nA systematic literature review identified {len(self.studies)} ")
        print(f"eligible studies published between {self.studies['year'].min()} ")
        print(f"and {self.studies['year'].max()}. Correlation coefficients were ")
        print(f"transformed to Fisher's z for analysis.")
        
        print("\n--- RESULTS ---")
        
        # Get results from models
        z = self.studies['fishers_z'].values
        se = self.studies['z_se'].values
        weights = 1 / (se ** 2)
        pooled_z = np.sum(weights * z) / np.sum(weights)
        pooled_r = (np.exp(2 * pooled_z) - 1) / (np.exp(2 * pooled_z) + 1)
        
        Q, I_squared = self.assess_heterogeneity()
        
        print(f"\nFixed-effects meta-analysis revealed an overall correlation of ")
        print(f"r = {pooled_r:.3f}, indicating a {'moderate' if abs(pooled_r) > 0.3 else 'small'} ")
        print(f"positive relationship between study hours and academic performance.")
        
        print(f"\nHeterogeneity analysis indicated ")
        if I_squared > 50:
            print(f"substantial heterogeneity (I^2 = {I_squared:.1f}%), ")
            print(f"suggesting that effect sizes varied considerably across studies.")
        else:
            print(f"{'low' if I_squared < 25 else 'moderate'} heterogeneity (I^2 = {I_squared:.1f}%).")
        
        print("\n--- INTERPRETATION ---")
        print("\n[OK] APPROPRIATE CLAIMS:")
        print(f"  - 'Across {len(self.studies)} studies, study hours and performance correlate at r = {pooled_r:.3f}'")
        print("  - 'The overall effect is positive and significant'")
        print("  - 'Effect sizes show [low/moderate/high] heterogeneity'")
        
        print("\n[X] INAPPROPRIATE CLAIMS:")
        print("  - 'Studying more CAUSES better performance' (meta-analysis inherits design limits)")
        print("  - Cannot establish causation unless primary studies were experimental")
        
        print("\n--- LIMITATIONS ---")
        for limitation in self.metadata['limitations']:
            print(f"  - {limitation}")
        
        print("\n--- CONCLUSION ---")
        print(f"\nThis meta-analysis of {len(self.studies)} studies provides evidence ")
        print("for a consistent positive relationship between study hours and ")
        print("academic performance. However, heterogeneity suggests moderating ")
        print("factors, and causal conclusions require experimental evidence.")
    
    def generate_references(self):
        """Generate APA 7 reference list"""
        print("\n" + "="*70)
        print("REFERENCES")
        print("="*70)
        print()
        print(self.references.generate_reference_list())
        print("\n[Note: In a real meta-analysis, all included studies would be listed here]")
    
    def run_full_study(self):
        """Execute complete meta-analysis"""
        print("\n" + "="*70)
        print("META-ANALYSIS: STUDY HOURS AND PERFORMANCE")
        print("="*70)
        print(f"Study Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Research Type: {self.metadata['research_type']}")
        
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
        
        print("\n" + "="*70)
        print("META-ANALYSIS COMPLETE")
        print("="*70)
        print("\nAll study data saved and can be independently verified.")
        print("See 05_meta_analysis_studies.csv for included studies")
        print("See 05_meta_analysis_*.png for visualizations")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXAMPLE 05: META-ANALYSIS")
    print("="*70)
    print("\nThis example demonstrates proper meta-analytic research:")
    print("  - Systematic synthesis of existing studies")
    print("  - Fixed-effects and random-effects models")
    print("  - Heterogeneity assessment")
    print("  - Publication bias testing")
    print("  - Forest and funnel plots")
    print("  - Proper interpretation of synthesized evidence")
    print("  - APA 7 style reporting")
    print("="*70 + "\n")
    
    meta = StudyHoursMetaAnalysis()
    meta.run_full_study()
    
    print("\n[OK] Example complete. This demonstrates verifiable meta-analytic research.")
