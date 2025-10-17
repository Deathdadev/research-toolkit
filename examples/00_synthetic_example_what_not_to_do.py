"""
SYNTHETIC (NON-VERIFIABLE) RESEARCH EXAMPLE

WARNING: This is an example of what NOT to do for empirical research.
This demonstrates synthetic data that CANNOT be verified or peer-reviewed.

Use this as a comparison to understand why verifiable_research.py is the correct approach.

Research Question: Does the number of study hours affect test scores?

PROBLEMS WITH THIS APPROACH:
1. Data is artificially generated (not real observations)
2. Results are predetermined by the code
3. Cannot be independently verified
4. Not suitable for peer review
5. No real-world validity
6. Relationships are hardcoded, not discovered

This is useful for:
- Learning statistical techniques
- Teaching methodology
- Testing analysis pipelines
- Demonstrations

But NOT for:
- Actual empirical research
- Making real-world claims
- Publishing findings
- Training AI to conduct research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


class SyntheticStudyExample:
    """
    Example of synthetic data study - FOR EDUCATIONAL COMPARISON ONLY
    This demonstrates what NOT to do for verifiable empirical research.
    """
    
    def __init__(self, n_samples=200, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        self.data = None
        
    def generate_synthetic_data(self):
        """
        PROBLEM: This generates fake data, not real observations.
        The relationships are predetermined by the formulas below.
        """
        # Independent variables (made up)
        study_hours = np.random.uniform(0, 10, self.n_samples)
        prior_knowledge = np.random.uniform(40, 90, self.n_samples)
        
        # PROBLEM: We're hardcoding the relationship we want to "discover"
        interaction = 0.3 * study_hours * (prior_knowledge / 100)
        
        # PROBLEM: The correlation is built in, not discovered
        test_scores = (
            50 +  # baseline
            3.5 * study_hours +  # We decide this coefficient
            0.4 * prior_knowledge +  # We decide this too
            interaction +  # We decide the interaction
            np.random.normal(0, 5, self.n_samples)  # We decide the noise
        )
        
        test_scores = np.clip(test_scores, 0, 100)
        
        self.data = pd.DataFrame({
            'study_hours': study_hours,
            'prior_knowledge': prior_knowledge,
            'test_score': test_scores,
            'group': np.where(prior_knowledge > 65, 'High Prior', 'Low Prior')
        })
        
        return self.data
    
    def descriptive_statistics(self):
        """Calculate descriptive statistics on synthetic data"""
        print("=" * 60)
        print("DESCRIPTIVE STATISTICS (SYNTHETIC DATA)")
        print("=" * 60)
        print(self.data.describe())
        print("\nCorrelation Matrix:")
        print(self.data[['study_hours', 'prior_knowledge', 'test_score']].corr())
        print()
        
    def hypothesis_testing(self):
        """
        PROBLEM: These tests will "confirm" relationships we hardcoded.
        This is circular reasoning - we're just discovering what we put in.
        """
        print("=" * 60)
        print("HYPOTHESIS TESTING (SYNTHETIC DATA)")
        print("=" * 60)
        
        high_prior = self.data[self.data['group'] == 'High Prior']['test_score']
        low_prior = self.data[self.data['group'] == 'Low Prior']['test_score']
        
        t_stat, p_value = stats.ttest_ind(high_prior, low_prior)
        print(f"Independent T-Test (High vs Low Prior Knowledge):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  PROBLEM: This 'confirms' what we programmed in!")
        print()
        
        r, p = stats.pearsonr(self.data['study_hours'], self.data['test_score'])
        print(f"Pearson Correlation (Study Hours vs Test Score):")
        print(f"  r = {r:.4f}, p = {p:.4f}")
        print(f"  PROBLEM: We decided this coefficient was 3.5 in the formula!")
        print()
        
    def regression_analysis(self):
        """Regression on synthetic data - just recovering what we put in"""
        print("=" * 60)
        print("REGRESSION ANALYSIS (SYNTHETIC DATA)")
        print("=" * 60)
        
        X = self.data[['study_hours', 'prior_knowledge']]
        y = self.data['test_score']
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        print(f"Intercept: {model.intercept_:.4f}")
        print(f"  (We set this to ~50 in the formula)")
        print(f"Coefficients:")
        for name, coef in zip(X.columns, model.coef_):
            print(f"  {name}: {coef:.4f}")
        print(f"  (We set study_hours to 3.5 and prior_knowledge to 0.4)")
        print()
        print(f"R-squared Score: {r2_score(y, y_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
        print()
        print("PROBLEM: We're just recovering the parameters we invented!")
        print()
        
        return model
    
    def visualize_results(self):
        """Create visualizations of synthetic data"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SYNTHETIC DATA EXAMPLE - NOT VERIFIABLE RESEARCH', 
                     fontsize=16, fontweight='bold', color='red')
        
        axes[0, 0].scatter(self.data['study_hours'], self.data['test_score'], 
                          alpha=0.6, c=self.data['prior_knowledge'], 
                          cmap='viridis', s=50)
        axes[0, 0].set_xlabel('Study Hours (Fake Data)')
        axes[0, 0].set_ylabel('Test Score (Fake Data)')
        axes[0, 0].set_title('Synthetic: Study Hours vs Test Score')
        axes[0, 0].text(0.5, 0.95, 'WARNING: Not real data', 
                       transform=axes[0, 0].transAxes, 
                       ha='center', va='top', color='red', fontweight='bold')
        
        z = np.polyfit(self.data['study_hours'], self.data['test_score'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.data['study_hours'], p(self.data['study_hours']), 
                       "r--", alpha=0.8, linewidth=2)
        axes[0, 0].grid(True, alpha=0.3)
        
        self.data.boxplot(column='test_score', by='group', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Prior Knowledge Group (Fake)')
        axes[0, 1].set_ylabel('Test Score (Fake)')
        axes[0, 1].set_title('Synthetic: Test Scores by Group')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=0)
        
        axes[1, 0].hist(self.data['test_score'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(self.data['test_score'].mean(), color='r', 
                          linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Test Score (Fake Data)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Synthetic: Distribution of Test Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        corr_matrix = self.data[['study_hours', 'prior_knowledge', 'test_score']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, ax=axes[1, 1], square=True)
        axes[1, 1].set_title('Synthetic: Correlation Matrix')
        
        fig.text(0.5, 0.02, 
                'THIS IS SYNTHETIC DATA - Cannot be verified or peer-reviewed', 
                ha='center', fontsize=12, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('synthetic_example_results.png', dpi=300, bbox_inches='tight')
        print("Synthetic visualizations saved to 'synthetic_example_results.png'")
        plt.close()
        
    def run_analysis(self):
        """Execute synthetic study - FOR COMPARISON ONLY"""
        print("\n" + "=" * 60)
        print("SYNTHETIC DATA STUDY - EDUCATIONAL EXAMPLE ONLY")
        print("=" * 60)
        print("\nWARNING: This demonstrates what NOT to do!")
        print("This data is artificially generated and cannot be verified.")
        print("Compare with verifiable_research.py to see the correct approach.")
        print("=" * 60 + "\n")
        
        print("Generating synthetic (fake) data...")
        self.generate_synthetic_data()
        print(f"Generated {self.n_samples} fake observations\n")
        
        self.descriptive_statistics()
        self.hypothesis_testing()
        model = self.regression_analysis()
        
        print("=" * 60)
        print("WHY THIS APPROACH IS WRONG FOR EMPIRICAL RESEARCH")
        print("=" * 60)
        print("1. Data is not real - just numbers from random.uniform()")
        print("2. Results are predetermined by our formulas")
        print("3. Cannot be independently verified by others")
        print("4. No real-world observations or measurements")
        print("5. Relationships are hardcoded, not discovered")
        print("6. Not suitable for peer review or publication")
        print("7. Makes no contribution to actual knowledge")
        print()
        print("FOR VERIFIABLE RESEARCH: Use verifiable_research.py instead!")
        print("=" * 60 + "\n")
        
        self.visualize_results()
        
        return model


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SYNTHETIC DATA EXAMPLE - EDUCATIONAL PURPOSES ONLY")
    print("="*60)
    print("\nThis script demonstrates INCORRECT empirical research.")
    print("It uses synthetic (made-up) data that cannot be verified.")
    print("\nUse this to understand:")
    print("  - Why synthetic data is not suitable for real research")
    print("  - How predetermined results differ from discoveries")
    print("  - Why verification and reproducibility matter")
    print("\nFor CORRECT verifiable research, see: verifiable_research.py")
    print("="*60 + "\n")
    
    study = SyntheticStudyExample(n_samples=200, random_state=42)
    model = study.run_analysis()
    
    print("\nRemember: This is what NOT to do for empirical research!")
    print("Real research requires real, verifiable data.\n")
