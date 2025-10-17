"""
Research Template for AI Agents

Use this template to structure your research projects using the research_toolkit library.
Replace the placeholder sections with your specific research details.

BEFORE STARTING:
1. Read: guidelines/AI_RESEARCH_GUIDELINES.md
2. Determine your research type using the decision tree
3. Ensure you have appropriate data sources
4. Install research_toolkit: pip install -e .

FEATURES PROVIDED BY research_toolkit:
- APA 7 reference formatting (10 reference types)
- Statistical formatters (9 APA-compliant methods)
- SafeOutput (cross-platform Unicode/ASCII handling)
- ReportFormatter (professional report generation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json

# Import research_toolkit components
from research_toolkit import (
    ReportFormatter,
    SafeOutput,
    StatisticalFormatter,
    APA7ReferenceManager,
    ScientificNotation
)


class ResearchTemplate:
    """
    Template for conducting verifiable research.
    Adapt this for your specific research question.
    """
    
    def __init__(self):
        # Initialize research_toolkit components
        self.formatter = ReportFormatter()
        self.ref_manager = APA7ReferenceManager()
        
        # STEP 1: Define your research metadata
        self.metadata = {
            'research_type': 'CHOOSE: empirical/methodological/theoretical/simulation',
            'study_date': datetime.now().isoformat(),
            'research_question': 'STATE YOUR SPECIFIC RESEARCH QUESTION HERE',
            'hypothesis': 'STATE YOUR TESTABLE HYPOTHESIS (if applicable)',
            'data_sources': [
                'LIST ALL DATA SOURCES WITH URLS',
                'Example: OpenWeatherMap API - https://openweathermap.org/api'
            ],
            'methodology': 'DESCRIBE YOUR STUDY DESIGN',
            'sample_description': 'DESCRIBE YOUR SAMPLE',
            'variables': {
                'independent': 'DESCRIBE INDEPENDENT VARIABLE(S)',
                'dependent': 'DESCRIBE DEPENDENT VARIABLE(S)',
                'control': 'LIST ANY CONTROL VARIABLES'
            },
            'statistical_methods': [
                'LIST METHODS: e.g., Pearson correlation, t-test, regression'
            ],
            'limitations': [
                'LIST ALL LIMITATIONS OF YOUR STUDY',
                'Example: Cross-sectional design',
                'Example: Small sample size',
                'Example: Confounding variables not controlled'
            ],
            'ethical_considerations': 'DESCRIBE ETHICAL ASPECTS',
        }
        
        self.data = None
    
    def collect_data(self):
        """
        STEP 2: Collect or load your data
        
        FOR EMPIRICAL RESEARCH:
        - Use real data from APIs, databases, or files
        - Document source, access method, and timestamp
        - Handle errors transparently
        
        FOR METHODOLOGICAL RESEARCH:
        - Generate synthetic data for method testing
        - Document generation process
        - State clearly this is method testing
        """
        self.formatter.print_section("DATA COLLECTION")
        
        # Example structure - adapt for your needs:
        try:
            # Option 1: Load from API
            # response = requests.get(api_url, params=params)
            # data = response.json()
            
            # Option 2: Load from file
            # data = pd.read_csv('your_data.csv')
            
            # Option 3: Generate for methodological research only
            # data = self._generate_methodological_data()
            
            # REPLACE THIS with your actual data collection:
            print("TODO: Implement data collection")
            print("  1. Identify data source")
            print("  2. Access data")
            print("  3. Document source and timestamp")
            print("  4. Handle errors")
            
            self.data = None  # Replace with your data
            
        except Exception as e:
            print(f"Error collecting data: {e}")
            return None
        
        return self.data
    
    def save_raw_data(self, filename='raw_research_data.csv'):
        """
        STEP 3: Save raw data for verification
        
        REQUIRED FOR EMPIRICAL RESEARCH
        """
        if self.data is not None:
            self.data.to_csv(filename, index=False)
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            SafeOutput.safe_print(f"\n[OK] Raw data saved to: {filename}")
            SafeOutput.safe_print(f"[OK] Metadata saved to: {metadata_file}")
            SafeOutput.safe_print("  -> Anyone can verify these results using the saved data")
        else:
            print("[ERROR] No data to save")
    
    def validate_data(self):
        """
        STEP 4: Validate your data
        
        Check for:
        - Missing values
        - Outliers
        - Data types
        - Expected ranges
        """
        self.formatter.print_section("DATA VALIDATION")
        
        if self.data is None:
            print("[ERROR] No data to validate")
            return False
        
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nMissing values:")
        print(self.data.isnull().sum())
        
        print(f"\nData types:")
        print(self.data.dtypes)
        
        print(f"\nBasic statistics:")
        print(self.data.describe())
        
        # Add your specific validation checks here
        
        return True
    
    def descriptive_statistics(self):
        """
        STEP 5: Calculate descriptive statistics
        """
        self.formatter.print_section("DESCRIPTIVE STATISTICS")
        
        if self.data is None:
            print("[ERROR] No data available")
            return
        
        print(f"\nSample Size: {len(self.data)}")
        print(f"\nDescriptive Statistics:")
        print(self.data.describe())
        
        # Add your specific descriptive analyses here
        # Examples:
        # - Frequency distributions
        # - Correlation matrices
        # - Group summaries
    
    def check_assumptions(self):
        """
        STEP 6: Check statistical assumptions
        
        Before running statistical tests, check:
        - Normality (if required)
        - Homogeneity of variance
        - Independence of observations
        - Linearity (for regression)
        - etc.
        """
        self.formatter.print_section("CHECKING ASSUMPTIONS")
        
        # Example: Check normality
        # for var in numeric_variables:
        #     stat, p = stats.shapiro(self.data[var])
        #     print(f"{var}: Shapiro-Wilk p={p:.4f}")
        
        print("TODO: Implement assumption checks for your analysis")
    
    def hypothesis_testing(self):
        """
        STEP 7: Perform hypothesis tests
        
        IMPORTANT:
        - Report ALL tests (not just significant ones)
        - Report effect sizes and confidence intervals
        - Interpret carefully (no claiming causation from correlation)
        """
        self.formatter.print_section("HYPOTHESIS TESTING")
        
        # Example structure:
        print(f"\nNull Hypothesis (H0): [STATE YOUR NULL HYPOTHESIS]")
        print(f"Alternative Hypothesis (H1): [STATE YOUR ALTERNATIVE]")
        
        # Perform your statistical tests here
        # Examples:
        # - r, p = stats.pearsonr(x, y)
        # - t, p = stats.ttest_ind(group1, group2)
        # - f, p = stats.f_oneway(group1, group2, group3)
        
        print("\nTODO: Implement hypothesis tests")
        print("  1. Choose appropriate test")
        print("  2. Run test")
        print("  3. Report statistics, p-value, effect size")
        print("  4. Interpret carefully")
    
    def regression_analysis(self):
        """
        STEP 8: Perform regression analysis (if applicable)
        
        Report:
        - Coefficients
        - R-squared
        - Residual diagnostics
        - Confidence intervals
        """
        self.formatter.print_section("REGRESSION ANALYSIS")
        
        # Example structure:
        # X = self.data[['independent_var']]
        # y = self.data['dependent_var']
        # model = LinearRegression()
        # model.fit(X, y)
        
        print("TODO: Implement regression analysis if applicable")
    
    def visualize_results(self):
        """
        STEP 9: Create visualizations
        
        Include:
        - Main relationship plots
        - Distribution plots
        - Diagnostic plots
        """
        self.formatter.print_section("VISUALIZATIONS")
        
        # Create your visualizations here
        # Save to file for documentation
        
        print("TODO: Create visualizations")
        print("  1. Main results")
        print("  2. Diagnostics")
        print("  3. Save to file")
    
    def state_limitations(self):
        """
        STEP 10: Explicitly state limitations
        
        REQUIRED: Be honest about what you can and cannot conclude
        """
        self.formatter.print_section("LIMITATIONS")
        
        for limitation in self.metadata['limitations']:
            print(f"  - {limitation}")
        
        print("\nThese limitations mean:")
        print("  - [EXPLAIN IMPLICATIONS]")
        print("  - [WHAT CANNOT BE CONCLUDED]")
        print("  - [FUTURE RESEARCH NEEDED]")
    
    def generate_conclusions(self):
        """
        STEP 11: State conclusions carefully
        
        Match claims to evidence:
        - Don't overstate
        - Acknowledge uncertainty
        - Suggest future research
        """
        self.formatter.print_section("CONCLUSIONS")
        
        print(f"\nResearch Question: {self.metadata['research_question']}")
        print(f"\nFindings:")
        print("  1. [FINDING 1 with statistical support]")
        print("  2. [FINDING 2 with statistical support]")
        print("  3. [etc.]")
        
        print(f"\nInterpretation:")
        print("  - [WHAT THIS MEANS]")
        print("  - [PRACTICAL IMPLICATIONS]")
        
        print(f"\nFuture Research:")
        print("  - [SUGGESTED NEXT STEPS]")
    
    def provide_verification_instructions(self):
        """
        STEP 12: Provide instructions for others to verify your work
        
        REQUIRED FOR EMPIRICAL RESEARCH
        """
        self.formatter.print_section("VERIFICATION INSTRUCTIONS")
        
        print("\nTo reproduce this research:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. [ACCESS DATA: specific instructions]")
        print("  3. Run: python your_research_file.py")
        print("  4. Compare results with raw_research_data.csv")
        print("  5. [ANY ADDITIONAL STEPS]")
        
        print("\nTo verify the analysis:")
        print("  1. Load raw_research_data.csv")
        print("  2. Run the statistical tests independently")
        print("  3. Compare results with reported findings")
    
    def run_full_analysis(self):
        """
        Execute complete research workflow
        """
        self.formatter.print_section(f"RESEARCH PROJECT: {self.metadata['research_question']}")
        SafeOutput.safe_print(f"Study Date: {self.metadata['study_date']}")
        SafeOutput.safe_print(f"Research Type: {self.metadata['research_type']}")
        SafeOutput.safe_print("="*70)
        
        # Step-by-step workflow
        data = self.collect_data()
        
        if data is None:
            print("\n[ERROR] Cannot proceed without data")
            return None
        
        self.save_raw_data()
        
        if not self.validate_data():
            print("\n[WARNING] Data validation issues detected")
        
        self.descriptive_statistics()
        self.check_assumptions()
        self.hypothesis_testing()
        self.regression_analysis()
        self.visualize_results()
        self.state_limitations()
        self.generate_conclusions()
        self.provide_verification_instructions()
        
        self.formatter.print_section("RESEARCH COMPLETE")
        SafeOutput.safe_print("\nRemember:")
        SafeOutput.safe_print("  - All results must be verifiable")
        SafeOutput.safe_print("  - Claims must match evidence")
        SafeOutput.safe_print("  - Limitations must be acknowledged")
        SafeOutput.safe_print("="*70 + "\n")
        
        return self.data


if __name__ == "__main__":
    formatter = ReportFormatter()
    formatter.print_section("RESEARCH TEMPLATE")
    SafeOutput.safe_print("\nThis is a template. To use it:")
    SafeOutput.safe_print("  1. Copy this file")
    SafeOutput.safe_print("  2. Fill in your research details")
    SafeOutput.safe_print("  3. Implement data collection")
    SafeOutput.safe_print("  4. Implement analyses specific to your question")
    SafeOutput.safe_print("  5. Follow the guidelines in: guidelines/AI_RESEARCH_GUIDELINES.md")
    SafeOutput.safe_print("  6. Use research_toolkit library components")
    SafeOutput.safe_print("\nExample: Format statistics with StatisticalFormatter")
    SafeOutput.safe_print("  r, p = stats.pearsonr(x, y)")
    SafeOutput.safe_print("  print(StatisticalFormatter.format_correlation(r, p, n))")
    SafeOutput.safe_print("\nExample: Add APA 7 references with APA7ReferenceManager")
    SafeOutput.safe_print("  manager = APA7ReferenceManager()")
    SafeOutput.safe_print("  key = manager.add_reference('journal', author='Smith, J.', ...)")
    SafeOutput.safe_print("\nExample: Use SafeOutput for cross-platform printing")
    SafeOutput.safe_print("  SafeOutput.safe_print('Temperature: 25 degrees C')")
    SafeOutput.safe_print("="*70 + "\n")
    
    # Example instantiation:
    # research = ResearchTemplate()
    # research.run_full_analysis()
