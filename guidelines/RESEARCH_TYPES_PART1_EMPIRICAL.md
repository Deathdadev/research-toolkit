# Complete Research Types Guide for AI Agents

## Overview

This document provides in-depth explanations and examples of all research types an AI agent can conduct in a Python environment, with clear guidelines on when to use each type and what data requirements apply.

---

## Classification System

### By Data Requirements

**EMPIRICAL** (Real data required)
- Correlational studies
- Comparative studies
- Time series analysis
- Observational studies
- Meta-analysis
- Longitudinal studies
- Cross-sectional studies

**NON-EMPIRICAL** (Synthetic/theoretical data acceptable with caveats)
- Simulation studies
- Methodological studies
- Theoretical models

### By Research Goal

**EXPLORATORY** - Generate hypotheses, explore new areas
**DESCRIPTIVE** - Describe characteristics and patterns
**EXPLANATORY** - Explain relationships and causes
**PREDICTIVE** - Forecast future outcomes
**EVALUATIVE** - Assess effectiveness or quality

---

## 1. CORRELATIONAL STUDY (Empirical)

### Definition
Examines relationships between two or more variables without manipulation, determining if and how strongly they are associated.

### When to Use
- Investigating relationships between naturally occurring variables
- Cannot manipulate variables ethically or practically
- Want to identify patterns for further investigation
- Exploratory phase of research

### Data Requirements
✅ **MUST use real data** from observations or measurements
❌ Cannot use synthetic data for empirical claims

### Key Characteristics
- No manipulation of variables
- Measures naturally occurring relationships
- Cannot establish causation (correlation ≠ causation)
- Can suggest directions for experimental research

### Example Research Questions
- "Is population density correlated with air pollution?"
- "Do GDP and life expectancy correlate across countries?"
- "Is social media usage related to mental health outcomes?"

### Python Implementation Pattern

```python
class CorrelationalStudy:
    """
    Template for correlational research
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Correlational Study (Empirical)',
            'design': 'Non-experimental, observational',
            'data_requirement': 'Real observations required',
            'can_infer_causation': False,
            'statistical_methods': [
                'Pearson correlation (parametric)',
                'Spearman correlation (non-parametric)',
                'Scatterplots',
                'Correlation matrices'
            ]
        }
    
    def collect_data(self):
        """Collect real data from verifiable sources"""
        # Example: API, database, file
        response = requests.get(api_url)
        data = pd.DataFrame(response.json())
        return data
    
    def analyze_correlation(self, x, y):
        """Analyze correlation between variables"""
        # Pearson (assumes normality)
        r_pearson, p_pearson = stats.pearsonr(x, y)
        
        # Spearman (non-parametric)
        r_spearman, p_spearman = stats.spearmanr(x, y)
        
        print(f"Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")
        print(f"Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")
        
        # Interpret strength
        strength = self._interpret_correlation(abs(r_pearson))
        print(f"Correlation strength: {strength}")
        
        return r_pearson, p_pearson
    
    def _interpret_correlation(self, r):
        """Interpret correlation coefficient"""
        if r < 0.1: return "Negligible"
        elif r < 0.3: return "Weak"
        elif r < 0.5: return "Moderate"
        elif r < 0.7: return "Strong"
        else: return "Very strong"
    
    def state_limitations(self):
        """CRITICAL: State limitations"""
        print("LIMITATIONS:")
        print("- Correlation does NOT imply causation")
        print("- Cannot determine direction of effect")
        print("- Third variables may explain relationship")
        print("- Limited to linear relationships (Pearson)")

# Example: Already implemented in examples/01_correlational_study.py
```

### Critical Interpretation Rules
❌ **DON'T say:** "X causes Y"
✅ **DO say:** "X and Y are associated/correlated"
✅ **DO say:** "Higher X is related to higher Y"
✅ **DO say:** "These variables co-vary, but causation cannot be inferred"

---

## 2. COMPARATIVE STUDY (Empirical)

### Definition
Compares two or more groups on one or more variables to identify differences or similarities.

### When to Use
- Comparing pre-existing groups (not randomly assigned)
- Examining differences between categories
- Evaluating group characteristics
- Cross-cultural or cross-national comparisons

### Data Requirements
✅ **MUST use real data** from actual groups
❌ Cannot use made-up group data

### Key Characteristics
- Groups exist naturally (not created by researcher)
- Between-group comparisons
- Can be descriptive or explanatory
- May suggest but not prove causal relationships

### Example Research Questions
- "Do coastal cities differ from inland cities in temperature trends?"
- "Are there differences in health outcomes between regions?"
- "Do countries with different political systems differ in economic growth?"

### Python Implementation Pattern

```python
class ComparativeStudy:
    """
    Template for comparative research
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Comparative Study (Empirical)',
            'design': 'Non-experimental group comparison',
            'data_requirement': 'Real group data required',
            'can_infer_causation': False,  # Groups not randomly assigned
            'statistical_methods': [
                'Independent t-test (2 groups)',
                'ANOVA (3+ groups)',
                'Mann-Whitney U (non-parametric)',
                'Kruskal-Wallis (non-parametric)',
                'Effect sizes (Cohen\'s d)',
                'Box plots',
                'Group descriptives'
            ]
        }
    
    def compare_two_groups(self, group1, group2, var_name):
        """Compare two independent groups"""
        # Descriptive statistics
        print(f"\nComparing {var_name} between groups:")
        print(f"Group 1: M={group1.mean():.2f}, SD={group1.std():.2f}, n={len(group1)}")
        print(f"Group 2: M={group2.mean():.2f}, SD={group2.std():.2f}, n={len(group2)}")
        
        # Check normality
        _, p1 = stats.shapiro(group1)
        _, p2 = stats.shapiro(group2)
        
        if p1 > 0.05 and p2 > 0.05:
            # Use parametric test
            t_stat, p_value = stats.ttest_ind(group1, group2)
            print(f"\nIndependent t-test: t={t_stat:.3f}, p={p_value:.4f}")
            
            # Effect size (Cohen's d)
            cohens_d = self._calculate_cohens_d(group1, group2)
            print(f"Cohen's d = {cohens_d:.3f} ({self._interpret_cohens_d(cohens_d)})")
        else:
            # Use non-parametric test
            u_stat, p_value = stats.mannwhitneyu(group1, group2)
            print(f"\nMann-Whitney U test: U={u_stat:.1f}, p={p_value:.4f}")
        
        return p_value
    
    def compare_multiple_groups(self, *groups, var_name):
        """Compare three or more independent groups"""
        # Check normality for all groups
        normal = all(stats.shapiro(g)[1] > 0.05 for g in groups)
        
        if normal:
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"\nOne-way ANOVA for {var_name}:")
            print(f"F={f_stat:.3f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print("Significant group differences detected")
                print("Consider post-hoc tests (e.g., Tukey HSD)")
        else:
            # Kruskal-Wallis (non-parametric)
            h_stat, p_value = stats.kruskal(*groups)
            print(f"\nKruskal-Wallis test for {var_name}:")
            print(f"H={h_stat:.3f}, p={p_value:.4f}")
        
        return p_value
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        pooled_std = np.sqrt(
            ((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) /
            (len(group1) + len(group2) - 2)
        )
        return (group1.mean() - group2.mean()) / pooled_std
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d"""
        d = abs(d)
        if d < 0.2: return "negligible"
        elif d < 0.5: return "small"
        elif d < 0.8: return "medium"
        else: return "large"
    
    def visualize_comparison(self, data, group_col, value_col):
        """Create comparison visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        data.boxplot(column=value_col, by=group_col, ax=axes[0])
        axes[0].set_title(f'{value_col} by {group_col}')
        
        # Violin plot
        sns.violinplot(data=data, x=group_col, y=value_col, ax=axes[1])
        axes[1].set_title(f'Distribution of {value_col}')
        
        plt.tight_layout()
        plt.savefig('group_comparison.png')
        
    def state_limitations(self):
        """State limitations"""
        print("\nLIMITATIONS:")
        print("- Groups not randomly assigned (selection bias possible)")
        print("- Cannot infer causation from group differences")
        print("- Confounding variables may explain differences")
        print("- Generalization limited to sampled populations")

# Example usage
"""
study = ComparativeStudy()

# Load real data
coastal_cities = load_coastal_temperature_data()
inland_cities = load_inland_temperature_data()

# Compare groups
study.compare_two_groups(
    coastal_cities['temperature'],
    inland_cities['temperature'],
    var_name='Average Temperature'
)

study.state_limitations()
"""
```

### Critical Interpretation Rules
❌ **DON'T say:** "Being in group A causes outcome Y"
✅ **DO say:** "Groups A and B differ significantly on variable Y"
✅ **DO say:** "Group A scored higher than Group B on average"
⚠️ **BE CAREFUL:** Selection bias - groups may differ in unmeasured ways

---

## 3. TIME SERIES ANALYSIS (Empirical)

### Definition
Analyzes data collected over time to identify trends, patterns, seasonality, and make forecasts.

### When to Use
- Examining temporal patterns or trends
- Detecting seasonality or cycles
- Forecasting future values
- Analyzing before-after changes
- Monitoring processes over time

### Data Requirements
✅ **MUST use real time-ordered data**
✅ Data points must be collected at regular intervals
❌ Cannot fabricate temporal patterns

### Key Characteristics
- Sequential data ordered by time
- Accounts for temporal dependencies
- Can reveal trends, seasonality, cycles
- Enables forecasting
- Requires sufficient time points (typically 50+)

### Example Research Questions
- "How has air quality changed over the past decade?"
- "Is there a seasonal pattern in crime rates?"
- "What is the trend in global temperature?"
- "Can we forecast next year's economic indicators?"

### Python Implementation Pattern

```python
class TimeSeriesAnalysis:
    """
    Template for time series research
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Time Series Analysis (Empirical)',
            'design': 'Longitudinal observational',
            'data_requirement': 'Real temporal data required',
            'temporal_ordering': 'Critical',
            'min_observations': 50,  # Rule of thumb
            'statistical_methods': [
                'Trend analysis',
                'Seasonality decomposition',
                'Autocorrelation analysis',
                'ARIMA modeling',
                'Moving averages',
                'Forecasting'
            ]
        }
    
    def collect_temporal_data(self, start_date, end_date, source):
        """Collect time-ordered real data"""
        # Example: Historical data from API
        data = pd.read_csv(source, parse_dates=['date'])
        data = data.sort_values('date')  # Ensure temporal order
        data.set_index('date', inplace=True)
        
        # Verify temporal completeness
        self._check_temporal_completeness(data)
        
        return data
    
    def _check_temporal_completeness(self, data):
        """Check for missing time points"""
        # Check frequency
        inferred_freq = pd.infer_freq(data.index)
        print(f"Detected frequency: {inferred_freq}")
        
        # Check for gaps
        gaps = data.index.to_series().diff()
        if gaps.max() > 2 * gaps.median():
            print("WARNING: Gaps detected in time series")
    
    def decompose_series(self, data, period=12):
        """Decompose into trend, seasonal, and residual"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(
            data,
            model='additive',  # or 'multiplicative'
            period=period
        )
        
        # Plot components
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        data.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        plt.savefig('time_series_decomposition.png')
        
        return decomposition
    
    def test_stationarity(self, data):
        """Test if series is stationary (required for some models)"""
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(data.dropna())
        
        print("Augmented Dickey-Fuller Test:")
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        
        if result[1] < 0.05:
            print("Series is stationary")
        else:
            print("Series is non-stationary (consider differencing)")
        
        return result[1] < 0.05
    
    def analyze_trend(self, data, var_name):
        """Analyze linear trend"""
        # Create time index
        x = np.arange(len(data))
        y = data.values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        print(f"\nTrend Analysis for {var_name}:")
        print(f"Slope: {slope:.4f} per time unit")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
            print(f"Significant {direction} trend detected")
        else:
            print("No significant linear trend")
        
        return slope, p_value
    
    def forecast_arima(self, data, steps=12):
        """Forecast using ARIMA model"""
        from statsmodels.tsa.arima.model import ARIMA
        
        # Fit ARIMA (p,d,q) - simplified, use auto_arima for best params
        model = ARIMA(data, order=(1, 1, 1))
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.forecast(steps=steps)
        
        print(f"\nForecast for next {steps} periods:")
        print(forecast)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data.values, label='Historical')
        forecast_index = pd.date_range(
            start=data.index[-1],
            periods=steps+1,
            freq=data.index.freq
        )[1:]
        plt.plot(forecast_index, forecast, 'r--', label='Forecast')
        plt.legend()
        plt.title('Time Series Forecast')
        plt.savefig('forecast.png')
        
        return forecast
    
    def detect_changepoint(self, data):
        """Detect significant changes in time series"""
        # Simple changepoint detection using CUSUM
        cumsum = (data - data.mean()).cumsum()
        changepoint_idx = cumsum.abs().idxmax()
        
        print(f"\nPotential changepoint detected at: {changepoint_idx}")
        
        return changepoint_idx
    
    def state_limitations(self):
        """State limitations"""
        print("\nLIMITATIONS:")
        print("- Past patterns may not continue (non-stationarity)")
        print("- External events can disrupt patterns")
        print("- Forecasts become less reliable further into future")
        print("- Cannot establish causation from temporal patterns alone")
        print("- Missing data or irregular intervals affect validity")

# Example usage
"""
study = TimeSeriesAnalysis()

# Load real temporal data
pollution_data = pd.read_csv('historical_pollution.csv',
                             parse_dates=['date'],
                             index_col='date')

# Analyze
study.test_stationarity(pollution_data['pm25'])
study.analyze_trend(pollution_data['pm25'], 'PM2.5')
decomposition = study.decompose_series(pollution_data['pm25'])
forecast = study.forecast_arima(pollution_data['pm25'], steps=12)

study.state_limitations()
"""
```

### Critical Considerations
- **Autocorrelation**: Time series data points are often correlated (violates independence assumption)
- **Seasonality**: Must account for repeating patterns
- **Trend vs. Cycle**: Distinguish long-term trends from cyclical patterns
- **External Shocks**: Unexpected events can break patterns
- **Forecast Uncertainty**: Always provide confidence intervals

---

## 4. OBSERVATIONAL STUDY (Empirical - Descriptive)

### Definition
Systematically observes and records phenomena as they naturally occur without intervention or manipulation.

### When to Use
- Describing current state or characteristics
- Cannot or should not manipulate variables
- Exploratory phase before experiments
- Studying natural behavior or phenomena
- Ethical constraints prevent manipulation

### Data Requirements
✅ **MUST use real observational data**
✅ Systematic and structured observation
❌ Cannot invent observations

### Key Characteristics
- No manipulation or intervention
- Describes "what is"
- Can be quantitative or qualitative
- Foundation for generating hypotheses
- Cannot establish causation

### Example Research Questions
- "What are the characteristics of high-performing companies?"
- "What is the distribution of programming languages in GitHub?"
- "What are the demographics of social media users?"
- "What patterns exist in urban development?"

### Python Implementation Pattern

```python
class ObservationalStudy:
    """
    Template for observational/descriptive research
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Observational Study (Empirical - Descriptive)',
            'design': 'Non-experimental, descriptive',
            'data_requirement': 'Real observations required',
            'can_infer_causation': False,
            'purpose': 'Describe characteristics, patterns, distributions',
            'statistical_methods': [
                'Descriptive statistics',
                'Frequency distributions',
                'Cross-tabulations',
                'Visualization',
                'Pattern identification'
            ]
        }
    
    def collect_observations(self, source, variables):
        """Collect systematic observations"""
        # Example: Real data from database, API, or file
        data = pd.read_csv(source)
        
        print(f"Collected {len(data)} observations")
        print(f"Variables: {list(data.columns)}")
        
        # Data quality check
        self._check_data_quality(data)
        
        return data
    
    def _check_data_quality(self, data):
        """Check observation quality"""
        print("\nData Quality Check:")
        print(f"Complete cases: {data.dropna().shape[0]} ({data.dropna().shape[0]/len(data)*100:.1f}%)")
        print("\nMissing data:")
        print(data.isnull().sum())
    
    def describe_sample(self, data):
        """Comprehensive descriptive analysis"""
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS")
        print("="*70)
        
        # Numeric variables
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumeric Variables:")
            print(data[numeric_cols].describe())
            
            # Additional statistics
            for col in numeric_cols:
                print(f"\n{col}:")
                print(f"  Median: {data[col].median():.2f}")
                print(f"  Mode: {data[col].mode().values[0] if len(data[col].mode()) > 0 else 'N/A'}")
                print(f"  Range: {data[col].min():.2f} to {data[col].max():.2f}")
                print(f"  IQR: {data[col].quantile(0.75) - data[col].quantile(0.25):.2f}")
                
                # Check for outliers
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
                print(f"  Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
        
        # Categorical variables
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print("\nCategorical Variables:")
            for col in categorical_cols:
                print(f"\n{col}:")
                print(data[col].value_counts())
                print(f"  Unique values: {data[col].nunique()}")
    
    def analyze_distributions(self, data, var):
        """Analyze distribution of a variable"""
        print(f"\nDistribution Analysis: {var}")
        
        # Test normality
        if data[var].dtype in [np.float64, np.int64]:
            stat, p = stats.shapiro(data[var].dropna())
            print(f"Shapiro-Wilk test: p={p:.4f}")
            if p > 0.05:
                print("  Distribution appears normal")
            else:
                print("  Distribution is non-normal")
            
            # Skewness and kurtosis
            skew = stats.skew(data[var].dropna())
            kurt = stats.kurtosis(data[var].dropna())
            print(f"Skewness: {skew:.3f}")
            print(f"Kurtosis: {kurt:.3f}")
    
    def identify_patterns(self, data):
        """Identify patterns and relationships"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            print("\nCorrelation Matrix:")
            corr_matrix = data[numeric_cols].corr()
            print(corr_matrix)
            
            # Visualize
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png')
            
            # Identify strong correlations
            print("\nStrong correlations (|r| > 0.5):")
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: r={corr_matrix.iloc[i, j]:.3f}")
    
    def visualize_observations(self, data):
        """Create visualizations of observations"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Distributions
        fig, axes = plt.subplots(
            (len(numeric_cols) + 1) // 2,
            2,
            figsize=(14, 5 * ((len(numeric_cols) + 1) // 2))
        )
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            data[col].hist(bins=30, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('distributions.png')
    
    def state_limitations(self):
        """State limitations"""
        print("\nLIMITATIONS:")
        print("- Descriptive only - no causal inferences")
        print("- Limited to observed sample - may not generalize")
        print("- Patterns observed may be due to confounding")
        print("- Cannot test hypotheses about causation")
        print("- Temporal sequence unclear")

# Example usage
"""
study = ObservationalStudy()

# Collect real observations
github_data = study.collect_observations(
    'github_repos.csv',
    ['language', 'stars', 'forks', 'contributors']
)

# Describe what was observed
study.describe_sample(github_data)
study.identify_patterns(github_data)
study.visualize_observations(github_data)

study.state_limitations()
"""
```

### Critical Points
- **No manipulation** - purely descriptive
- **Hypothesis generation** - can suggest ideas to test experimentally
- **Cannot infer causation** - only describe what exists
- **Sampling matters** - representativeness affects generalization

---

## 5. META-ANALYSIS (Empirical - Synthesis)

### Definition
Quantitatively combines results from multiple independent studies on the same topic to derive overall conclusions.

### When to Use
- Synthesizing existing research evidence
- Resolving conflicting findings
- Increasing statistical power
- Identifying overall effect sizes
- Finding moderators of effects

### Data Requirements
✅ **MUST use real data from published studies**
✅ Requires effect sizes or statistics from each study
✅ Need study characteristics (sample size, methods, etc.)
❌ Cannot invent study results

### Key Characteristics
- Systematic literature review first
- Quantitative synthesis of effects
- Weighted by study quality/size
- Assesses heterogeneity
- More powerful than single studies
- "Study of studies"

### Example Research Questions
- "What is the overall effect of intervention X across all studies?"
- "Do study characteristics moderate the effect?"
- "Is there publication bias in this literature?"
- "What is the average correlation reported across studies?"

### Python Implementation Pattern

```python
class MetaAnalysis:
    """
    Template for meta-analytic research
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Meta-Analysis (Empirical - Synthesis)',
            'design': 'Quantitative synthesis of existing studies',
            'data_requirement': 'Real effect sizes from published research',
            'level_of_analysis': 'Study-level (not individual participants)',
            'statistical_methods': [
                'Fixed-effects model',
                'Random-effects model',
                'Heterogeneity analysis (I², Q)',
                'Publication bias tests',
                'Meta-regression',
                'Forest plots',
                'Funnel plots'
            ]
        }
        self.studies = []
    
    def collect_studies(self, search_strategy):
        """
        Systematic literature search and data extraction
        
        CRITICAL: This must be systematic and transparent
        """
        print("SYSTEMATIC LITERATURE REVIEW")
        print("="*70)
        print(f"Search strategy: {search_strategy}")
        
        # Example structure for extracted data
        studies = pd.DataFrame({
            'study_id': [],
            'author': [],
            'year': [],
            'sample_size': [],
            'effect_size': [],  # e.g., Cohen's d, correlation r
            'std_error': [],
            'outcome_measure': [],
            'population': [],
            'study_quality': []
        })
        
        # In practice: Extract from databases like PubMed, PsycINFO, etc.
        # Use APIs or manual extraction
        
        print(f"\nIncluded studies: {len(studies)}")
        return studies
    
    def calculate_effect_sizes(self, studies):
        """
        Calculate or convert effect sizes to common metric
        """
        # Example: Converting r to Fisher's z
        if 'correlation_r' in studies.columns:
            studies['fishers_z'] = 0.5 * np.log((1 + studies['correlation_r']) / 
                                                 (1 - studies['correlation_r']))
            studies['z_se'] = 1 / np.sqrt(studies['sample_size'] - 3)
        
        # Example: Cohen's d with standard error
        if 'cohens_d' in studies.columns:
            studies['d_se'] = np.sqrt(
                (studies['n1'] + studies['n2']) / (studies['n1'] * studies['n2']) +
                studies['cohens_d']**2 / (2 * (studies['n1'] + studies['n2']))
            )
        
        return studies
    
    def fixed_effects_model(self, effect_sizes, std_errors):
        """
        Fixed-effects meta-analysis
        Assumes one true effect size
        """
        # Weights (inverse variance)
        weights = 1 / (std_errors ** 2)
        
        # Weighted mean effect
        pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        
        # Standard error
        pooled_se = np.sqrt(1 / np.sum(weights))
        
        # 95% CI
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se
        
        # Z-test
        z_score = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("\nFIXED-EFFECTS MODEL")
        print(f"Pooled effect size: {pooled_effect:.3f}")
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"Z = {z_score:.3f}, p = {p_value:.6f}")
        
        return pooled_effect, pooled_se, ci_lower, ci_upper
    
    def random_effects_model(self, effect_sizes, std_errors):
        """
        Random-effects meta-analysis
        Assumes distribution of true effect sizes
        """
        # First get fixed-effects estimate
        weights_fixed = 1 / (std_errors ** 2)
        fixed_effect = np.sum(weights_fixed * effect_sizes) / np.sum(weights_fixed)
        
        # Calculate Q statistic
        Q = np.sum(weights_fixed * (effect_sizes - fixed_effect) ** 2)
        df = len(effect_sizes) - 1
        
        # Estimate between-study variance (τ²) - DerSimonian & Laird method
        C = np.sum(weights_fixed) - np.sum(weights_fixed ** 2) / np.sum(weights_fixed)
        tau_squared = max(0, (Q - df) / C)
        
        # New weights including between-study variance
        weights_random = 1 / (std_errors ** 2 + tau_squared)
        
        # Pooled effect
        pooled_effect = np.sum(weights_random * effect_sizes) / np.sum(weights_random)
        pooled_se = np.sqrt(1 / np.sum(weights_random))
        
        # 95% CI
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se
        
        # Z-test
        z_score = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("\nRANDOM-EFFECTS MODEL")
        print(f"Pooled effect size: {pooled_effect:.3f}")
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"τ² (between-study variance): {tau_squared:.4f}")
        print(f"Z = {z_score:.3f}, p = {p_value:.6f}")
        
        return pooled_effect, pooled_se, ci_lower, ci_upper, tau_squared
    
    def assess_heterogeneity(self, effect_sizes, std_errors):
        """
        Assess heterogeneity across studies
        """
        weights = 1 / (std_errors ** 2)
        fixed_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        
        # Q statistic
        Q = np.sum(weights * (effect_sizes - fixed_effect) ** 2)
        df = len(effect_sizes) - 1
        p_value = 1 - stats.chi2.cdf(Q, df)
        
        # I² statistic (percentage of variance due to heterogeneity)
        I_squared = max(0, ((Q - df) / Q) * 100)
        
        print("\nHETEROGENEITY ANALYSIS")
        print(f"Q = {Q:.3f}, df = {df}, p = {p_value:.4f}")
        print(f"I² = {I_squared:.1f}%")
        
        if I_squared < 25:
            print("Interpretation: Low heterogeneity")
        elif I_squared < 50:
            print("Interpretation: Moderate heterogeneity")
        elif I_squared < 75:
            print("Interpretation: Substantial heterogeneity")
        else:
            print("Interpretation: Considerable heterogeneity")
        
        if I_squared > 50:
            print("Consider: Meta-regression to identify moderators")
        
        return Q, I_squared
    
    def test_publication_bias(self, effect_sizes, std_errors):
        """
        Test for publication bias (file drawer problem)
        """
        # Egger's regression test
        precision = 1 / std_errors
        
        # Regression: effect_size ~ precision
        slope, intercept, r, p_value, se = stats.linregress(precision, effect_sizes)
        
        print("\nPUBLICATION BIAS TESTS")
        print(f"Egger's test: intercept = {intercept:.3f}, p = {p_value:.4f}")
        
        if p_value < 0.05:
            print("WARNING: Significant publication bias detected")
            print("Small studies show different effects than large studies")
        else:
            print("No significant evidence of publication bias")
    
    def create_forest_plot(self, studies, effect_col, se_col):
        """
        Create forest plot visualization
        """
        fig, ax = plt.subplots(figsize=(10, len(studies) * 0.5 + 2))
        
        y_pos = range(len(studies))
        
        # Plot individual studies
        for i, (idx, row) in enumerate(studies.iterrows()):
            effect = row[effect_col]
            se = row[se_col]
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se
            
            # Point estimate
            ax.plot(effect, i, 'ko', markersize=8)
            
            # CI
            ax.plot([ci_lower, ci_upper], [i, i], 'k-', linewidth=2)
            
            # Study label
            label = f"{row['author']} ({row['year']})"
            ax.text(-0.1, i, label, ha='right', va='center')
        
        # Reference line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([''] * len(studies))
        ax.set_xlabel('Effect Size')
        ax.set_title('Forest Plot: Individual Study Effects')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('forest_plot.png', dpi=300, bbox_inches='tight')
        
    def create_funnel_plot(self, effect_sizes, std_errors):
        """
        Create funnel plot to visualize publication bias
        """
        plt.figure(figsize=(10, 8))
        
        # Plot studies
        plt.scatter(effect_sizes, 1/std_errors, alpha=0.6, edgecolors='black')
        
        # Reference line at pooled effect
        pooled_effect = np.sum((1/std_errors**2) * effect_sizes) / np.sum(1/std_errors**2)
        plt.axvline(x=pooled_effect, color='r', linestyle='--', label='Pooled Effect')
        
        plt.xlabel('Effect Size')
        plt.ylabel('Precision (1 / Standard Error)')
        plt.title('Funnel Plot: Publication Bias Assessment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('funnel_plot.png', dpi=300, bbox_inches='tight')
    
    def meta_regression(self, studies, effect_col, moderator_col):
        """
        Meta-regression to identify moderators
        """
        X = studies[[moderator_col]].values
        y = studies[effect_col].values
        weights = 1 / (studies['std_error'].values ** 2)
        
        # Weighted regression
        model = LinearRegression()
        model.fit(X, y, sample_weight=weights)
        
        print(f"\nMETA-REGRESSION: {moderator_col} as moderator")
        print(f"Intercept: {model.intercept_:.3f}")
        print(f"Slope: {model.coef_[0]:.3f}")
        
        # R² (proportion of heterogeneity explained)
        y_pred = model.predict(X)
        residuals = y - y_pred
        Q_residual = np.sum(weights * residuals ** 2)
        Q_total = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        R_squared = 1 - (Q_residual / Q_total)
        
        print(f"R² = {R_squared:.3f} ({R_squared*100:.1f}% of heterogeneity explained)")
    
    def state_limitations(self):
        """State limitations"""
        print("\nLIMITATIONS:")
        print("- Quality depends on quality of included studies")
        print("- Publication bias may affect results")
        print("- Heterogeneity may limit interpretability")
        print("- Study-level analysis (ecological fallacy possible)")
        print("- Cannot establish causation (depends on primary studies)")
        print("- 'Apples and oranges' problem if studies too different")

# Example usage
"""
meta = MetaAnalysis()

# Collect studies (from systematic review)
studies = pd.DataFrame({
    'author': ['Smith et al.', 'Jones et al.', ...],
    'year': [2020, 2021, ...],
    'sample_size': [100, 150, ...],
    'correlation_r': [0.30, 0.45, ...],
    'study_quality': [8, 9, ...]
})

# Calculate effect sizes
studies = meta.calculate_effect_sizes(studies)

# Conduct meta-analysis
meta.fixed_effects_model(studies['fishers_z'], studies['z_se'])
meta.random_effects_model(studies['fishers_z'], studies['z_se'])

# Assess heterogeneity
meta.assess_heterogeneity(studies['fishers_z'], studies['z_se'])

# Check publication bias
meta.test_publication_bias(studies['fishers_z'], studies['z_se'])

# Visualize
meta.create_forest_plot(studies, 'fishers_z', 'z_se')
meta.create_funnel_plot(studies['fishers_z'], studies['z_se'])

meta.state_limitations()
"""
```

### Critical Considerations
- **Systematic review first**: Must be comprehensive and transparent
- **Inclusion criteria**: Clear and justified
- **Publication bias**: Major threat to validity
- **Heterogeneity**: High I² suggests studies differ substantially
- **Quality assessment**: Weight studies by quality/risk of bias
- **PRISMA guidelines**: Follow reporting standards

---

## Summary: Empirical Research Types

All five research types covered in this document share critical characteristics:

✅ **REQUIRE real data** - Cannot use synthetic/made-up data
✅ **Cannot establish causation** - All are observational/correlational
✅ **Must document data sources** - Full transparency required
✅ **Must state limitations** - Explicit about what cannot be concluded
✅ **Enable verification** - Others can check your work

### Quick Comparison

| Type | Purpose | Data | Key Strength | Key Limitation |
|------|---------|------|--------------|----------------|
| Correlational | Explore relationships | Real observations | Identify associations | No causation |
| Comparative | Compare groups | Real groups | Show differences | Groups not random |
| Time Series | Analyze trends | Temporal data | Detect patterns | Temporal ≠ causal |
| Observational | Describe phenomena | Observations | Rich description | Descriptive only |
| Meta-Analysis | Synthesize research | Published studies | High power | Depends on primaries |

---

## Next Steps

**See Part 2** (`RESEARCH_TYPES_PART2_NON_EMPIRICAL.md`) for:
- Simulation Studies (model-based, conditional claims)
- Methodological Studies (testing methods - synthetic OK)
- Theoretical Models (pure theory development)

**See Examples** in `examples/` directory for working implementations

**Use Template** in `templates/research_template.py` to create new studies

---

## Remember

**Empirical research = Making claims about the observable world**

This REQUIRES:
- Real data from actual observations
- Transparent methodology
- Saved raw data for verification
- Honest limitations
- Appropriate interpretation

Never use synthetic data to make empirical claims about reality.
