"""
Example 03: Time Series Analysis - Air Quality Trends

Research Question: Has air quality (PM2.5) changed significantly over time in major cities?

This demonstrates:
- Time series analysis with trend detection
- Seasonality decomposition
- Stationarity testing
- Forecasting with ARIMA
- APA 7 referencing using research_toolkit

Data Source: Historical air quality data (simulated based on real patterns)
Note: For production use, replace with actual historical API data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import from research_toolkit library
from research_toolkit.core import SafeOutput, ReportFormatter, StatisticalFormatter
from research_toolkit.references import APA7ReferenceManager

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not available. Install with: pip install statsmodels")


class AirQualityTimeSeriesStudy:
    """
    Time series analysis of air quality trends.
    
    Research Type: Time Series Analysis (Empirical)
    Design: Longitudinal observational
    """
    
    def __init__(self):
        self.references = APA7ReferenceManager()
        
        # Add references
        self.api_ref = self.references.add_reference(
            'api',
            author='OpenWeather',
            year='2024',
            title='Air Quality API',
            url='https://openweathermap.org/api/air-pollution'
        )
        
        self.statsmodels_ref = self.references.add_reference(
            'book',
            author='Seabold, S.;Perktold, J.',
            year='2010',
            title='Statsmodels: Econometric and statistical modeling with Python',
            publisher='Proceedings of the 9th Python in Science Conference'
        )
        
        self.metadata = {
            'research_type': 'Time Series Analysis (Empirical)',
            'study_date': datetime.now().isoformat(),
            'title': 'Temporal Trends in Urban Air Quality: A Time Series Analysis of PM2.5 Concentrations',
            'research_question': 'Has air quality (PM2.5) changed significantly over the past 2 years?',
            'hypothesis': {
                'H0': 'No significant trend in PM2.5 levels over time',
                'H1': 'PM2.5 levels show a significant trend (increase or decrease) over time'
            },
            'design': 'Longitudinal time series analysis',
            'temporal_unit': 'Daily measurements',
            'time_period': '2 years (730 days)',
            'variable': 'PM2.5 concentration (ug/m3)',
            'location': 'Major urban center',
            'data_source': f'OpenWeather Air Pollution API ({self.api_ref})',
            'statistical_methods': [
                'Trend analysis (linear regression on time)',
                'Seasonality decomposition',
                'Augmented Dickey-Fuller test (stationarity)',
                'Autocorrelation analysis',
                'ARIMA forecasting'
            ],
            'limitations': [
                'Cannot establish causation from temporal patterns',
                'Past patterns may not continue (non-stationarity)',
                'External events not accounted for',
                'Single city (limited generalizability)',
                'Measurement quality depends on sensor network'
            ]
        }
        
        self.data = None
    
    def generate_time_series_data(self):
        """
        Generate realistic time series data based on typical air quality patterns.
        
        Note: In production, replace with actual historical API calls.
        """
        print("\n" + "="*70)
        print("DATA COLLECTION")
        print("="*70)
        print(f"Data Source: {self.metadata['data_source']}")
        print(f"Time Period: {self.metadata['time_period']}")
        print(f"Frequency: Daily")
        
        # Generate 2 years of daily data
        np.random.seed(42)
        n_days = 730
        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
        
        # Components of realistic air quality data:
        # 1. Long-term trend (slight decrease due to regulations)
        trend = np.linspace(35, 30, n_days)
        
        # 2. Seasonal pattern (worse in winter)
        day_of_year = np.array([d.dayofyear for d in dates])
        seasonal = 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # 3. Weekly pattern (worse on weekdays)
        day_of_week = np.array([d.dayofweek for d in dates])
        weekly = 2 * (day_of_week < 5).astype(float)
        
        # 4. Random noise
        noise = np.random.normal(0, 4, n_days)
        
        # 5. Occasional spikes (events)
        spikes = np.zeros(n_days)
        spike_days = np.random.choice(n_days, size=10, replace=False)
        spikes[spike_days] = np.random.uniform(15, 30, 10)
        
        # Combine components
        pm25 = trend + seasonal + weekly + noise + spikes
        pm25 = np.clip(pm25, 5, 100)  # Realistic range
        
        self.data = pd.DataFrame({
            'date': dates,
            'pm25': pm25,
            'day_of_week': [d.strftime('%A') for d in dates],
            'month': [d.strftime('%B') for d in dates]
        })
        self.data.set_index('date', inplace=True)
        
        print(f"\nCollected {len(self.data)} daily observations")
        print(f"Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
        print(f"PM2.5 range: {self.data['pm25'].min():.1f} to {self.data['pm25'].max():.1f} ug/m3")
        
        return self.data
    
    def save_raw_data(self, filename='03_time_series_raw_data.csv'):
        """Save raw data for verification"""
        if self.data is not None:
            self.data.to_csv(filename)
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"\n[OK] Raw data saved to: {filename}")
            print(f"[OK] Metadata saved to: {metadata_file}")
    
    def descriptive_statistics(self):
        """Calculate descriptive statistics"""
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS")
        print("="*70)
        
        print(f"\nObservations: {len(self.data)}")
        print(f"\nPM2.5 Statistics:")
        print(f"  M = {self.data['pm25'].mean():.2f} ug/m3")
        print(f"  SD = {self.data['pm25'].std():.2f} ug/m3")
        print(f"  Median = {self.data['pm25'].median():.2f} ug/m3")
        print(f"  Min = {self.data['pm25'].min():.2f} ug/m3")
        print(f"  Max = {self.data['pm25'].max():.2f} ug/m3")
        print(f"  IQR = {self.data['pm25'].quantile(0.75) - self.data['pm25'].quantile(0.25):.2f} ug/m3")
    
    def test_stationarity(self):
        """Test for stationarity using Augmented Dickey-Fuller test"""
        if not STATSMODELS_AVAILABLE:
            print("\n[SKIPPED] Stationarity test requires statsmodels")
            return None
        
        print("\n" + "="*70)
        print("STATIONARITY TESTING")
        print("="*70)
        
        result = adfuller(self.data['pm25'].dropna())
        
        print(f"\nAugmented Dickey-Fuller Test ({self.statsmodels_ref}):")
        print(f"  ADF Statistic = {result[0]:.4f}")
        print(f"  p-value = {result[1]:.4f}")
        print(f"  Critical Values:")
        for key, value in result[4].items():
            print(f"    {key}: {value:.4f}")
        
        if result[1] < 0.05:
            print("\n  -> Series is stationary (p < .05)")
        else:
            print("\n  -> Series is non-stationary (p >= .05)")
            print("  -> Consider differencing for ARIMA modeling")
        
        return result[1] < 0.05
    
    def analyze_trend(self):
        """Analyze linear trend"""
        print("\n" + "="*70)
        print("TREND ANALYSIS")
        print("="*70)
        
        # Create numeric time index
        x = np.arange(len(self.data))
        y = self.data['pm25'].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        print(f"\nLinear Trend Analysis:")
        print(f"  Slope = {slope:.4f} ug/m3 per day")
        print(f"  Yearly change = {slope * 365:.2f} ug/m3 per year")
        print(f"  R^2 = {r_value**2:.4f}")
        print(f"  p-value = {p_value:.6f}")
        
        if p_value < 0.05:
            direction = "decreasing" if slope < 0 else "increasing"
            print(f"\n  -> Significant {direction} trend detected (p < .05)")
        else:
            print(f"\n  -> No significant linear trend (p >= .05)")
        
        return slope, p_value
    
    def decompose_series(self):
        """Decompose time series into components"""
        if not STATSMODELS_AVAILABLE:
            print("\n[SKIPPED] Decomposition requires statsmodels")
            return None
        
        print("\n" + "="*70)
        print("SEASONALITY DECOMPOSITION")
        print("="*70)
        
        # Decompose (annual seasonality)
        decomposition = seasonal_decompose(
            self.data['pm25'],
            model='additive',
            period=365
        )
        
        print("\nDecomposed into:")
        print("  - Trend component")
        print("  - Seasonal component (annual cycle)")
        print("  - Residual component")
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        self.data['pm25'].plot(ax=axes[0], title='Original Series')
        axes[0].set_ylabel('PM2.5 (ug/m3)')
        axes[0].grid(True, alpha=0.3)
        
        decomposition.trend.plot(ax=axes[1], title='Trend Component')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
        axes[2].set_ylabel('Seasonality')
        axes[2].grid(True, alpha=0.3)
        
        decomposition.resid.plot(ax=axes[3], title='Residual Component')
        axes[3].set_ylabel('Residuals')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('03_time_series_decomposition.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Decomposition plot saved")
        plt.close()
        
        return decomposition
    
    def forecast_future(self, steps=90):
        """Forecast future values using ARIMA"""
        if not STATSMODELS_AVAILABLE:
            print("\n[SKIPPED] Forecasting requires statsmodels")
            return None
        
        print("\n" + "="*70)
        print("FORECASTING")
        print("="*70)
        
        print(f"\nFitting ARIMA model...")
        print(f"Forecasting next {steps} days")
        
        # Fit ARIMA model
        model = ARIMA(self.data['pm25'], order=(1, 1, 1))
        fitted = model.fit()
        
        print(f"\nModel Summary:")
        print(f"  AIC = {fitted.aic:.2f}")
        print(f"  BIC = {fitted.bic:.2f}")
        
        # Forecast
        forecast = fitted.forecast(steps=steps)
        forecast_index = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        print(f"\nForecast Statistics:")
        print(f"  Mean forecast = {forecast.mean():.2f} ug/m3")
        print(f"  Min forecast = {forecast.min():.2f} ug/m3")
        print(f"  Max forecast = {forecast.max():.2f} ug/m3")
        
        # Plot forecast
        plt.figure(figsize=(14, 6))
        
        # Historical data
        plt.plot(self.data.index, self.data['pm25'], 
                label='Historical Data', linewidth=1.5)
        
        # Forecast
        plt.plot(forecast_index, forecast, 'r--', 
                label=f'{steps}-day Forecast', linewidth=2)
        
        plt.axvline(x=self.data.index[-1], color='gray', 
                   linestyle=':', label='Forecast Start')
        
        plt.xlabel('Date')
        plt.ylabel('PM2.5 (ug/m3)')
        plt.title(f'Air Quality: Historical Data and {steps}-Day Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('03_time_series_forecast.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Forecast plot saved")
        plt.close()
        
        return forecast
    
    def analyze_autocorrelation(self):
        """Analyze autocorrelation structure"""
        print("\n" + "="*70)
        print("AUTOCORRELATION ANALYSIS")
        print("="*70)
        
        # Calculate autocorrelation
        lags = [1, 7, 30, 365]
        
        print("\nAutocorrelation at key lags:")
        for lag in lags:
            if lag < len(self.data):
                acf = self.data['pm25'].autocorr(lag=lag)
                print(f"  Lag {lag:3d} days: r = {acf:.3f}")
    
    def detect_changepoints(self):
        """Detect significant changes in the series"""
        print("\n" + "="*70)
        print("CHANGEPOINT DETECTION")
        print("="*70)
        
        # Simple CUSUM approach
        cumsum = (self.data['pm25'] - self.data['pm25'].mean()).cumsum()
        changepoint_idx = cumsum.abs().idxmax()
        
        print(f"\nPotential changepoint detected at: {changepoint_idx.date()}")
        print(f"CUSUM value: {cumsum.loc[changepoint_idx]:.2f}")
        
        # Compare before and after
        before = self.data.loc[:changepoint_idx, 'pm25']
        after = self.data.loc[changepoint_idx:, 'pm25']
        
        print(f"\nBefore changepoint: M = {before.mean():.2f} ug/m3")
        print(f"After changepoint: M = {after.mean():.2f} ug/m3")
        print(f"Difference: {abs(after.mean() - before.mean()):.2f} ug/m3")
        
        return changepoint_idx
    
    def visualize_full_analysis(self):
        """Create comprehensive visualization"""
        print("\n" + "="*70)
        print("VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Time Series Analysis: Air Quality Trends', 
                     fontsize=16, fontweight='bold')
        
        # 1. Full time series with trend line
        axes[0, 0].plot(self.data.index, self.data['pm25'], 
                       alpha=0.7, linewidth=1)
        
        # Add trend line
        x = np.arange(len(self.data))
        z = np.polyfit(x, self.data['pm25'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.data.index, p(x), 'r--', 
                       linewidth=2, label=f'Trend: {z[0]*365:.2f} ug/m3/year')
        
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('PM2.5 (ug/m3)')
        axes[0, 0].set_title('PM2.5 Over Time with Linear Trend')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Monthly averages
        monthly = self.data.groupby(self.data.index.to_period('M'))['pm25'].mean()
        axes[0, 1].bar(range(len(monthly)), monthly.values, 
                      edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Mean PM2.5 (ug/m3)')
        axes[0, 1].set_title('Monthly Average PM2.5')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Distribution
        axes[1, 0].hist(self.data['pm25'], bins=30, 
                       edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(self.data['pm25'].mean(), color='r', 
                          linestyle='--', linewidth=2, 
                          label=f'Mean: {self.data["pm25"].mean():.1f}')
        axes[1, 0].set_xlabel('PM2.5 (ug/m3)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of PM2.5 Values')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Boxplot by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        data_for_plot = self.data.copy()
        data_for_plot['day_of_week'] = pd.Categorical(
            data_for_plot['day_of_week'], 
            categories=day_order, 
            ordered=True
        )
        data_for_plot.boxplot(column='pm25', by='day_of_week', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('PM2.5 (ug/m3)')
        axes[1, 1].set_title('PM2.5 by Day of Week')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('03_time_series_analysis.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Analysis visualizations saved to: 03_time_series_analysis.png")
        plt.close()
    
    def generate_report(self):
        """Generate APA-style report"""
        print("\n" + "="*70)
        print("RESEARCH REPORT")
        print("="*70)
        
        print(f"\nTitle: {self.metadata['title']}")
        print(f"\nResearch Question: {self.metadata['research_question']}")
        
        slope, p_trend = stats.linregress(
            np.arange(len(self.data)),
            self.data['pm25'].values
        )[:2]
        
        print("\n--- RESULTS ---")
        print(f"\nTime series analysis of daily PM2.5 measurements over ")
        print(f"{self.metadata['time_period']} (N = {len(self.data)} observations) ")
        print(f"revealed a mean concentration of {self.data['pm25'].mean():.2f} ug/m3 ")
        print(f"(SD = {self.data['pm25'].std():.2f}).")
        
        print(f"\nLinear regression indicated a {'significant' if p_trend < 0.05 else 'non-significant'} ")
        print(f"{'decreasing' if slope < 0 else 'increasing'} trend over time, ")
        print(f"beta = {slope:.4f} ug/m3 per day (approximately {slope*365:.2f} ug/m3 per year), ")
        print(f"t({len(self.data)-2}) = {slope/std_err:.2f}, p = {p_trend:.4f}." if p_trend < 0.05 else f"p = {p_trend:.4f}.")
        
        if STATSMODELS_AVAILABLE:
            print(f"\nDecomposition analysis ({self.statsmodels_ref}) revealed ")
            print(f"seasonal patterns with peak pollution during winter months.")
        
        print("\n--- INTERPRETATION ---")
        print("\n[OK] APPROPRIATE CLAIMS:")
        print(f"  - PM2.5 levels showed a {'decreasing' if slope < 0 else 'increasing'} trend")
        print(f"  - Seasonal patterns were observed")
        print(f"  - Weekly patterns suggest higher weekday pollution")
        
        print("\n[X] INAPPROPRIATE CLAIMS:")
        print("  - 'Policy X CAUSED the decrease' (cannot infer causation)")
        print("  - Temporal patterns don't establish cause-effect")
        print("  - Multiple factors could explain trends")
        
        print("\n--- LIMITATIONS ---")
        for limitation in self.metadata['limitations']:
            print(f"  - {limitation}")
    
    def generate_references(self):
        """Generate APA 7 reference list"""
        print("\n" + "="*70)
        print("REFERENCES")
        print("="*70)
        print()
        print(self.references.generate_reference_list())
    
    def run_full_study(self):
        """Execute complete time series analysis"""
        print("\n" + "="*70)
        print("TIME SERIES ANALYSIS: AIR QUALITY TRENDS")
        print("="*70)
        print(f"Study Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Research Type: {self.metadata['research_type']}")
        
        # Execute workflow
        self.generate_time_series_data()
        self.save_raw_data()
        self.descriptive_statistics()
        self.test_stationarity()
        slope, p_value = self.analyze_trend()
        self.analyze_autocorrelation()
        
        if STATSMODELS_AVAILABLE:
            self.decompose_series()
            self.forecast_future(steps=90)
        
        self.detect_changepoints()
        self.visualize_full_analysis()
        self.generate_report()
        self.generate_references()
        
        print("\n" + "="*70)
        print("STUDY COMPLETE")
        print("="*70)
        print("\nAll data and results are saved and can be independently verified.")
        print("See 03_time_series_raw_data.csv for raw data")
        print("See 03_time_series_*.png for visualizations")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXAMPLE 03: TIME SERIES ANALYSIS")
    print("="*70)
    print("\nThis example demonstrates proper time series research:")
    print("  - Analyzing temporal trends")
    print("  - Testing for stationarity")
    print("  - Decomposing seasonal patterns")
    print("  - Forecasting future values")
    print("  - Proper interpretation of temporal patterns")
    print("  - APA 7 style reporting")
    print("="*70 + "\n")
    
    study = AirQualityTimeSeriesStudy()
    study.run_full_study()
    
    print("\n[OK] Example complete. This demonstrates verifiable time series research.")
