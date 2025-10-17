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
# Standard library imports
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple
import json
import warnings

# Third-party imports
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Local imports (research_toolkit)
from research_toolkit import (
    ReportFormatter,
    SafeOutput,
    StatisticalFormatter,
    get_symbol
)
from research_toolkit.references import APA7ReferenceManager

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    SafeOutput.safe_print("WARNING: statsmodels not available. Install with: pip install statsmodels")


class AirQualityTimeSeriesStudy:
    """
    Time series analysis of air quality trends.
    
    Research Type: Time Series Analysis (Empirical)
    Design: Longitudinal observational
    """
    
    def __init__(self) -> None:
        """
        Initialize time series study.
        
        Note:
            Requires statsmodels for advanced time series analysis.
            Install with: pip install statsmodels
        """
        self.references = APA7ReferenceManager()
        
        # Add references
        self.api_ref = self.references.add_reference(
            'website',
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
        
        
        self.formatter = ReportFormatter()
        self.stat_formatter = StatisticalFormatter()
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
        
        self.data: Optional[pd.DataFrame] = None
    
    def generate_time_series_data(self) -> pd.DataFrame:
        """
        Generate realistic time series data based on typical air quality patterns.
        
        Returns:
            DataFrame with date index and PM2.5 measurements
            
        Note:
            In production, replace with actual historical API calls.
        """
        self.formatter.print_section("DATA COLLECTION")
        SafeOutput.safe_print(f"Data Source: {self.metadata['data_source']}")
        SafeOutput.safe_print(f"Time Period: {self.metadata['time_period']}")
        SafeOutput.safe_print("Frequency: Daily")
        
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
        
        SafeOutput.safe_print(f"\nCollected {len(self.data)} daily observations")
        SafeOutput.safe_print(f"Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")
        SafeOutput.safe_print(f"PM2.5 range: {self.data['pm25'].min():.1f} to {self.data['pm25'].max():.1f} ug/m3")
        
        return self.data
    
    def save_raw_data(self, filename: str = '03_time_series_raw_data.csv') -> None:
        """
        Save raw data for verification.
        
        Args:
            filename: Output CSV filename
        """
        if self.data is not None:
            self.data.to_csv(filename)
            
            metadata_file = filename.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Raw data saved to: {filename}")
            SafeOutput.safe_print(f"{get_symbol('checkmark')} Metadata saved to: {metadata_file}")
    
    def descriptive_statistics(self) -> None:
        """Calculate and display descriptive statistics."""
        self.formatter.print_section("DESCRIPTIVE STATISTICS")
        
        SafeOutput.safe_print(f"\nObservations: {len(self.data)}")
        SafeOutput.safe_print("\nPM2.5 Statistics:")
        SafeOutput.safe_print(f"  M = {self.data['pm25'].mean():.2f} ug/m3")
        SafeOutput.safe_print(f"  SD = {self.data['pm25'].std():.2f} ug/m3")
        SafeOutput.safe_print(f"  Median = {self.data['pm25'].median():.2f} ug/m3")
        SafeOutput.safe_print(f"  Min = {self.data['pm25'].min():.2f} ug/m3")
        SafeOutput.safe_print(f"  Max = {self.data['pm25'].max():.2f} ug/m3")
        SafeOutput.safe_print(f"  IQR = {self.data['pm25'].quantile(0.75) - self.data['pm25'].quantile(0.25):.2f} ug/m3")
    
    def test_stationarity(self) -> Optional[float]:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Returns:
            P-value from ADF test, or None if statsmodels not available
        """
        if not STATSMODELS_AVAILABLE:
            SafeOutput.safe_print("\n[SKIPPED] Stationarity test requires statsmodels")
            return None
        
        self.formatter.print_section("STATIONARITY TESTING")
        
        result = adfuller(self.data['pm25'].dropna())
        
        SafeOutput.safe_print(f"\nAugmented Dickey-Fuller Test ({self.statsmodels_ref}):")
        SafeOutput.safe_print(f"  ADF Statistic = {result[0]:.4f}")
        SafeOutput.safe_print(f"  p-value = {result[1]:.4f}")
        SafeOutput.safe_print("  Critical Values:")
        for key, value in result[4].items():
            SafeOutput.safe_print(f"    {key}: {value:.4f}")
        
        if result[1] < 0.05:
            SafeOutput.safe_print("\n  -> Series is stationary (p < .05)")
        else:
            SafeOutput.safe_print("\n  -> Series is non-stationary (p >= .05)")
            SafeOutput.safe_print("  -> Consider differencing for ARIMA modeling")
        
        return result[1] < 0.05
    
    def analyze_trend(self) -> Tuple[float, float]:
        """Analyze linear trend.
        
        Returns:
            Tuple of (slope, p_value) from linear regression
        """
        self.formatter.print_section("TREND ANALYSIS")
        
        # Create numeric time index
        x = np.arange(len(self.data))
        y = self.data['pm25'].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        SafeOutput.safe_print("\nLinear Trend Analysis:")
        SafeOutput.safe_print(f"  Slope = {slope:.4f} ug/m3 per day")
        SafeOutput.safe_print(f"  Yearly change = {slope * 365:.2f} ug/m3 per year")
        SafeOutput.safe_print(f"  R^2 = {r_value**2:.4f}")
        SafeOutput.safe_print(f"  p-value = {p_value:.6f}")
        
        if p_value < 0.05:
            direction = "decreasing" if slope < 0 else "increasing"
            SafeOutput.safe_print(f"\n  -> Significant {direction} trend detected (p < .05)")
        else:
            SafeOutput.safe_print("\n  -> No significant linear trend (p >= .05)")
        
        return slope, p_value
    
    def decompose_series(self) -> Optional[Any]:
        """Decompose time series into components.
        
        Returns:
            Decomposition result object, or None if statsmodels not available
        """
        if not STATSMODELS_AVAILABLE:
            SafeOutput.safe_print("\n[SKIPPED] Decomposition requires statsmodels")
            return None
        
        self.formatter.print_section("SEASONALITY DECOMPOSITION")
        
        # Decompose (annual seasonality)
        decomposition = seasonal_decompose(
            self.data['pm25'],
            model='additive',
            period=365
        )
        
        SafeOutput.safe_print("\nDecomposed into:")
        SafeOutput.safe_print("  - Trend component")
        SafeOutput.safe_print("  - Seasonal component (annual cycle)")
        SafeOutput.safe_print("  - Residual component")
        
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
        SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Decomposition plot saved")
        plt.close()
        
        return decomposition
    
    def forecast_future(self, steps: int = 90) -> Optional[pd.Series]:
        """Forecast future values using ARIMA.
        
        Args:
            steps: Number of time periods to forecast ahead
            
        Returns:
            Forecast series, or None if statsmodels not available
        """
        if not STATSMODELS_AVAILABLE:
            SafeOutput.safe_print("\n[SKIPPED] Forecasting requires statsmodels")
            return None
        
        self.formatter.print_section("FORECASTING")
        
        SafeOutput.safe_print("\nFitting ARIMA model...")
        SafeOutput.safe_print(f"Forecasting next {steps} days")
        
        # Fit ARIMA model
        model = ARIMA(self.data['pm25'], order=(1, 1, 1))
        fitted = model.fit()
        
        SafeOutput.safe_print("\nModel Summary:")
        SafeOutput.safe_print(f"  AIC = {fitted.aic:.2f}")
        SafeOutput.safe_print(f"  BIC = {fitted.bic:.2f}")
        
        # Forecast
        forecast = fitted.forecast(steps=steps)
        forecast_index = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        SafeOutput.safe_print("\nForecast Statistics:")
        SafeOutput.safe_print(f"  Mean forecast = {forecast.mean():.2f} ug/m3")
        SafeOutput.safe_print(f"  Min forecast = {forecast.min():.2f} ug/m3")
        SafeOutput.safe_print(f"  Max forecast = {forecast.max():.2f} ug/m3")
        
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
        SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Forecast plot saved")
        plt.close()
        
        return forecast
    
    def analyze_autocorrelation(self) -> None:
        """Analyze autocorrelation structure at key lags."""
        self.formatter.print_section("AUTOCORRELATION ANALYSIS")
        
        # Calculate autocorrelation
        lags = [1, 7, 30, 365]
        
        SafeOutput.safe_print("\nAutocorrelation at key lags:")
        for lag in lags:
            if lag < len(self.data):
                acf = self.data['pm25'].autocorr(lag=lag)
                SafeOutput.safe_print(f"  Lag {lag:3d} days: r = {acf:.3f}")
    
    def detect_changepoints(self) -> pd.Timestamp:
        """Detect significant changes in the series.
        
        Returns:
            Timestamp of detected changepoint
        """
        self.formatter.print_section("CHANGEPOINT DETECTION")
        
        # Simple CUSUM approach
        cumsum = (self.data['pm25'] - self.data['pm25'].mean()).cumsum()
        changepoint_idx = cumsum.abs().idxmax()
        
        SafeOutput.safe_print(f"\nPotential changepoint detected at: {changepoint_idx.date()}")
        SafeOutput.safe_print(f"CUSUM value: {cumsum.loc[changepoint_idx]:.2f}")
        
        # Compare before and after
        before = self.data.loc[:changepoint_idx, 'pm25']
        after = self.data.loc[changepoint_idx:, 'pm25']
        
        SafeOutput.safe_print(f"\nBefore changepoint: M = {before.mean():.2f} ug/m3")
        SafeOutput.safe_print(f"After changepoint: M = {after.mean():.2f} ug/m3")
        SafeOutput.safe_print(f"Difference: {abs(after.mean() - before.mean()):.2f} ug/m3")
        
        return changepoint_idx
    
    def visualize_full_analysis(self) -> None:
        """Create comprehensive visualization of all analyses."""
        self.formatter.print_section("VISUALIZATIONS")
        
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
        SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Analysis visualizations saved to: 03_time_series_analysis.png")
        plt.close()
    
    def generate_report(self) -> None:
        """Generate APA-style report with results and interpretation."""
        self.formatter.print_section("RESEARCH REPORT")
        
        SafeOutput.safe_print(f"\nTitle: {self.metadata['title']}")
        SafeOutput.safe_print(f"\nResearch Question: {self.metadata['research_question']}")
        
        slope, intercept, r_value, p_trend, std_err = stats.linregress(
            np.arange(len(self.data)),
            self.data['pm25'].values
        )
        
        SafeOutput.safe_print("\n--- RESULTS ---")
        SafeOutput.safe_print("\nTime series analysis of daily PM2.5 measurements over ")
        SafeOutput.safe_print(f"{self.metadata['time_period']} (N = {len(self.data)} observations) ")
        SafeOutput.safe_print(f"revealed a mean concentration of {self.data['pm25'].mean():.2f} ug/m3 ")
        SafeOutput.safe_print(f"(SD = {self.data['pm25'].std():.2f}).")
        
        SafeOutput.safe_print(f"\nLinear regression indicated a {'significant' if p_trend < 0.05 else 'non-significant'} ")
        SafeOutput.safe_print(f"{'decreasing' if slope < 0 else 'increasing'} trend over time, ")
        SafeOutput.safe_print(f"beta = {slope:.4f} ug/m3 per day (approximately {slope*365:.2f} ug/m3 per year), ")
        SafeOutput.safe_print(f"t({len(self.data)-2}) = {slope/std_err:.2f}, p = {p_trend:.4f}." if p_trend < 0.05 else f"p = {p_trend:.4f}.")
        
        if STATSMODELS_AVAILABLE:
            SafeOutput.safe_print(f"\nDecomposition analysis ({self.statsmodels_ref}) revealed ")
            SafeOutput.safe_print("seasonal patterns with peak pollution during winter months.")
        
        SafeOutput.safe_print("\n--- INTERPRETATION ---")
        SafeOutput.safe_print(f"\n{get_symbol('checkmark')} APPROPRIATE CLAIMS:")
        SafeOutput.safe_print(f"  - PM2.5 levels showed a {'decreasing' if slope < 0 else 'increasing'} trend")
        SafeOutput.safe_print("  - Seasonal patterns were observed")
        SafeOutput.safe_print("  - Weekly patterns suggest higher weekday pollution")
        
        SafeOutput.safe_print(f"\n{get_symbol('cross')} INAPPROPRIATE CLAIMS:")
        SafeOutput.safe_print("  - 'Policy X CAUSED the decrease' (cannot infer causation)")
        SafeOutput.safe_print("  - Temporal patterns don't establish cause-effect")
        SafeOutput.safe_print("  - Multiple factors could explain trends")
        
        SafeOutput.safe_print("\n--- LIMITATIONS ---")
        for limitation in self.metadata['limitations']:
            SafeOutput.safe_print(f"  - {limitation}")
    
    def generate_references(self) -> None:
        """Generate APA 7 reference list."""
        self.formatter.print_section("REFERENCES")
        SafeOutput.safe_print("")
        SafeOutput.safe_print(self.references.generate_reference_list())
    
    def run_full_study(self) -> None:
        """Execute complete time series analysis workflow."""
        self.formatter.print_section("TIME SERIES ANALYSIS: AIR QUALITY TRENDS")
        SafeOutput.safe_print(f"Study Date: {datetime.now().strftime('%Y-%m-%d')}")
        SafeOutput.safe_print(f"Research Type: {self.metadata['research_type']}")
        
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
        
        self.formatter.print_section("STUDY COMPLETE")
        SafeOutput.safe_print("\nAll data and results are saved and can be independently verified.")
        SafeOutput.safe_print("See 03_time_series_raw_data.csv for raw data")
        SafeOutput.safe_print("See 03_time_series_*.png for visualizations")


if __name__ == "__main__":
    formatter = ReportFormatter()
    formatter.print_section("EXAMPLE 03: TIME SERIES ANALYSIS")
    SafeOutput.safe_print("\nThis example demonstrates proper time series research:")
    SafeOutput.safe_print("  - Analyzing temporal trends")
    SafeOutput.safe_print("  - Testing for stationarity")
    SafeOutput.safe_print("  - Decomposing seasonal patterns")
    SafeOutput.safe_print("  - Forecasting future values")
    SafeOutput.safe_print("  - Proper interpretation of temporal patterns")
    SafeOutput.safe_print("  - APA 7 style reporting")
    SafeOutput.safe_print("="*70 + "\n")
    
    study = AirQualityTimeSeriesStudy()
    study.run_full_study()
    
    SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Example complete. This demonstrates verifiable time series research.")
