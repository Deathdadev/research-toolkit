"""
Example 06: Simulation Study - SIR Epidemic Model

Research Question: Under SIR model assumptions, how do intervention strategies affect epidemic spread?

CRITICAL: This demonstrates CONDITIONAL research - results depend entirely on model assumptions.
Claims are "IF assumptions hold, THEN..." NOT facts about real epidemics.

This demonstrates:
- Simulation/model-based research
- Explicit statement of assumptions
- Scenario comparison
- Sensitivity analysis
- CONDITIONAL interpretation (IF-THEN)
- Validation requirement for real-world claims
- APA 7 referencing using research_toolkit
"""
# Standard library imports
from datetime import datetime
from typing import Dict

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports (research_toolkit)
from research_toolkit import ReportFormatter, SafeOutput, StatisticalFormatter, get_symbol
from research_toolkit.references import APA7ReferenceManager

class SIREpidemicSimulation:
    """
    SIR (Susceptible-Infected-Recovered) epidemic simulation.
    
    Research Type: Simulation Study (Model-Based)
    IMPORTANT: Results are CONDITIONAL on model assumptions
    """
    
    def __init__(self) -> None:
        """
        Initialize simulation study.
        
        Note:
            Uses computational modeling to explore theoretical scenarios.
        """
        self.references = APA7ReferenceManager()
        
        self.kermack_ref = self.references.add_reference(
            'journal',
            author='Kermack, W. O.; McKendrick, A. G.',
            year='1927',
            title='A contribution to the mathematical theory of epidemics',
            journal='Proceedings of the Royal Society A',
            volume='115',
            pages='700-721'
        )
        
        
        self.formatter = ReportFormatter()
        self.stat_formatter = StatisticalFormatter()
        self.metadata = {
            'research_type': 'Simulation Study (Model-Based)',
            'study_date': datetime.now().isoformat(),
            'title': 'Intervention Strategies in Epidemic Spread: A Simulation Study',
            'research_question': 'Under SIR model assumptions, how do interventions affect epidemic dynamics?',
            'model': f'SIR (Susceptible-Infected-Recovered) model ({self.kermack_ref})',
            'data_type': 'MODEL-GENERATED (synthetic)',
            'validity': 'CONDITIONAL on model assumptions',
            'assumptions': [
                'Population is well-mixed (homogeneous mixing)',
                'Infection rate is constant over time',
                'Recovered individuals have permanent immunity',
                'No births, deaths (unrelated to disease), or migration',
                'Disease transmission follows mass action principle',
                'Population size is fixed'
            ],
            'parameters': {
                'population_size': {'value': 10000, 'justification': 'Typical small city'},
                'initial_infected': {'value': 10, 'justification': 'Small outbreak scenario'},
                'infection_rate': {'value': 0.3, 'justification': 'Moderate transmissibility'},
                'recovery_rate': {'value': 0.1, 'justification': '10-day average recovery period'}
            },
            'scenarios': {
                'baseline': 'No intervention',
                'social_distancing': 'Reduce infection rate by 50%',
                'improved_treatment': 'Increase recovery rate by 50%',
                'combined': 'Both interventions'
            },
            'limitations': [
                'Model is simplified - real behavior more complex',
                'Assumes homogeneous population (unrealistic)',
                'Parameters are estimates, not precise measurements',
                'Does not account for individual variation',
                'Deterministic model (no stochasticity in this version)',
                'RESULTS CANNOT BE TREATED AS PREDICTIONS without validation'
            ]
        }
    
    def state_assumptions_upfront(self) -> None:
        """State all assumptions and limitations of the simulation."""
        self.formatter.print_section("MODEL ASSUMPTIONS - CRITICAL")
        SafeOutput.safe_print("\n[!] ALL RESULTS ARE CONDITIONAL ON THESE ASSUMPTIONS:")
        SafeOutput.safe_print("\nAssumptions:")
        for i, assumption in enumerate(self.metadata['assumptions'], 1):
            SafeOutput.safe_print(f"  {i}. {assumption}")
        
        SafeOutput.safe_print("\nParameters:")
        for param, info in self.metadata['parameters'].items():
            SafeOutput.safe_print(f"  {param}: {info['value']}")
            SafeOutput.safe_print(f"    Justification: {info['justification']}")
        
        self.formatter.print_section("IF ASSUMPTIONS HOLD, THEN the following results emerge:")
    
    def run_sir_simulation(self, days: int = 200, infection_rate: float = 0.3, recovery_rate: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Run SIR epidemiological simulation.
        
        Args:
            days: Number of days to simulate
            infection_rate: Beta parameter (infection rate)
            recovery_rate: Gamma parameter (recovery rate)
            
        Returns:
            Dictionary containing time series for S, I, R compartments
        """
        N = self.metadata['parameters']['population_size']['value']
        I0 = self.metadata['parameters']['initial_infected']['value']
        
        S = np.zeros(days)
        I = np.zeros(days)
        R = np.zeros(days)
        
        S[0] = N - I0
        I[0] = I0
        R[0] = 0
        
        for t in range(1, days):
            new_infections = infection_rate * S[t-1] * I[t-1] / N
            new_recoveries = recovery_rate * I[t-1]
            
            S[t] = S[t-1] - new_infections
            I[t] = I[t-1] + new_infections - new_recoveries
            R[t] = R[t-1] + new_recoveries
        
        return S, I, R
    
    def scenario_analysis(self) -> None:
        """Compare different intervention scenarios."""
        self.formatter.print_section("SCENARIO ANALYSIS")
        
        scenarios = {
            'Baseline': {'beta': 0.3, 'gamma': 0.1},
            'Social Distancing': {'beta': 0.15, 'gamma': 0.1},
            'Improved Treatment': {'beta': 0.3, 'gamma': 0.15},
            'Combined': {'beta': 0.15, 'gamma': 0.15}
        }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            S, I, R = self.run_sir_simulation(
                infection_rate=params['beta'],
                recovery_rate=params['gamma']
            )
            
            peak_infected = I.max()
            peak_day = I.argmax()
            total_infected = R[-1]
            
            results[scenario_name] = {
                'peak_infected': peak_infected,
                'peak_day': peak_day,
                'total_infected': total_infected,
                'S': S, 'I': I, 'R': R
            }
            
            SafeOutput.safe_print(f"\n{scenario_name}:")
            SafeOutput.safe_print(f"  Peak infections: {peak_infected:.0f} (day {peak_day})")
            SafeOutput.safe_print(f"  Total infected: {total_infected:.0f} ({total_infected/10000*100:.1f}%)")
        
        # Visualize scenarios
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SIR Model: Intervention Scenario Comparison', fontsize=14, fontweight='bold')
        
        colors = {'Baseline': 'red', 'Social Distancing': 'blue', 
                 'Improved Treatment': 'green', 'Combined': 'purple'}
        
        for scenario_name, data in results.items():
            axes[0, 0].plot(data['I'], label=scenario_name, color=colors[scenario_name], linewidth=2)
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Infected Individuals')
        axes[0, 0].set_title('Infected Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Peak infections comparison
        peaks = [results[s]['peak_infected'] for s in scenarios.keys()]
        axes[0, 1].bar(range(len(scenarios)), peaks, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xticks(range(len(scenarios)))
        axes[0, 1].set_xticklabels(scenarios.keys(), rotation=45, ha='right')
        axes[0, 1].set_ylabel('Peak Infected')
        axes[0, 1].set_title('Peak Infections by Scenario')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Total infections comparison
        totals = [results[s]['total_infected'] for s in scenarios.keys()]
        axes[1, 0].bar(range(len(scenarios)), totals, edgecolor='black', alpha=0.7, color='coral')
        axes[1, 0].set_xticks(range(len(scenarios)))
        axes[1, 0].set_xticklabels(scenarios.keys(), rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Infected')
        axes[1, 0].set_title('Total Infections by Scenario')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Full SIR dynamics for baseline
        ax = axes[1, 1]
        ax.plot(results['Baseline']['S'], label='Susceptible', linewidth=2)
        ax.plot(results['Baseline']['I'], label='Infected', linewidth=2)
        ax.plot(results['Baseline']['R'], label='Recovered', linewidth=2)
        ax.set_xlabel('Day')
        ax.set_ylabel('Individuals')
        ax.set_title('SIR Dynamics: Baseline Scenario')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('06_simulation_scenarios.png', dpi=300, bbox_inches='tight')
        SafeOutput.safe_print("\n{get_symbol('checkmark')} Scenario visualizations saved")
        plt.close()
        
        return results
    
    def sensitivity_analysis(self) -> None:
        """Test sensitivity to parameter changes."""
        self.formatter.print_section("SENSITIVITY ANALYSIS")
        
        # Test sensitivity to infection rate
        infection_rates = np.linspace(0.1, 0.5, 10)
        peak_infections = []
        
        for beta in infection_rates:
            S, I, R = self.run_sir_simulation(infection_rate=beta, recovery_rate=0.1)
            peak_infections.append(I.max())
        
        plt.figure(figsize=(10, 6))
        plt.plot(infection_rates, peak_infections, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Infection Rate (β)')
        plt.ylabel('Peak Infected Individuals')
        plt.title('Sensitivity to Infection Rate Parameter')
        plt.grid(True, alpha=0.3)
        plt.savefig('06_simulation_sensitivity.png', dpi=300, bbox_inches='tight')
        SafeOutput.safe_print("\n{get_symbol('checkmark')} Sensitivity analysis saved")
        plt.close()
        
        SafeOutput.safe_print(f"\nInfection rate range: {infection_rates.min():.2f} to {infection_rates.max():.2f}")
        SafeOutput.safe_print(f"Peak infections range: {min(peak_infections):.0f} to {max(peak_infections):.0f}")
        SafeOutput.safe_print(f"Result variability: {(max(peak_infections)-min(peak_infections))/min(peak_infections)*100:.1f}% change")
    
    def generate_report(self) -> None:
        """Generate simulation study report."""
        self.formatter.print_section("SIMULATION STUDY REPORT")
        
        SafeOutput.safe_print(f"\nTitle: {self.metadata['title']}")
        SafeOutput.safe_print(f"\nResearch Question: {self.metadata['research_question']}")
        
        SafeOutput.safe_print("\n--- IMPORTANT NOTICE ---")
        SafeOutput.safe_print("This is a SIMULATION STUDY. Results are CONDITIONAL on model assumptions.")
        SafeOutput.safe_print("These are NOT predictions about real epidemics without validation.")
        
        SafeOutput.safe_print("\n--- RESULTS ---")
        SafeOutput.safe_print("\nSimulation results indicated that:")
        SafeOutput.safe_print("  - Baseline scenario: Peak infections occurred around day X")
        SafeOutput.safe_print("  - Social distancing: Reduced peak by approximately Y%")
        SafeOutput.safe_print("  - Improved treatment: Reduced total infections by Z%")
        SafeOutput.safe_print("  - Combined interventions: Most effective at flattening curve")
        
        SafeOutput.safe_print("\n--- INTERPRETATION ---")
        SafeOutput.safe_print("\n{get_symbol('checkmark')} APPROPRIATE CLAIMS (Conditional):")
        SafeOutput.safe_print("  - 'Under model assumptions, social distancing reduces peak infections'")
        SafeOutput.safe_print("  - 'IF infection rate is 0.3, THEN peak occurs around day X'")
        SafeOutput.safe_print("  - 'The model suggests that combined interventions are most effective'")
        SafeOutput.safe_print("  - 'According to this simulation...'")
        
        SafeOutput.safe_print("\n{get_symbol('cross')} INAPPROPRIATE CLAIMS (Unconditional):")
        SafeOutput.safe_print("  - 'Real epidemics will behave this way' (NO! Needs validation)")
        SafeOutput.safe_print("  - 'This proves intervention X works' (NO! Model only)")
        SafeOutput.safe_print("  - Any claim as fact without 'according to model' qualifier")
        
        SafeOutput.safe_print("\n[!] CRITICAL LIMITATIONS:")
        for limitation in self.metadata['limitations']:
            SafeOutput.safe_print(f"  - {limitation}")
        
        SafeOutput.safe_print("\n--- CONCLUSION ---")
        SafeOutput.safe_print("\nThis simulation study explored epidemic dynamics under SIR model ")
        SafeOutput.safe_print(f"assumptions ({self.kermack_ref}). Results suggest potential effectiveness ")
        SafeOutput.safe_print("of interventions. However, EMPIRICAL VALIDATION is required before ")
        SafeOutput.safe_print("applying these findings to real-world epidemic management.")
    
    def generate_references(self) -> None:
        """Generate APA 7 reference list."""
        self.formatter.print_section("REFERENCES")
        SafeOutput.safe_print("")
        SafeOutput.safe_print(self.references.generate_reference_list())
    
    def run_full_study(self) -> None:
        """Execute complete simulation study workflow."""
        self.formatter.print_section("SIMULATION STUDY: EPIDEMIC INTERVENTIONS")
        SafeOutput.safe_print(f"Study Date: {datetime.now().strftime('%Y-%m-%d')}")
        SafeOutput.safe_print(f"Research Type: {self.metadata['research_type']}")
        
        self.state_assumptions_upfront()
        self.scenario_analysis()
        self.sensitivity_analysis()
        self.generate_report()
        self.generate_references()
        
        self.formatter.print_section("SIMULATION STUDY COMPLETE")
        SafeOutput.safe_print("\n[!] REMEMBER: These are model results, not real-world predictions!")
        SafeOutput.safe_print("Empirical validation required for real-world application.")


if __name__ == "__main__":
    formatter = ReportFormatter()
    formatter.print_section("EXAMPLE 06: SIMULATION STUDY")
    SafeOutput.safe_print("\nThis example demonstrates proper simulation research:")
    SafeOutput.safe_print("  - Explicit statement of all assumptions")
    SafeOutput.safe_print("  - Model-based scenario exploration")
    SafeOutput.safe_print("  - Sensitivity analysis")
    SafeOutput.safe_print("  - CONDITIONAL interpretation (IF-THEN)")
    SafeOutput.safe_print("  - Clear boundaries: model vs reality")
    SafeOutput.safe_print("  - Validation requirement stated")
    SafeOutput.safe_print("\n[!] Shows when synthetic data is acceptable WITH CAVEATS")
    SafeOutput.safe_print("="*70 + "\n")
    
    sim = SIREpidemicSimulation()
    sim.run_full_study()
    
    SafeOutput.safe_print(f"\n{get_symbol('checkmark')} Example complete. This demonstrates conditional simulation research.")
