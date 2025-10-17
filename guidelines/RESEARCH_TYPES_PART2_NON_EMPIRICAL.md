# Complete Research Types Guide - Part 2: Non-Empirical Research

## Overview

This document covers research types where synthetic or theoretical data may be appropriate, with critical caveats about their use and interpretation.

---

## 6. SIMULATION STUDY (Model-Based - Conditional)

### Definition
Uses computational models to simulate system behavior under various conditions, exploring theoretical scenarios or testing interventions.

### When to Use
- Studying complex systems too expensive/dangerous to experiment on
- Exploring "what if" scenarios
- Testing theoretical models
- Understanding emergent properties
- Comparing intervention strategies

### Data Requirements
⚠️ **Synthetic data acceptable** BUT with major caveats:
- Model assumptions must be explicitly stated
- Parameters should be justified (ideally from real data)
- Results are conditional on model validity
- ❌ Cannot make real-world claims without validation

### Key Characteristics
- Based on theoretical models
- Explores logical consequences of assumptions
- Results are "if-then" statements
- Can generate hypotheses for empirical testing
- Validity depends entirely on model accuracy

### Example Research Questions
- "IF people behave according to model X, THEN what epidemic patterns emerge?"
- "Under assumptions A, B, C, how does traffic flow change?"
- "What climate patterns result from these parameter values?"
- "How does network structure affect information spread?"

### Python Implementation Pattern

```python
class SimulationStudy:
    """
    Template for simulation-based research
    
    CRITICAL: Results are conditional on model assumptions
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Simulation Study (Theoretical/Model-Based)',
            'design': 'Computational modeling',
            'data_requirement': 'Model-generated (synthetic)',
            'validity': 'CONDITIONAL on model assumptions',
            'can_infer_causation': 'Only within model, not real world',
            'key_requirement': 'Explicit statement of assumptions',
            'methods': [
                'Agent-based modeling',
                'System dynamics',
                'Monte Carlo simulation',
                'Discrete event simulation',
                'Network simulation'
            ]
        }
        self.assumptions = []
        self.parameters = {}
    
    def state_assumptions_upfront(self):
        """
        MANDATORY: State all model assumptions before running
        """
        print("="*70)
        print("MODEL ASSUMPTIONS")
        print("="*70)
        print("\nCRITICAL: All results are conditional on these assumptions")
        print("\nAssumptions:")
        for i, assumption in enumerate(self.assumptions, 1):
            print(f"  {i}. {assumption}")
        
        print("\nParameters:")
        for key, value in self.parameters.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        print("IF these assumptions hold, THEN the following results emerge:")
        print("="*70 + "\n")
    
    def justify_parameters(self, param_name, value, justification):
        """
        Document parameter choices with justification
        """
        self.parameters[param_name] = {
            'value': value,
            'justification': justification
        }
    
    def run_simulation(self, n_iterations=1000):
        """
        Run simulation model
        """
        # Example structure
        results = []
        
        for iteration in range(n_iterations):
            # Run model with current parameters
            outcome = self._single_simulation_run()
            results.append(outcome)
        
        return np.array(results)
    
    def _single_simulation_run(self):
        """
        Single simulation iteration
        Implement your specific model here
        """
        # Example: Simple epidemic model
        pass
    
    def sensitivity_analysis(self, param_name, param_range):
        """
        CRITICAL: Test how results change with parameters
        Shows robustness of findings
        """
        print(f"\nSENSITIVITY ANALYSIS: {param_name}")
        print("="*70)
        
        results = []
        for value in param_range:
            self.parameters[param_name]['value'] = value
            outcome = self.run_simulation(n_iterations=100)
            results.append(outcome.mean())
        
        # Plot sensitivity
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, results, 'o-', linewidth=2)
        plt.xlabel(f'{param_name}')
        plt.ylabel('Mean Outcome')
        plt.title(f'Sensitivity to {param_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'sensitivity_{param_name}.png')
        
        print(f"Results range from {min(results):.3f} to {max(results):.3f}")
        print(f"Coefficient of variation: {np.std(results)/np.mean(results):.3f}")
    
    def validate_against_reality(self, real_data):
        """
        CRITICAL: Compare simulation to real data
        Required before making real-world claims
        """
        print("\nMODEL VALIDATION")
        print("="*70)
        
        # Run simulation
        sim_results = self.run_simulation()
        
        # Compare distributions
        stat, p = stats.ks_2samp(sim_results, real_data)
        
        print(f"Kolmogorov-Smirnov test: D={stat:.3f}, p={p:.4f}")
        
        if p > 0.05:
            print("Model output consistent with real data")
        else:
            print("WARNING: Model output differs from real data")
            print("Model may need revision before making predictions")
        
        # Visual comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(sim_results, bins=30, alpha=0.7, label='Simulated', edgecolor='black')
        plt.hist(real_data, bins=30, alpha=0.7, label='Real Data', edgecolor='black')
        plt.legend()
        plt.title('Distribution Comparison')
        
        plt.subplot(1, 2, 2)
        plt.boxplot([sim_results, real_data], labels=['Simulated', 'Real'])
        plt.title('Box Plot Comparison')
        
        plt.tight_layout()
        plt.savefig('model_validation.png')
    
    def scenario_analysis(self, scenarios):
        """
        Compare different scenarios
        """
        print("\nSCENARIO ANALYSIS")
        print("="*70)
        
        results = {}
        for scenario_name, params in scenarios.items():
            print(f"\nScenario: {scenario_name}")
            
            # Update parameters
            for param, value in params.items():
                self.parameters[param]['value'] = value
            
            # Run simulation
            outcome = self.run_simulation(n_iterations=500)
            results[scenario_name] = outcome
            
            print(f"  Mean outcome: {outcome.mean():.3f}")
            print(f"  95% CI: [{np.percentile(outcome, 2.5):.3f}, {np.percentile(outcome, 97.5):.3f}]")
        
        # Visualize scenarios
        plt.figure(figsize=(12, 6))
        plt.boxplot(results.values(), labels=results.keys())
        plt.ylabel('Outcome')
        plt.title('Comparison of Scenarios')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('scenario_comparison.png')
        
        return results
    
    def state_limitations(self):
        """
        MANDATORY: State limitations
        """
        print("\n" + "="*70)
        print("CRITICAL LIMITATIONS")
        print("="*70)
        print("\n1. MODEL ASSUMPTIONS:")
        print("   - All results conditional on assumptions listed above")
        print("   - Real-world behavior may differ from model")
        print("   - Assumptions are simplifications of reality")
        
        print("\n2. PARAMETER UNCERTAINTY:")
        print("   - Parameters estimated/assumed, not directly measured")
        print("   - Different parameter values may yield different results")
        print("   - See sensitivity analysis for robustness")
        
        print("\n3. VALIDATION:")
        if hasattr(self, 'validated'):
            print("   - Model compared to real data")
        else:
            print("   - ⚠️ WARNING: Model NOT validated against real data")
            print("   - Cannot make confident real-world predictions")
        
        print("\n4. SCOPE:")
        print("   - Results show what COULD happen IF assumptions hold")
        print("   - NOT predictions of what WILL happen")
        print("   - Use for hypothesis generation and theory development")
        
        print("\n5. EXTERNAL VALIDITY:")
        print("   - Generalization to real world requires empirical validation")
        print("   - Model captures some but not all real-world complexity")
    
    def interpret_results(self, results):
        """
        Proper interpretation for simulation studies
        """
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        print("\n✓ APPROPRIATE CLAIMS:")
        print("  - 'Under the stated assumptions, the model predicts...'")
        print("  - 'IF assumptions A, B, C hold, THEN we expect...'")
        print("  - 'The model suggests that...'")
        print("  - 'According to the simulation...'")
        
        print("\n❌ INAPPROPRIATE CLAIMS:")
        print("  - 'This proves that in reality...' (NO! Need validation)")
        print("  - 'Real-world outcomes will be...' (NO! Model is simplified)")
        print("  - 'This demonstrates that...' (NO! Shows model behavior only)")
        
        print("\n⚠️ CONDITIONAL CLAIMS:")
        print("  - Always state: 'According to this model...'")
        print("  - Always state: 'If assumptions hold...'")
        print("  - Always note: 'Empirical validation needed'")

# Example: Epidemic Simulation
"""
class EpidemicSimulation(SimulationStudy):
    def __init__(self):
        super().__init__()
        
        # CRITICAL: State assumptions
        self.assumptions = [
            "Population is well-mixed (everyone can contact everyone)",
            "Infection probability is constant over time",
            "Recovered individuals have permanent immunity",
            "No births, deaths, or migration",
            "Contact rate is homogeneous across population"
        ]
        
        # Justify parameters
        self.justify_parameters(
            'infection_rate',
            value=0.3,
            justification='Based on Smith et al. (2020) empirical study'
        )
        self.justify_parameters(
            'recovery_rate',
            value=0.1,
            justification='Average recovery time of 10 days from literature'
        )
    
    def _single_simulation_run(self):
        # SIR model implementation
        # S: Susceptible, I: Infected, R: Recovered
        population = 10000
        infected = 10
        susceptible = population - infected
        recovered = 0
        
        days = 100
        peak_infected = 0
        
        for day in range(days):
            # New infections
            new_infections = (self.parameters['infection_rate']['value'] * 
                            susceptible * infected / population)
            
            # New recoveries
            new_recoveries = self.parameters['recovery_rate']['value'] * infected
            
            # Update compartments
            susceptible -= new_infections
            infected += new_infections - new_recoveries
            recovered += new_recoveries
            
            peak_infected = max(peak_infected, infected)
        
        return peak_infected
    
    def run_study(self):
        # State assumptions upfront
        self.state_assumptions_upfront()
        
        # Run simulation
        results = self.run_simulation(n_iterations=1000)
        
        print(f"\nSimulation Results (n=1000 runs):")
        print(f"Mean peak infections: {results.mean():.0f}")
        print(f"95% CI: [{np.percentile(results, 2.5):.0f}, {np.percentile(results, 97.5):.0f}]")
        
        # Sensitivity analysis
        self.sensitivity_analysis('infection_rate', np.linspace(0.1, 0.5, 10))
        
        # Scenario analysis
        scenarios = {
            'No Intervention': {'infection_rate': 0.3, 'recovery_rate': 0.1},
            'Social Distancing': {'infection_rate': 0.15, 'recovery_rate': 0.1},
            'Improved Treatment': {'infection_rate': 0.3, 'recovery_rate': 0.15}
        }
        self.scenario_analysis(scenarios)
        
        # Critical interpretation
        self.interpret_results(results)
        self.state_limitations()

# Usage
sim = EpidemicSimulation()
sim.run_study()
"""
```

### Critical Guidelines for Simulation Studies

**✓ DO:**
- State all assumptions explicitly upfront
- Justify parameter values (ideally from literature/data)
- Conduct sensitivity analysis
- Present results as conditional ("IF assumptions hold...")
- Validate against real data when possible
- Use for hypothesis generation

**❌ DON'T:**
- Claim model predictions are facts about reality
- Hide or downplay assumptions
- Make unconditional predictions without validation
- Ignore parameter uncertainty
- Claim causation in real world based on model

---

## 7. METHODOLOGICAL STUDY (Testing Methods)

### Definition
Research that tests, evaluates, or compares statistical methods, algorithms, or analytical techniques.

### When to Use
- Developing new statistical methods
- Comparing algorithm performance
- Testing method properties (power, bias, efficiency)
- Validating analytical procedures
- Establishing method benchmarks

### Data Requirements
✅ **Synthetic data IS appropriate** - you're testing the METHOD
- Generate data with known properties
- Test if method recovers known parameters
- Compare methods under controlled conditions

### Key Characteristics
- Focus on method/algorithm, not substantive phenomena
- Synthetic data with known ground truth
- Systematic variation of data properties
- Performance metrics (power, Type I error, efficiency)
- Results about method, not world

### Example Research Questions
- "What is the statistical power of this test under various conditions?"
- "How robust is this algorithm to outliers?"
- "Which clustering method performs best for these data types?"
- "Does this estimation method produce unbiased results?"

### Python Implementation Pattern

```python
class MethodologicalStudy:
    """
    Template for methodological research
    
    APPROPRIATE use of synthetic data - testing methods
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Methodological Study',
            'purpose': 'Test/compare statistical methods or algorithms',
            'data_requirement': 'Synthetic data with known properties',
            'claims_about': 'Method performance, NOT real-world phenomena',
            'synthetic_data_ok': True,  # ✓ YES, for this type
            'methods': [
                'Monte Carlo simulation',
                'Power analysis',
                'Bias assessment',
                'Efficiency comparison',
                'Robustness testing'
            ]
        }
    
    def state_purpose(self):
        """
        CRITICAL: Clarify this is methodological research
        """
        print("="*70)
        print("METHODOLOGICAL STUDY")
        print("="*70)
        print("\nPURPOSE: Testing statistical method/algorithm performance")
        print("\nDATA: Synthetic data with known properties")
        print("  - This is APPROPRIATE for methodological research")
        print("  - We are testing the METHOD, not making claims about the world")
        
        print("\nRESEARCH QUESTION: ", end="")
        print("How does [method] perform under [conditions]?")
        print("\nCLAIMS: About method properties, NOT real-world phenomena")
        print("="*70 + "\n")
    
    def generate_test_data(self, n, true_effect, noise_sd, **kwargs):
        """
        Generate synthetic data with KNOWN properties
        This is valid for methodological research
        """
        print(f"Generating test data:")
        print(f"  Sample size: {n}")
        print(f"  True effect: {true_effect}")
        print(f"  Noise SD: {noise_sd}")
        
        # Generate data with known ground truth
        x = np.random.normal(0, 1, n)
        y = true_effect * x + np.random.normal(0, noise_sd, n)
        
        return x, y, true_effect  # Return ground truth for comparison
    
    def test_method_power(self, method, effect_sizes, sample_sizes, 
                          n_simulations=1000, alpha=0.05):
        """
        Test statistical power of a method
        """
        print("\nPOWER ANALYSIS")
        print("="*70)
        
        results = pd.DataFrame(
            columns=['effect_size', 'sample_size', 'power']
        )
        
        for effect in effect_sizes:
            for n in sample_sizes:
                significant_count = 0
                
                for sim in range(n_simulations):
                    # Generate data with known effect
                    x, y, true_effect = self.generate_test_data(
                        n=n,
                        true_effect=effect,
                        noise_sd=1.0
                    )
                    
                    # Test method
                    _, p_value = method(x, y)
                    
                    if p_value < alpha:
                        significant_count += 1
                
                power = significant_count / n_simulations
                
                results = pd.concat([results, pd.DataFrame({
                    'effect_size': [effect],
                    'sample_size': [n],
                    'power': [power]
                })], ignore_index=True)
                
                print(f"Effect={effect:.2f}, n={n}: Power={power:.3f}")
        
        # Visualize
        for effect in effect_sizes:
            subset = results[results['effect_size'] == effect]
            plt.plot(subset['sample_size'], subset['power'], 
                    'o-', label=f'Effect={effect:.2f}')
        
        plt.axhline(y=0.80, color='r', linestyle='--', label='80% Power')
        plt.xlabel('Sample Size')
        plt.ylabel('Statistical Power')
        plt.title('Power Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('power_analysis.png')
        plt.close()
        
        return results
    
    def test_method_bias(self, method, true_parameter, n_simulations=1000):
        """
        Test if method produces biased estimates
        """
        print("\nBIAS ASSESSMENT")
        print("="*70)
        
        estimates = []
        
        for sim in range(n_simulations):
            # Generate data with known parameter
            x, y, _ = self.generate_test_data(
                n=100,
                true_effect=true_parameter,
                noise_sd=1.0
            )
            
            # Estimate parameter
            estimate = method(x, y)
            estimates.append(estimate)
        
        estimates = np.array(estimates)
        
        # Calculate bias
        bias = estimates.mean() - true_parameter
        relative_bias = bias / true_parameter * 100
        
        print(f"True parameter: {true_parameter:.4f}")
        print(f"Mean estimate: {estimates.mean():.4f}")
        print(f"Bias: {bias:.4f} ({relative_bias:.2f}%)")
        print(f"RMSE: {np.sqrt(np.mean((estimates - true_parameter)**2)):.4f}")
        
        # Distribution of estimates
        plt.figure(figsize=(10, 6))
        plt.hist(estimates, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=true_parameter, color='r', linestyle='--', 
                   linewidth=2, label='True Value')
        plt.axvline(x=estimates.mean(), color='g', linestyle='--',
                   linewidth=2, label='Mean Estimate')
        plt.xlabel('Estimate')
        plt.ylabel('Frequency')
        plt.title('Distribution of Estimates')
        plt.legend()
        plt.savefig('bias_assessment.png')
        plt.close()
        
        return bias, estimates
    
    def compare_methods(self, methods, conditions, metric='power'):
        """
        Compare multiple methods under various conditions
        """
        print("\nMETHOD COMPARISON")
        print("="*70)
        
        results = {}
        
        for method_name, method in methods.items():
            print(f"\nTesting: {method_name}")
            
            method_results = []
            
            for condition in conditions:
                # Test method under this condition
                performance = self._test_under_condition(method, condition, metric)
                method_results.append(performance)
            
            results[method_name] = method_results
        
        # Visualize comparison
        plt.figure(figsize=(12, 6))
        for method_name, performance in results.items():
            plt.plot(range(len(conditions)), performance, 'o-', 
                    label=method_name, linewidth=2, markersize=8)
        
        plt.xlabel('Condition')
        plt.ylabel(metric.capitalize())
        plt.title(f'Method Comparison: {metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(conditions)), 
                  [f"C{i+1}" for i in range(len(conditions))])
        plt.savefig('method_comparison.png')
        plt.close()
        
        return results
    
    def _test_under_condition(self, method, condition, metric):
        """Test method under specific condition"""
        # Implement testing logic
        pass
    
    def test_robustness(self, method, perturbations):
        """
        Test method robustness to various perturbations
        """
        print("\nROBUSTNESS TESTING")
        print("="*70)
        
        for perturbation_name, perturbation_func in perturbations.items():
            print(f"\nTesting: {perturbation_name}")
            
            # Generate clean data
            x, y, true_effect = self.generate_test_data(n=100, true_effect=0.5, noise_sd=1.0)
            
            # Apply perturbation
            x_perturbed, y_perturbed = perturbation_func(x, y)
            
            # Test method on clean vs perturbed
            result_clean = method(x, y)
            result_perturbed = method(x_perturbed, y_perturbed)
            
            difference = abs(result_clean - result_perturbed)
            relative_change = difference / abs(result_clean) * 100
            
            print(f"  Clean data: {result_clean:.4f}")
            print(f"  Perturbed data: {result_perturbed:.4f}")
            print(f"  Change: {relative_change:.2f}%")
    
    def interpret_results(self):
        """
        Appropriate interpretation for methodological studies
        """
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        print("\n✓ APPROPRIATE CLAIMS:")
        print("  - 'This method has 80% power to detect effect size d=0.5'")
        print("  - 'Method A outperforms Method B under these conditions'")
        print("  - 'The estimator is unbiased for sample sizes n>50'")
        print("  - 'The algorithm is robust to 5% outliers'")
        
        print("\n❌ INAPPROPRIATE CLAIMS:")
        print("  - 'This proves X causes Y' (NO! Testing method, not theory)")
        print("  - 'In the real world, we observe...' (NO! Synthetic data)")
        print("  - Any substantive claim about phenomena (NO! Testing method)")
        
        print("\n✓ VALID USE OF RESULTS:")
        print("  - Recommend method for specific conditions")
        print("  - Inform sample size planning")
        print("  - Identify method limitations")
        print("  - Guide method selection for applied research")

# Example: Testing correlation test power
"""
study = MethodologicalStudy()
study.state_purpose()

# Test power of Pearson correlation
def pearson_test(x, y):
    r, p = stats.pearsonr(x, y)
    return r, p

power_results = study.test_method_power(
    method=pearson_test,
    effect_sizes=[0.1, 0.3, 0.5],
    sample_sizes=[20, 50, 100, 200],
    n_simulations=1000
)

# Test bias
def correlation_estimator(x, y):
    return stats.pearsonr(x, y)[0]

study.test_method_bias(
    method=correlation_estimator,
    true_parameter=0.5,
    n_simulations=1000
)

study.interpret_results()
"""
```

### Critical Points for Methodological Studies

**This is the ONLY research type where synthetic data is fully appropriate** because:
- You're testing the METHOD, not studying the world
- You NEED known ground truth to evaluate method
- Synthetic data lets you control conditions systematically

**But remember:**
- Claims are about METHOD performance
- NOT about real-world phenomena
- Results guide method selection for future research

---

## 8. THEORETICAL MODEL (Pure Theory)

### Definition
Develops, explores, or tests theoretical frameworks, mathematical models, or conceptual systems without empirical data collection.

### When to Use
- Developing new theoretical frameworks
- Exploring logical implications of assumptions
- Mathematical proofs or derivations
- Conceptual analysis
- Theory building or refinement

### Data Requirements
⚠️ **May use no data or illustrative synthetic data**
- Focus is on logical coherence and theoretical development
- If using data, it's illustrative only
- Results are theoretical, not empirical claims

### Key Characteristics
- Purely theoretical/conceptual
- Logical analysis and argumentation
- Mathematical derivations
- No empirical testing (or minimal illustration)
- Advances theoretical understanding

### Example Research Questions
- "What are the mathematical properties of system X?"
- "What does theory Y predict under conditions A, B, C?"
- "How do these theoretical concepts relate?"
- "What are the logical implications of assumption Z?"

### Python Implementation Pattern

```python
class TheoreticalModel:
    """
    Template for theoretical research
    
    Focus: Theory development and logical analysis
    """
    
    def __init__(self):
        self.metadata = {
            'research_type': 'Theoretical Model',
            'purpose': 'Theory development and logical analysis',
            'data_requirement': 'None (or illustrative only)',
            'claims_about': 'Theoretical relationships and logical implications',
            'empirical_testing': 'Future work (not part of this study)',
            'methods': [
                'Logical analysis',
                'Mathematical derivation',
                'Conceptual framework development',
                'Theoretical modeling'
            ]
        }
        self.assumptions = []
        self.propositions = []
    
    def state_theoretical_framework(self):
        """
        Present theoretical framework being developed/explored
        """
        print("="*70)
        print("THEORETICAL FRAMEWORK")
        print("="*70)
        print("\nPURPOSE: Develop/explore theoretical model")
        print("\nTYPE: Pure theory (not empirical research)")
        print("\nOUTCOME: Theoretical propositions and logical implications")
        print("\nEMPIRICAL TESTING: Required in future work")
        print("="*70 + "\n")
    
    def define_constructs(self, constructs):
        """
        Formally define theoretical constructs
        """
        print("THEORETICAL CONSTRUCTS")
        print("="*70)
        for name, definition in constructs.items():
            print(f"\n{name}:")
            print(f"  Definition: {definition['definition']}")
            print(f"  Properties: {definition['properties']}")
    
    def state_axioms(self, axioms):
        """
        State foundational assumptions/axioms
        """
        print("\nAXIOMS (Foundational Assumptions)")
        print("="*70)
        for i, axiom in enumerate(axioms, 1):
            print(f"{i}. {axiom}")
            self.assumptions.append(axiom)
    
    def derive_propositions(self):
        """
        Logically derive propositions from axioms
        """
        print("\nDERIVED PROPOSITIONS")
        print("="*70)
        print("\nFrom the stated axioms, the following propositions follow logically:\n")
        
        # Example structure
        # Proposition 1: If A and B, then C
        # Proof: [logical steps]
        
        pass
    
    def explore_implications(self, scenario):
        """
        Explore theoretical implications
        """
        print(f"\nTHEORETICAL IMPLICATIONS: {scenario}")
        print("="*70)
        
        # Logical analysis of what theory predicts
        
        print("\nIF assumptions hold, THEN theory predicts:")
        print("  - [Prediction 1]")
        print("  - [Prediction 2]")
        print("\nThese predictions can be tested empirically in future research")
    
    def illustrate_with_example(self):
        """
        Optional: Illustrate theoretical concepts
        
        NOTE: This is illustration, not empirical testing
        """
        print("\nILLUSTRATIVE EXAMPLE")
        print("="*70)
        print("NOTE: This is for illustration only, not empirical evidence")
        
        # Generate illustrative data
        # Visualize theoretical relationships
        # Show how theory works
        
        print("\nThis example illustrates the theoretical concept")
        print("Empirical validation required to test theory against reality")
    
    def compare_to_alternative_theories(self, alternatives):
        """
        Compare theoretical predictions
        """
        print("\nCOMPARISON TO ALTERNATIVE THEORIES")
        print("="*70)
        
        for theory_name, predictions in alternatives.items():
            print(f"\n{theory_name} predicts:")
            for prediction in predictions:
                print(f"  - {prediction}")
        
        print("\nEmpirical research can test which theory better matches reality")
    
    def identify_testable_hypotheses(self):
        """
        CRITICAL: Identify how theory can be tested
        """
        print("\nTESTABLE HYPOTHESES FOR FUTURE RESEARCH")
        print("="*70)
        print("\nThis theory generates the following testable hypotheses:")
        
        # List specific, falsifiable predictions
        # Explain how they could be tested empirically
        
        print("\n1. Hypothesis: [specific prediction]")
        print("   Test: [how to test empirically]")
        print("\n2. Hypothesis: [specific prediction]")
        print("   Test: [how to test empirically]")
    
    def state_scope_and_limitations(self):
        """
        Define theory scope and limitations
        """
        print("\nSCOPE AND LIMITATIONS")
        print("="*70)
        
        print("\nSCOPE:")
        print("  - This theory applies to: [domain]")
        print("  - Under conditions: [boundary conditions]")
        
        print("\nLIMITATIONS:")
        print("  - Theoretical only - not empirically tested")
        print("  - Validity depends on assumptions")
        print("  - Simplified model of reality")
        print("  - May not capture all relevant factors")
        
        print("\nFUTURE WORK:")
        print("  - Empirical testing required")
        print("  - Refinement based on evidence")
        print("  - Extension to additional domains")
    
    def interpret_theoretical_contribution(self):
        """
        Explain theoretical contribution
        """
        print("\n" + "="*70)
        print("THEORETICAL CONTRIBUTION")
        print("="*70)
        
        print("\n✓ THIS STUDY CONTRIBUTES:")
        print("  - New theoretical framework")
        print("  - Logical analysis of implications")
        print("  - Testable predictions for future research")
        print("  - Conceptual clarification")
        
        print("\n⚠️ THIS STUDY DOES NOT:")
        print("  - Provide empirical evidence")
        print("  - Prove theory is correct")
        print("  - Test theory against reality")
        print("  - Make claims about real-world phenomena")
        
        print("\n✓ NEXT STEPS:")
        print("  - Empirical studies to test predictions")
        print("  - Refinement based on evidence")
        print("  - Application to real-world problems")

# Example: Theoretical model of information diffusion
"""
class InformationDiffusionTheory(TheoreticalModel):
    def develop_theory(self):
        self.state_theoretical_framework()
        
        # Define constructs
        constructs = {
            'Information Value': {
                'definition': 'Perceived utility of information to recipient',
                'properties': ['Non-negative', 'Context-dependent', 'Subjective']
            },
            'Transmission Cost': {
                'definition': 'Effort required to transmit information',
                'properties': ['Non-negative', 'Depends on medium', 'May be asymmetric']
            }
        }
        self.define_constructs(constructs)
        
        # State axioms
        axioms = [
            "Axiom 1: Agents transmit information when perceived value exceeds cost",
            "Axiom 2: Information value decreases with redundancy",
            "Axiom 3: Network structure affects transmission costs"
        ]
        self.state_axioms(axioms)
        
        # Derive propositions
        # ... logical derivations ...
        
        # Identify testable hypotheses
        self.identify_testable_hypotheses()
        
        # State scope and limitations
        self.state_scope_and_limitations()
        
        self.interpret_theoretical_contribution()

theory = InformationDiffusionTheory()
theory.develop_theory()
"""
```

### Critical Points for Theoretical Research

**Appropriate for:**
- Theory development
- Logical analysis
- Mathematical models
- Conceptual frameworks

**NOT appropriate for:**
- Making empirical claims without testing
- Claiming theory is "proven" without evidence
- Substituting for empirical research

**Must include:**
- Clear statement this is theoretical work
- Identification of testable predictions
- Acknowledgment that empirical testing is needed

---

## Summary Table: When to Use Each Research Type

| Research Type | Real Data Required? | Synthetic Data OK? | Causal Claims? | Purpose |
|--------------|--------------------|--------------------|---------------|---------|
| Correlational | ✅ YES | ❌ NO | ❌ NO | Explore relationships |
| Comparative | ✅ YES | ❌ NO | ❌ NO (groups not random) | Compare groups |
| Time Series | ✅ YES | ❌ NO | ❌ NO (temporal ≠ causal) | Analyze trends |
| Observational | ✅ YES | ❌ NO | ❌ NO | Describe phenomena |
| Meta-Analysis | ✅ YES | ❌ NO | Depends on primary studies | Synthesize research |
| Simulation | ⚠️ Model-based | ⚠️ YES (with caveats) | Only within model | Explore scenarios |
| Methodological | Optional | ✅ YES (appropriate!) | ❌ NO (about methods) | Test methods |
| Theoretical | ❌ NO | N/A (illustrative only) | ❌ NO (needs testing) | Develop theory |

---

## Summary: Non-Empirical Research Types

The three research types in this document have different data requirements than empirical research:

### Quick Comparison

| Type | Synthetic Data OK? | Claims About | Validation Needed? |
|------|-------------------|--------------|-------------------|
| Simulation | ⚠️ YES (with caveats) | Model behavior | ✅ YES (for real-world claims) |
| Methodological | ✅ YES (appropriate!) | Method performance | ❌ NO (testing method itself) |
| Theoretical | N/A (illustration only) | Logical implications | ✅ YES (for empirical testing) |

### Critical Distinctions

**Simulation Studies:**
- Purpose: Explore "what if" scenarios
- Data: Model-generated (conditional on assumptions)
- Claims: "IF assumptions hold, THEN..."
- Limitations: Results conditional on model validity
- **Before real-world claims**: MUST validate against reality

**Methodological Studies:**
- Purpose: Test statistical methods/algorithms
- Data: Synthetic with known properties (appropriate here!)
- Claims: About method performance
- Limitations: Says nothing about real phenomena
- **This is the ONLY type where synthetic data is fully appropriate**

**Theoretical Models:**
- Purpose: Develop theory and logical frameworks
- Data: Not required (or illustrative only)
- Claims: Theoretical predictions
- Limitations: Needs empirical testing
- **Future work**: Empirical studies to test predictions

---

## When to Use Each Type

### Use Simulation When:
- System too complex/expensive to study directly
- Want to explore scenarios before implementation
- Testing theoretical models
- **But remember**: Can't make real-world claims without validation

### Use Methodological When:
- Developing new statistical methods
- Comparing algorithm performance
- Testing method properties (power, bias)
- **This is appropriate use of synthetic data**

### Use Theoretical When:
- Developing new theoretical frameworks
- Exploring logical implications
- Building conceptual models
- **Follow with empirical testing**

---

## Critical Warnings

### ❌ DO NOT:
- Use simulation results as facts about reality (without validation)
- Use synthetic data from methodological studies to make empirical claims
- Present theoretical models as proven (without testing)
- Confuse model behavior with real-world behavior

### ✅ DO:
- State assumptions explicitly (simulation)
- Validate models against reality before claiming predictions
- Use synthetic data ONLY for methodological research
- Present theory as requiring empirical testing
- Be transparent about limitations

---

## Relationship to Empirical Research

**These types support but don't replace empirical research:**

- **Simulation** → Generates hypotheses for empirical testing
- **Methodological** → Develops tools for empirical research
- **Theoretical** → Provides frameworks for empirical investigation

**None can substitute for collecting real data when making claims about the world.**

---

## Next Steps

**See Part 1** (`RESEARCH_TYPES_PART1_EMPIRICAL.md`) for:
- Correlational Studies
- Comparative Studies
- Time Series Analysis
- Observational Studies
- Meta-Analysis

**See Index** (`RESEARCH_TYPES_INDEX.md`) for:
- Decision trees
- Quick reference tables
- Use case examples

**See Examples** in `examples/` directory for implementations

**Use Template** in `templates/research_template.py` to create studies

---

## Final Reminder

**The hierarchy of evidence:**

1. **Empirical data from reality** (strongest for real-world claims)
2. **Validated models** (simulations checked against reality)
3. **Methodological findings** (about tools, not phenomena)
4. **Theoretical predictions** (require empirical testing)

**For training AI agents:**
- Know which research type to use
- Match data requirements to research type
- Never use synthetic data for empirical claims
- Always validate before making real-world predictions
