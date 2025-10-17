# AI Agent Research Guidelines

## Purpose
This document provides clear, actionable guidelines for AI agents to conduct valid empirical research.

## Core Principle

**EMPIRICAL RESEARCH = MAKING VERIFIABLE CLAIMS ABOUT THE OBSERVABLE WORLD**

All research must be:
1. **Verifiable** - Others can check your work
2. **Reproducible** - Others can repeat your methods
3. **Falsifiable** - Your claims can be tested and potentially disproven
4. **Transparent** - All methods, data, and limitations are documented

---

## Decision Tree: What Type of Research Should I Conduct?

```
START: What is your research question about?
│
├─ About the REAL WORLD (phenomena, relationships, effects)
│  │
│  ├─ Can you access REAL DATA from the world?
│  │  │
│  │  ├─ YES → Use EMPIRICAL RESEARCH (examples 01-05)
│  │  │         - Correlational Study
│  │  │         - Comparative Study  
│  │  │         - Time Series Analysis
│  │  │         - Observational Study
│  │  │         - Meta-Analysis
│  │  │
│  │  └─ NO → ❌ STOP: Cannot make empirical claims without real data
│  │            Consider: Can you get real data? If not, reconsider question.
│  │
│  └─ Testing THEORETICAL MODEL
│     │
│     └─ Use SIMULATION RESEARCH (example 06)
│         BUT: Make NO claims about real world without validation
│
├─ About STATISTICAL METHODS or ALGORITHMS
│  │
│  └─ Use METHODOLOGICAL RESEARCH (example 07)
│      - Synthetic data is acceptable here
│      - You're testing the METHOD, not studying the world
│
└─ About LOGICAL IMPLICATIONS of assumptions
   │
   └─ Use THEORETICAL RESEARCH (example 08)
       - State assumptions clearly
       - Present as "IF-THEN" statements
       - No claims about reality without validation
```

---

## Research Type Classifications

### Type 1: EMPIRICAL RESEARCH (Real Data Required)

**Definition:** Research that makes claims about observable phenomena in the real world.

**Data Requirement:** ✅ MUST use real, verifiable data

**Examples:**
- "Is population density correlated with air quality?"
- "Do countries with higher GDP have better health outcomes?"
- "Is there a relationship between temperature and crime rates?"

**Checklist:**
- [ ] Research question is about real-world phenomena
- [ ] Data comes from actual observations/measurements
- [ ] Data sources are documented and verifiable
- [ ] Raw data is saved for independent verification
- [ ] Statistical methods are appropriate
- [ ] Limitations are explicitly stated
- [ ] Claims are falsifiable

**See:** `examples/01_correlational_study.py` to `examples/05_meta_analysis.py`

---

### Type 2: METHODOLOGICAL RESEARCH (Synthetic Data Acceptable)

**Definition:** Research about statistical methods, algorithms, or computational techniques.

**Data Requirement:** ✅ Can use synthetic data (you're studying the method)

**Examples:**
- "What is the statistical power of this test?"
- "How does this algorithm scale with data size?"
- "Does this estimation method produce unbiased results?"

**Checklist:**
- [ ] Research question is about a METHOD/ALGORITHM, not the world
- [ ] Clearly state you're testing a method
- [ ] Synthetic data generation process is documented
- [ ] Results are about method performance, not world phenomena
- [ ] No claims about real-world relationships

**See:** `examples/07_methodological_study.py`

---

### Type 3: THEORETICAL RESEARCH (Conditional Claims)

**Definition:** Research exploring logical consequences of assumptions.

**Data Requirement:** ⚠️ Can use synthetic/simulated data, BUT only for theoretical exploration

**Examples:**
- "IF agents behave according to model X, THEN what patterns emerge?"
- "What are the mathematical properties of system Y?"
- "Under assumptions A, B, C, what does theory predict?"

**Checklist:**
- [ ] All assumptions are explicitly stated
- [ ] Claims are presented as conditional (IF-THEN)
- [ ] No claims about reality without real-world validation
- [ ] Clear boundary between theory and empirical claims
- [ ] Limitations of assumptions acknowledged

**See:** `examples/08_theoretical_model.py`

---

### Type 4: SIMULATION RESEARCH (Model-Based)

**Definition:** Using computational models to explore system behavior.

**Data Requirement:** ⚠️ Synthetic data acceptable, BUT must validate against reality for real-world claims

**Examples:**
- "How does an epidemic spread under different intervention strategies?"
- "What traffic patterns emerge from these driver behaviors?"
- "How does climate change affect crop yields under various scenarios?"

**Checklist:**
- [ ] Model assumptions are clearly documented
- [ ] Parameters are justified (ideally from real data)
- [ ] Results presented as model outputs, not facts
- [ ] Model validated against real data when possible
- [ ] Uncertainty and sensitivity analysis included
- [ ] Clear about what model can and cannot predict

**See:** `examples/06_simulation_study.py`

---

## Data Source Requirements

### ✅ ACCEPTABLE Data Sources for Empirical Research

1. **Public APIs with real measurements**
   - Weather/climate data (OpenWeatherMap, NOAA)
   - Economic indicators (World Bank, FRED)
   - Demographics (Census, UN Data)
   - Health statistics (WHO, CDC)

2. **Government databases**
   - data.gov, data.gov.uk, etc.
   - National statistical offices
   - Regulatory agency data

3. **Scientific repositories**
   - GenBank, PubChem (biological data)
   - arXiv datasets
   - Published research datasets

4. **Public datasets**
   - UCI Machine Learning Repository
   - Kaggle datasets (with proper attribution)
   - Open research data

5. **Web scraping (with conditions)**
   - Publicly accessible data
   - Compliant with terms of service
   - Properly attributed

### ❌ UNACCEPTABLE for Empirical Research

1. **Made-up synthetic data**
   - `np.random.normal()` without real-world basis
   - Arbitrary number generation
   - Predetermined relationships

2. **Unverifiable sources**
   - "I heard that..."
   - Anonymous claims
   - Cannot be independently checked

3. **Private/proprietary data without verification**
   - Cannot be independently accessed
   - No way for others to verify

---

## Mandatory Documentation

### Every Research Project MUST Include:

1. **Research Question** (specific and clear)
   ```
   "Is there a relationship between X and Y in population Z?"
   ```

2. **Data Sources** (complete documentation)
   ```
   Source: OpenWeatherMap API v2.5
   URL: https://openweathermap.org/api
   Access date: 2024-01-15
   Method: REST API calls
   ```

3. **Raw Data** (saved for verification)
   ```python
   data.to_csv('raw_data.csv', index=False)
   with open('metadata.json', 'w') as f:
       json.dump(metadata, f)
   ```

4. **Methodology** (transparent and reproducible)
   ```
   Study Design: Cross-sectional correlational study
   Sample: 25 cities across 6 continents
   Variables: Population density (independent), PM2.5 (dependent)
   Analysis: Pearson correlation, linear regression
   Software: Python 3.10, scipy 1.10
   ```

5. **Limitations** (honest and explicit)
   ```
   Limitations:
   - Cross-sectional design (no causation)
   - Single time point
   - Confounding variables not controlled
   - Sample limited to major cities
   ```

6. **Verification Instructions**
   ```
   To reproduce:
   1. Install dependencies: pip install -r requirements.txt
   2. Get API key from: [URL]
   3. Run: python research.py
   4. Compare with raw_data.csv
   ```

---

## Statistical Analysis Requirements

### Before Conducting Analysis:

1. **Check sample size**
   - Is it adequate for your analysis?
   - Power analysis when appropriate

2. **Check assumptions**
   - Normality (when required)
   - Independence of observations
   - Homogeneity of variance
   - Linearity (for regression)

3. **Choose appropriate tests**
   - Parametric vs non-parametric
   - One-tailed vs two-tailed
   - Multiple comparison corrections

### When Reporting Results:

1. **Report effect sizes** (not just p-values)
   ```python
   print(f"Correlation: r = {r:.3f}, p = {p:.6f}")
   print(f"Effect size: {effect_size:.3f}")
   ```

2. **Report confidence intervals**
   ```python
   print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
   ```

3. **Report all tests** (not just significant ones)
   - No p-hacking
   - No selective reporting

4. **Interpret carefully**
   - Correlation ≠ causation
   - Statistical significance ≠ practical importance
   - Consider alternative explanations

---

## Common Pitfalls to AVOID

### ❌ 1. P-Hacking
```python
# DON'T do this:
for var in all_variables:
    if stats.pearsonr(x, var)[1] < 0.05:
        print(f"Significant result with {var}!")  # Cherry-picking
```

### ❌ 2. Claiming Causation from Correlation
```python
# DON'T say this:
"Higher density CAUSES higher pollution"

# DO say this:
"Higher density is ASSOCIATED with higher pollution.
This correlation does not prove causation.
Possible confounds: industrial activity, regulations, geography."
```

### ❌ 3. Overgeneralizing
```python
# DON'T say this:
"All cities worldwide show this pattern"

# DO say this:
"Among the 25 cities studied, we observed this pattern.
Generalization to other cities requires further research."
```

### ❌ 4. Ignoring Non-Significant Results
```python
# DON'T hide non-significant results
# DO report them:
print(f"No significant relationship found between X and Y")
print(f"r = {r:.3f}, p = {p:.3f}, suggesting...")
```

### ❌ 5. Using Synthetic Data for Empirical Claims
```python
# DON'T do this:
data = np.random.normal(100, 15, 1000)  # Made up
print("This proves people's IQ averages 100")  # ❌ NO!

# DO this:
data = load_real_iq_data()  # Real measurements
print("Among our sample, mean IQ = {data.mean():.1f}")  # ✓ YES
```

---

## Workflow for AI Agents

### Step 1: Understand the Question
```
Input: Research question
Output: Classification of research type

Questions to ask:
- Is this about the real world or about methods/theory?
- Can I access real data to answer this?
- What type of study design is appropriate?
```

### Step 2: Choose Research Type
```
Use decision tree above to determine:
- Empirical (real data required)
- Methodological (synthetic OK)
- Theoretical (conditional claims)
- Simulation (model-based)
```

### Step 3: Collect/Generate Data
```
IF Empirical:
    - Identify public data sources
    - Document source URLs and access methods
    - Collect real data
    - Save raw data with timestamps

ELSE IF Methodological:
    - Generate synthetic data for method testing
    - Document generation process
    - State clearly this is method testing

ELSE IF Theoretical/Simulation:
    - State assumptions explicitly
    - Generate data based on model
    - Present as conditional exploration
```

### Step 4: Analyze Data
```
- Check assumptions
- Use appropriate statistical tests
- Report effect sizes and CIs
- Interpret carefully
```

### Step 5: Document Everything
```
- Research question
- Data sources
- Methodology
- Raw data
- Analysis code
- Results
- Limitations
- Verification instructions
```

### Step 6: State Conclusions Carefully
```
- Match claims to evidence
- Acknowledge limitations
- Don't overstate findings
- Provide verification path
```

---

## Quick Reference: Can I Use This Data?

| Data Type | Empirical Research | Methodological | Theoretical |
|-----------|-------------------|----------------|-------------|
| Real measurements from APIs | ✅ YES | ✅ YES | ✅ YES |
| Government datasets | ✅ YES | ✅ YES | ✅ YES |
| Scientific repositories | ✅ YES | ✅ YES | ✅ YES |
| Published research data | ✅ YES | ✅ YES | ✅ YES |
| Synthetic from real data | ⚠️ CAUTION* | ✅ YES | ✅ YES |
| Pure random generation | ❌ NO | ✅ YES** | ⚠️ CAUTION*** |
| Made-up data | ❌ NO | ✅ YES** | ⚠️ CAUTION*** |

*Only if preserves real data properties and is validated
**Only if studying the method itself
***Only if presented as theoretical exploration with stated assumptions

---

## Examples to Study

Each example in the `examples/` directory demonstrates a specific research type:

- `00_synthetic_example_what_not_to_do.py` - ❌ Wrong approach
- `01_correlational_study.py` - ✅ Empirical: Correlational
- `02_comparative_study.py` - ✅ Empirical: Group comparison
- `03_time_series_analysis.py` - ✅ Empirical: Temporal patterns
- `04_observational_study.py` - ✅ Empirical: Descriptive
- `05_meta_analysis.py` - ✅ Empirical: Synthesizing studies
- `06_simulation_study.py` - ⚠️ Theoretical: Model-based
- `07_methodological_study.py` - ✅ Methods: Testing techniques
- `08_theoretical_model.py` - ⚠️ Theory: Exploring implications

---

## Final Checklist Before Publishing Research

- [ ] Research type clearly identified
- [ ] Data source appropriate for research type
- [ ] All data sources documented
- [ ] Raw data saved
- [ ] Methodology transparent
- [ ] Assumptions checked
- [ ] Results reported completely (including non-significant)
- [ ] Effect sizes and CIs included
- [ ] Limitations explicitly stated
- [ ] Claims match evidence
- [ ] No overgeneralization
- [ ] Verification instructions provided
- [ ] Code is reproducible

---

## Questions to Ask Yourself

1. **Am I making claims about the real world?**
   - YES → I need real data
   - NO → What am I making claims about?

2. **Can others verify my results?**
   - YES → Good
   - NO → Need better documentation

3. **Are my claims falsifiable?**
   - YES → Good
   - NO → Rethink your claims

4. **Have I stated my limitations?**
   - YES → Good
   - NO → Add them now

5. **Am I being transparent?**
   - YES → Good
   - NO → What am I hiding or omitting?

---

## Remember

**The goal is not just to produce results, but to produce VERIFIABLE, REPRODUCIBLE, TRUSTWORTHY results that advance knowledge.**

Better to say "I don't know" or "I can't answer this without real data" than to make unverifiable claims.
