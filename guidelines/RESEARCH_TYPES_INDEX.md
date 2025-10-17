# Complete Research Types Index

## Overview

This index provides a complete reference to all research types an AI agent can conduct in Python, with clear guidelines on data requirements, use cases, and limitations.

## Document Structure

### Part 1: Empirical Research (Real Data Required)
**File:** `RESEARCH_TYPES_PART1_EMPIRICAL.md`

1. **Correlational Study** - Examining relationships between variables
2. **Comparative Study** - Comparing groups
3. **Time Series Analysis** - Analyzing temporal patterns
4. **Observational Study** - Describing phenomena
5. **Meta-Analysis** - Synthesizing existing research

### Part 2: Non-Empirical Research (Conditional/Methodological)
**File:** `RESEARCH_TYPES_PART2_NON_EMPIRICAL.md`

6. **Simulation Study** - Model-based exploration (conditional claims)
7. **Methodological Study** - Testing methods/algorithms (synthetic data appropriate)
8. **Theoretical Model** - Theory development (no empirical claims)

---

## Quick Decision Guide

### START HERE: What is your research question about?

```
┌─────────────────────────────────────────────────────────┐
│ Is your question about REAL-WORLD PHENOMENA?           │
└─────────────────────────┬───────────────────────────────┘
                          │
        ┌─────────────────┴───────────────┐
        │                                 │
       YES                               NO
        │                                 │
        ▼                                 ▼
┌───────────────────┐            ┌──────────────────┐
│ EMPIRICAL         │            │ What ARE you     │
│ RESEARCH          │            │ studying?        │
│                   │            └────────┬─────────┘
│ Use Part 1        │                     │
│ Real data         │         ┌───────────┼───────────┐
│ required          │         │           │           │
└───────────────────┘         │           │           │
                              ▼           ▼           ▼
                        ┌─────────┐ ┌─────────┐ ┌──────────┐
                        │ METHOD/ │ │ THEORY/ │ │ SYSTEM   │
                        │ALGORITHM│ │ MODEL   │ │ BEHAVIOR │
                        └────┬────┘ └────┬────┘ └────┬─────┘
                             │           │           │
                             ▼           ▼           ▼
                      ┌────────────┐ ┌─────────┐ ┌──────────┐
                      │METHODOLOG- │ │THEORET- │ │SIMULATION│
                      │ICAL STUDY  │ │ICAL     │ │ STUDY    │
                      │            │ │MODEL    │ │          │
                      │Part 2, #7  │ │Part 2,#8│ │Part 2, #6│
                      └────────────┘ └─────────┘ └──────────┘
```

---

## Research Type Quick Reference

### Empirical Research (Part 1)

#### 1. Correlational Study
**When:** Exploring relationships between variables  
**Data:** ✅ Real observations required  
**Claims:** Association only, NOT causation  
**Example:** "Is GDP correlated with life expectancy?"

#### 2. Comparative Study  
**When:** Comparing pre-existing groups  
**Data:** ✅ Real group data required  
**Claims:** Group differences, NOT causation  
**Example:** "Do coastal vs inland cities differ in temperature?"

#### 3. Time Series Analysis
**When:** Analyzing temporal patterns/trends  
**Data:** ✅ Real time-ordered data required  
**Claims:** Trends/patterns, NOT causation  
**Example:** "How has air quality changed over time?"

#### 4. Observational Study
**When:** Describing current state/characteristics  
**Data:** ✅ Real observations required  
**Claims:** Descriptive only  
**Example:** "What are the characteristics of successful startups?"

#### 5. Meta-Analysis
**When:** Synthesizing existing research  
**Data:** ✅ Real effect sizes from studies  
**Claims:** Overall effect across studies  
**Example:** "What is the average effect of intervention X?"

### Non-Empirical Research (Part 2)

#### 6. Simulation Study
**When:** Exploring theoretical scenarios  
**Data:** ⚠️ Model-generated (conditional)  
**Claims:** "IF assumptions hold, THEN..." (requires validation)  
**Example:** "Under model assumptions, how does epidemic spread?"

#### 7. Methodological Study
**When:** Testing methods/algorithms  
**Data:** ✅ Synthetic OK (testing method)  
**Claims:** About method performance  
**Example:** "What is the power of this statistical test?"

#### 8. Theoretical Model
**When:** Developing theory  
**Data:** ❌ Not required (theoretical)  
**Claims:** Logical implications (needs empirical testing)  
**Example:** "What does theory X predict under conditions Y?"

---

## Data Requirements Matrix

| Research Type | Real Data? | Synthetic Data? | Why? |
|--------------|-----------|-----------------|------|
| Correlational | ✅ REQUIRED | ❌ NOT VALID | Making claims about real relationships |
| Comparative | ✅ REQUIRED | ❌ NOT VALID | Making claims about real groups |
| Time Series | ✅ REQUIRED | ❌ NOT VALID | Making claims about real trends |
| Observational | ✅ REQUIRED | ❌ NOT VALID | Describing real phenomena |
| Meta-Analysis | ✅ REQUIRED | ❌ NOT VALID | Synthesizing real research |
| Simulation | ⚠️ CONDITIONAL | ⚠️ WITH CAVEATS | Exploring model, not claiming facts |
| Methodological | Optional | ✅ APPROPRIATE | Testing method, not studying world |
| Theoretical | ❌ NOT NEEDED | N/A | Pure theory, needs testing later |

---

## Can I Claim Causation?

| Research Type | Causal Claims? | Why/Why Not? |
|--------------|----------------|--------------|
| Correlational | ❌ NO | Variables not manipulated, confounds possible |
| Comparative | ❌ NO | Groups not randomly assigned |
| Time Series | ❌ NO | Temporal order ≠ causation |
| Observational | ❌ NO | No manipulation, descriptive only |
| Meta-Analysis | ⚠️ DEPENDS | Depends on design of primary studies |
| Simulation | ❌ NO* | *Within model only, not real world |
| Methodological | ❌ NO | Testing methods, not phenomena |
| Theoretical | ❌ NO | Theory needs empirical testing |

**For Causal Claims:** Need randomized controlled experiments (not covered in these guides, as AI agents typically cannot conduct true experiments in real-world settings)

---

## Use Case Examples

### Scenario 1: "I want to study if X is related to Y"
→ **Correlational Study** (Part 1, #1)
- Need: Real data on X and Y
- Result: Correlation coefficient, p-value
- Claim: "X and Y are associated"
- Cannot claim: "X causes Y"

### Scenario 2: "I want to compare Group A vs Group B"
→ **Comparative Study** (Part 1, #2)
- Need: Real data from both groups
- Result: Mean differences, effect sizes
- Claim: "Groups differ on variable Y"
- Cannot claim: "Being in Group A causes Y"

### Scenario 3: "I want to analyze how something changed over time"
→ **Time Series Analysis** (Part 1, #3)
- Need: Real temporal data
- Result: Trends, seasonality, forecasts
- Claim: "Variable X shows upward trend"
- Cannot claim: "Event Z caused the trend"

### Scenario 4: "I want to describe current characteristics"
→ **Observational Study** (Part 1, #4)
- Need: Real observations
- Result: Descriptive statistics, patterns
- Claim: "Population has characteristics X, Y, Z"
- Cannot claim: Anything causal

### Scenario 5: "I want to combine results from multiple studies"
→ **Meta-Analysis** (Part 1, #5)
- Need: Real effect sizes from published research
- Result: Pooled effect size
- Claim: "Overall effect is X across studies"
- Cannot claim: Beyond what primary studies support

### Scenario 6: "I want to test what happens IF certain conditions hold"
→ **Simulation Study** (Part 2, #6)
- Need: Theoretical model
- Result: Model predictions
- Claim: "IF assumptions hold, THEN model predicts..."
- Cannot claim: Real-world outcomes without validation

### Scenario 7: "I want to test if a statistical method works well"
→ **Methodological Study** (Part 2, #7)
- Need: Synthetic data with known properties
- Result: Method performance metrics
- Claim: "Method has X% power under conditions Y"
- Cannot claim: Anything about real-world phenomena

### Scenario 8: "I want to develop a new theory"
→ **Theoretical Model** (Part 2, #8)
- Need: Logical analysis
- Result: Theoretical propositions
- Claim: "Theory predicts..." (logically)
- Cannot claim: Theory is proven without empirical testing

---

## Implementation Examples

All research types include:
- Full Python implementation templates
- Data collection/generation appropriate to type
- Statistical analysis methods
- Visualization approaches
- Limitations statements
- Proper interpretation guidelines

See respective guide documents for complete code examples.

---

## Critical Guidelines for AI Agents

### Before Starting Any Research:

1. **Identify research type** using decision tree
2. **Check data requirements** - do you need real data?
3. **Understand claim limitations** - what can/cannot be concluded?
4. **Plan verification** - how can others check your work?

### During Research:

1. **Collect appropriate data** for research type
2. **Document everything** - sources, methods, assumptions
3. **Use appropriate statistics** for your design
4. **Save raw data** (if empirical research)

### After Analysis:

1. **State limitations explicitly**
2. **Match claims to evidence**
3. **Provide verification instructions**
4. **Interpret cautiously** - don't overstate

### Common Mistakes to Avoid:

❌ Using synthetic data for empirical claims  
❌ Claiming causation from correlation  
❌ Overgeneralizing beyond sample  
❌ Hiding limitations  
❌ Making claims beyond what data supports  

---

## Where to Start

**For AI Agents:**
1. Read: `guidelines/AI_RESEARCH_GUIDELINES.md` (main guidelines)
2. Use: Decision tree to determine research type
3. Study: Appropriate research type guide (Part 1 or 2)
4. Use: `templates/research_template.py` as starting point
5. Check: `guidelines/QUICK_REFERENCE.md` for checklists

**For Understanding Examples:**
1. Wrong approach: `examples/00_synthetic_example_what_not_to_do.py`
2. Right approach: `examples/01_correlational_study.py`
3. More examples: Coming soon for each research type

---

## Additional Resources

- **Main Guidelines:** `guidelines/AI_RESEARCH_GUIDELINES.md`
- **Quick Reference:** `guidelines/QUICK_REFERENCE.md`
- **Template:** `templates/research_template.py`
- **Discussion:** `docs/synthetic_data_discussion.md`
- **Training Guide:** `docs/AI_TRAINING_GUIDE.md`

---

## Contributing New Examples

To add examples for remaining research types:
1. Copy `templates/research_template.py`
2. Follow guidelines for specific research type
3. Implement appropriate data collection
4. Include complete documentation
5. Add verification instructions
6. Save to `examples/` with descriptive name

**Needed examples:**
- [ ] Comparative study (02)
- [ ] Time series analysis (03)
- [ ] Observational study (04)
- [ ] Meta-analysis (05)
- [ ] Simulation study (06)
- [ ] Methodological study (07)
- [ ] Theoretical model (08)

---

## Remember

**The goal is not just to produce results, but to produce VERIFIABLE, REPRODUCIBLE, TRUSTWORTHY research that advances knowledge.**

Different research types have different standards, but all require:
- Transparency
- Appropriate methods
- Honest limitations
- Careful interpretation
- Ethical conduct

Choose the right research type for your question, follow the guidelines, and always prioritize validity over convenience.
