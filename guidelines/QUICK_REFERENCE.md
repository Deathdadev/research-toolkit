# Quick Reference for AI Research Agents

## 1. Before Starting Any Research

**Ask yourself:**
- [ ] What is my research question?
- [ ] What type of research is this? (See decision tree below)
- [ ] Do I have access to appropriate data?
- [ ] Can others verify my results?

## 2. Research Type Decision Tree

```
Is your question about the REAL WORLD?
│
YES → Do you have REAL DATA?
│     │
│     YES → ✅ EMPIRICAL RESEARCH
│     │     Use: examples/01-05
│     │     Data: Real measurements only
│     │
│     NO → ❌ STOP
│           Get real data or change question
│
NO → Is it about METHODS/ALGORITHMS?
     │
     YES → ✅ METHODOLOGICAL RESEARCH
     │     Use: examples/07
     │     Data: Synthetic OK (testing method)
     │
     NO → Is it about THEORY/MODEL?
           │
           YES → ✅ THEORETICAL/SIMULATION
                 Use: examples/06, 08
                 Data: Model-based
                 Note: Can't claim real-world results without validation
```

## 3. Data Source Rules

### ✅ ALWAYS ACCEPTABLE for Empirical Research
- Public APIs (OpenWeatherMap, World Bank, etc.)
- Government databases (data.gov, Census, etc.)
- Scientific repositories (GenBank, PubMed, etc.)
- Published research datasets

### ⚠️ USE WITH CAUTION
- Web scraping (check terms of service)
- Synthetic data from real data (must validate)

### ❌ NEVER ACCEPTABLE for Empirical Research
- Made-up random data (`np.random.normal()`)
- Predetermined relationships
- Unverifiable sources

## 4. Mandatory Documentation

Every project MUST include:

```python
metadata = {
    'research_question': 'Specific, clear question',
    'research_type': 'empirical/methodological/theoretical',
    'data_sources': ['Source 1 URL', 'Source 2 URL'],
    'study_date': '2024-01-15T12:00:00',
    'methodology': 'Describe study design',
    'sample_size': 100,
    'limitations': ['Limitation 1', 'Limitation 2'],
}
```

Save:
- Raw data: `data.to_csv('raw_data.csv')`
- Metadata: `json.dump(metadata, file)`

## 5. Statistical Reporting Checklist

When reporting results, include:
- [ ] Sample size
- [ ] Test statistic (r, t, F, etc.)
- [ ] p-value
- [ ] Effect size
- [ ] Confidence interval
- [ ] Interpretation (careful about causation!)

## 6. Common Mistakes to AVOID

❌ **Don't:** Use synthetic data for empirical claims
✅ **Do:** Use real data from verifiable sources

❌ **Don't:** Claim "X causes Y" from correlation
✅ **Do:** Say "X is associated with Y"

❌ **Don't:** Report only significant results
✅ **Do:** Report all tests performed

❌ **Don't:** Overgeneralize beyond your sample
✅ **Do:** Limit claims to your data

❌ **Don't:** Hide limitations
✅ **Do:** State them explicitly

## 7. Research Workflow

```
1. Define question → Clear and specific
2. Choose type → Use decision tree
3. Collect data → Real data for empirical research
4. Validate data → Check for issues
5. Save raw data → For verification
6. Check assumptions → Before testing
7. Analyze → Appropriate methods
8. Report fully → All results, effect sizes
9. State limitations → Be honest
10. Provide verification → Instructions to reproduce
```

## 8. Quick Example Comparison

### ❌ WRONG: Synthetic Data for Empirical Claim
```python
# Made-up data
height = np.random.normal(170, 10, 100)
weight = np.random.normal(70, 15, 100)
r, p = stats.pearsonr(height, weight)
print("Height correlates with weight")  # ❌ NO!
# Problem: Data is made up, not real observations
```

### ✅ RIGHT: Real Data for Empirical Claim
```python
# Real data from API
response = requests.get('api_url')
data = pd.DataFrame(response.json())
r, p = stats.pearsonr(data['height'], data['weight'])
print(f"In our sample (n={len(data)}), height and weight")
print(f"are correlated: r={r:.3f}, p={p:.4f}")
# Good: Real observations, careful interpretation
```

## 9. Files to Reference

- **Main Guidelines:** `guidelines/AI_RESEARCH_GUIDELINES.md`
- **Template:** `templates/research_template.py`
- **Wrong approach:** `examples/00_synthetic_example_what_not_to_do.py`
- **Right approach:** `examples/01_correlational_study.py`

## 10. When in Doubt

**If you're unsure:**
1. Check the guidelines document
2. Look at example code
3. Ask: "Can others verify this?"
4. Ask: "Am I using real data for empirical claims?"
5. Ask: "Have I stated my limitations?"

**Remember:** Better to say "I cannot answer this without real data" than to make unverifiable claims.

## 11. Verification Checklist

Before publishing results:
- [ ] Research type clearly identified
- [ ] Appropriate data for research type
- [ ] Data sources documented
- [ ] Raw data saved (if empirical)
- [ ] Methodology documented
- [ ] All assumptions checked
- [ ] All results reported (not just significant)
- [ ] Effect sizes included
- [ ] Limitations stated
- [ ] Claims match evidence
- [ ] Verification instructions provided

## 12. Emergency Stop Signs

**STOP and reconsider if:**
- 🛑 You're using synthetic data to make real-world claims
- 🛑 You can't document your data sources
- 🛑 Others can't verify your results
- 🛑 You're claiming causation from correlation
- 🛑 You're hiding non-significant results
- 🛑 You can't state clear limitations
- 🛑 Your claims go beyond your data

---

**Core Principle:** Empirical research = verifiable claims about the observable world = requires real observations
