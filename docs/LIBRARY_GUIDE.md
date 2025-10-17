# Research Toolkit - Library Guide

## Overview

The Research Toolkit is a comprehensive Python library for conducting research across all methodologies with proper scientific notation, encoding-safe output, APA 7 referencing, and AI integration.

**Version:** 2.0.0

---

## Installation

### From Source

```bash
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

### With Example Dependencies

```bash
pip install -e ".[examples]"
```

---

## Key Features

### 1. Encoding-Safe Output

Automatically handles Unicode/ASCII output based on console capabilities:

```python
from research_toolkit import SafeOutput

# Automatically uses ASCII fallback if Unicode not supported
SafeOutput.safe_print("Temperature: 25°C")  # Console-safe
SafeOutput.safe_print("Significance: α = 0.05")  # Console-safe
```

### 2. Scientific Notation

Proper Greek symbols for reports, ASCII for console:

```python
from research_toolkit import ScientificNotation, get_symbol

# Get symbols
alpha = get_symbol('alpha', use_unicode=True)   # 'α'
alpha_ascii = get_symbol('alpha', use_unicode=False)  # 'alpha'

# Format with units
temp = ScientificNotation.format_units(25.5, "{degree}C", use_unicode=True)
# Returns: "25.5 °C" (Unicode) or "25.5 degC" (ASCII)

pm25 = ScientificNotation.format_units(35.2, "{mu}g/m{cubed}", use_unicode=True)
# Returns: "35.2 μg/m³" (Unicode) or "35.2 ug/m^3" (ASCII)
```

### 3. Report Formatting

Professional report generation with automatic encoding handling:

```python
from research_toolkit import ReportFormatter

formatter = ReportFormatter()  # Auto-detects Unicode support

# Section headers
formatter.print_section("RESEARCH RESULTS")
formatter.print_subsection("Statistical Analysis")

# Statistical results with proper notation
formatter.print_statistical_result('alpha', 0.05, decimals=2, use_greek=True)
# Prints: "α = 0.05" (if Unicode supported) or "alpha = 0.05" (ASCII)
```

### 4. Statistical Formatting (APA 7 Compliant)

Complete APA 7 statistical formatters with 9 methods:

```python
from research_toolkit import StatisticalFormatter

# P-values
p_str = StatisticalFormatter.format_p_value(0.0234)  # "p = 0.023"
p_str = StatisticalFormatter.format_p_value(0.0001)  # "p < 0.001"

# Confidence intervals
ci = StatisticalFormatter.format_ci(1.23, 4.56, decimals=2)  # "[1.23, 4.56]"

# Mean and SD
stats = StatisticalFormatter.format_mean_sd(10.5, 2.3)  # "M = 10.50, SD = 2.30"

# Effect size interpretation
interp = StatisticalFormatter.interpret_effect_size(0.6, 'cohens_d')  # "medium"
```

---

## Complete Usage Examples

### Example 1: Basic Research Report

```python
from research_toolkit import ReportFormatter, SafeOutput, get_symbol
import pandas as pd
import numpy as np
from scipy import stats

# Initialize formatter
formatter = ReportFormatter()

# Print header
formatter.print_section("CORRELATION STUDY: TEMPERATURE VS POLLUTION")

# Collect data
formatter.print_subsection("Data Collection")
SafeOutput.safe_print(f"Temperature range: 15°C to 35°C")
SafeOutput.safe_print(f"PM2.5 range: 10 μg/m³ to 75 μg/m³")

# Analysis
formatter.print_subsection("Statistical Analysis")
SafeOutput.safe_print(f"Significance level: {get_symbol('alpha')} = 0.05")

# Results
data1 = np.random.normal(25, 5, 100)
data2 = data1 * 2 + np.random.normal(0, 5, 100)
r, p = stats.pearsonr(data1, data2)

formatter.print_subsection("Results")
print(f"Correlation: r = {r:.3f}, p = {p:.4f}")
```

### Example 2: Saving Reports with Unicode

```python
from research_toolkit import ReportFormatter

formatter = ReportFormatter()

# Build report content with Unicode symbols
report = """
RESEARCH REPORT
===============

Statistical Results:
- Significance level: α = 0.05
- Temperature: 25.5°C
- PM2.5: 35.2 μg/m³
- Effect size: β = 0.45
"""

# Save with Unicode (for PDF, publications)
formatter.save_report('report_unicode.txt', report, use_unicode=True)

# Save with ASCII (for plain text, universal compatibility)
formatter.save_report('report_ascii.txt', report, use_unicode=False)
```

### Example 3: Custom Formatting Functions

```python
from research_toolkit import format_temperature, format_pm25

# Temperature formatting
temp_unicode = format_temperature(25.5, use_unicode=True)   # "25.50°C"
temp_ascii = format_temperature(25.5, use_unicode=False)    # "25.50degC"

# PM2.5 formatting
pm_unicode = format_pm25(35.2, use_unicode=True)   # "35.2 μg/m³"
pm_ascii = format_pm25(35.2, use_unicode=False)    # "35.2 ug/m^3"
```

---

## Available Symbols

### Greek Letters

| Name | Unicode | ASCII |
|------|---------|-------|
| alpha | α | alpha |
| beta | β | beta |
| gamma | γ | gamma |
| delta | δ | delta |
| epsilon | ε | epsilon |
| mu | μ | mu |
| sigma | σ | sigma |
| rho | ρ | rho |
| tau | τ | tau |
| chi | χ | chi |
| phi | φ | phi |
| omega | ω | omega |

### Special Symbols

| Name | Unicode | ASCII |
|------|---------|-------|
| degree | ° | deg |
| squared | ² | ^2 |
| cubed | ³ | ^3 |
| plusminus | ± | +/- |
| leq | ≤ | <= |
| geq | ≥ | >= |
| neq | ≠ | != |
| checkmark | ✓ | [OK] |
| cross | ❌ | [X] |

---

## Best Practices

### 1. Console Output

**Always use `SafeOutput.safe_print()` for Unicode text:**

```python
from research_toolkit import SafeOutput

# GOOD - Automatically handles encoding
SafeOutput.safe_print("Temperature: 25°C")

# BAD - May crash on some systems
print("Temperature: 25°C")
```

### 2. File Output

**Use `SafeOutput.safe_file_output()` context manager:**

```python
from research_toolkit import SafeOutput

# GOOD - Proper encoding handling
with SafeOutput.safe_file_output('results.txt', 'w', 'utf-8') as f:
    f.write("Temperature: 25°C\n")
    f.write("Significance: α = 0.05\n")

# Also GOOD - Use formatter for reports
formatter = ReportFormatter()
formatter.save_report('report.txt', content, use_unicode=True)
```

### 3. Scientific Reports

**Preserve Greek symbols in scientific reports:**

```python
# For publications, PDFs, scientific reports
formatter.save_report('publication.txt', report_content, use_unicode=True)

# For console output (auto-detection)
SafeOutput.safe_print("Results: α = 0.05, β = 0.45")
```

### 4. Universal Compatibility

**Use ASCII when maximum compatibility needed:**

```python
# For plain text files, emails, legacy systems
formatter.save_report('compatibility.txt', report_content, use_unicode=False)
```

---

## Integration with Examples

All examples in `/examples/` can be updated to use the utilities:

### Before (Hardcoded Unicode):

```python
print(f"Temperature: 25.5°C")  # May crash
print(f"Significance: α = 0.05")  # May crash
```

### After (Using Toolkit):

```python
from research_toolkit import SafeOutput, get_symbol

SafeOutput.safe_print(f"Temperature: 25.5°C")  # Safe
SafeOutput.safe_print(f"Significance: {get_symbol('alpha')} = 0.05")  # Safe
```

---

## API Reference

### SafeOutput Class

#### Methods:

- `can_encode_unicode() -> bool`: Check if stdout supports Unicode
- `safe_print(text: str, fallback_text: str = None, end: str = '\\n')`: Print with auto-fallback
- `safe_file_output(filename: str, mode: str, encoding: str)`: Context manager for files
- `format_for_output(text: str, target: str) -> str`: Format for 'console' or 'file'

### ScientificNotation Class

#### Methods:

- `get_symbol(name: str, use_unicode: bool) -> str`: Get a scientific symbol
- `format_units(value: float, unit: str, use_unicode: bool) -> str`: Format value with units

### ReportFormatter Class

#### Methods:

- `__init__(use_unicode_console: bool = None)`: Initialize (auto-detects Unicode)
- `print_section(title: str, width: int = 70)`: Print section header
- `print_subsection(title: str)`: Print subsection header
- `print_statistical_result(statistic: str, value: float, decimals: int, use_greek: bool)`: Print stat result
- `save_report(filename: str, content: str, use_unicode: bool)`: Save report to file

### StatisticalFormatter Class

#### Methods (9 APA 7 Compliant Formatters):

- `format_p_value(p: float, threshold: float = 0.001) -> str`: Format p-value (no leading zero)
- `format_ci(lower: float, upper: float, confidence: int = 95, decimals: int = 2) -> str`: Format confidence interval
- `format_mean_sd(mean: float, sd: float, decimals: int = 2) -> str`: Format mean and standard deviation
- `format_correlation(r: float, p: float, n: int, decimals: int = 2) -> str`: Format correlation with df
- `format_t_test(t: float, df: int, p: float, decimals: int = 2) -> str`: Format t-test results
- `format_f_test(f: float, df1: int, df2: int, p: float, decimals: int = 2) -> str`: Format F-test results
- `format_chi_square(chi2: float, df: int, p: float, decimals: int = 2) -> str`: Format chi-square results
- `format_effect_size(effect_size: float, measure: str, decimals: int = 2) -> str`: Format effect size with interpretation
- `interpret_effect_size(effect_size: float, measure: str) -> str`: Interpret effect size magnitude

---

## APA 7 Reference Management (v2.0.0)

### APA7ReferenceManager Class

Comprehensive reference management with 10 reference types and advanced name parsing.

#### Supported Reference Types:

1. **journal** - Journal articles
2. **book** - Books
3. **chapter** - Book chapters
4. **website** - Web pages
5. **report** - Reports
6. **conference** - Conference proceedings
7. **dataset** - Datasets
8. **software** - Software
9. **dissertation** - Dissertations/theses
10. **government** - Government documents

#### Key Methods:

- `add_reference(ref_type: str, **fields) -> str`: Add reference, returns citation key
- `format_reference(key: str) -> str`: Format reference in APA 7 style
- `get_in_text_citation(keys: list, page: str = None, narrative: bool = False) -> str`: Generate in-text citation (v2.0.0)
- `export_bibtex(filename: str)`: Export references to BibTeX
- `validate_reference(key: str) -> dict`: Check reference completeness
- `parse_author_name(name: str) -> dict`: Parse author name into components (v2.0.0)
- `parse_multiple_authors(author_string: str) -> list`: Parse multiple authors (v2.0.0)
- `format_authors_for_citation(authors: list) -> str`: Format for in-text citation (v2.0.0)
- `format_authors_for_reference(authors: list) -> str`: Format for reference list (v2.0.0)

#### In-Text Citations (v2.0.0)

**IMPORTANT:** Use `get_in_text_citation()` to generate proper APA 7 citations. Do **NOT** insert citation keys directly into text.

```python
from research_toolkit import APA7ReferenceManager

manager = APA7ReferenceManager()

# Add reference
github_ref = manager.add_reference('website',
    author='GitHub',
    year='2024',
    title='GitHub REST API',
    url='https://docs.github.com/en/rest'
)

# ❌ WRONG - Don't use citation keys directly:
print(f"Data from GitHub API ({github_ref})")
# Output: "Data from GitHub API (ref1)"  <- Placeholder reference!

# ✅ CORRECT - Use get_in_text_citation():
print(f"Data from GitHub API {manager.get_in_text_citation([github_ref])}")
# Output: "Data from GitHub API (GitHub, 2024)"  <- Proper APA 7 citation!

# Examples with different author counts:
# Single author
citation1 = manager.get_in_text_citation([github_ref])
# Returns: "(GitHub, 2024)"

# Two authors (uses ampersand)
two_author_ref = manager.add_reference('journal',
    author='Smith, J.; Jones, K.',
    year='2023',
    title='Research methods',
    journal='Journal of Science',
    volume='10',
    pages='1-10'
)
citation2 = manager.get_in_text_citation([two_author_ref])
# Returns: "(Smith & Jones, 2023)"

# Three or more authors (uses "et al.")
three_author_ref = manager.add_reference('journal',
    author='Virtanen, P.; Gommers, R.; Oliphant, T. E.',
    year='2020',
    title='SciPy 1.0',
    journal='Nature Methods',
    volume='17',
    pages='261-272'
)
citation3 = manager.get_in_text_citation([three_author_ref])
# Returns: "(Virtanen et al., 2020)"

# With page numbers
citation4 = manager.get_in_text_citation([two_author_ref], page='45')
# Returns: "(Smith & Jones, 2023, p. 45)"

# Narrative format
citation5 = manager.get_in_text_citation([github_ref], narrative=True)
# Returns: "GitHub (2024)"
```

#### Name Parsing Examples (NEW in v2.0.0):

```python
from research_toolkit import APA7ReferenceManager

manager = APA7ReferenceManager()

# All these formats work automatically:
key1 = manager.add_reference('journal',
    author='Smith, J. M.',  # Last, Initials
    year='2023',
    title='Research methods',
    journal='Journal of Science',
    volume='10',
    pages='1-10'
)

key2 = manager.add_reference('journal',
    author='John Michael Smith',  # First Middle Last
    year='2023',
    title='Another study',
    journal='Journal of Science',
    volume='10',
    pages='11-20'
)

key3 = manager.add_reference('journal',
    author='Smith, John Michael',  # Last, First Middle
    year='2023',
    title='Third study',
    journal='Journal of Science',
    volume='10',
    pages='21-30'
)

# Multiple authors with any separator:
key4 = manager.add_reference('journal',
    author='Smith, J.; Jones, K. & Brown, M.',  # Mixed separators
    year='2023',
    title='Collaborative study',
    journal='Journal of Science',
    volume='10',
    pages='31-40'
)

# Get formatted references (all properly formatted):
print(manager.format_reference(key1))
# Smith, J. M. (2023). Research methods. *Journal of Science*, *10*, 1-10.

# Get citations:
print(manager.get_in_text_citation([key1]))
# (Smith, 2023)

print(manager.get_in_text_citation([key4]))
# (Smith et al., 2023)

print(manager.get_in_text_citation([key1], narrative=True))
# Smith (2023)

print(manager.get_in_text_citation([key1], page='45'))
# (Smith, 2023, p. 45)
```

#### Advanced Name Parsing:

```python
# Parse a single author name
parsed = APA7ReferenceManager.parse_author_name("Smith, John Michael")
print(parsed)
# {
#   'last': 'Smith',
#   'first': 'John',
#   'middle': 'Michael',
#   'initials': 'J. M.',
#   'formatted_citation': 'Smith',
#   'formatted_reference': 'Smith, J. M.'
# }

# Parse multiple authors
authors = APA7ReferenceManager.parse_multiple_authors(
    "Smith, J.; Jones, Karen & Brown, M."
)
print(len(authors))  # 3

# Format for citation
citation = APA7ReferenceManager.format_authors_for_citation(authors)
print(citation)  # "Smith et al." (for 3+ authors)

# Format for reference list
reference = APA7ReferenceManager.format_authors_for_reference(authors)
print(reference)  # "Smith, J., Jones, K., & Brown, M."
```

---

## Troubleshooting

### Unicode Errors

**Problem:** `UnicodeEncodeError: 'charmap' codec can't encode...`

**Solution:** Use `SafeOutput.safe_print()` instead of `print()`:

```python
# Before (causes errors)
print(f"α = 0.05")

# After (safe)
from research_toolkit import SafeOutput
SafeOutput.safe_print("α = 0.05")
```

### Detection Issues

**Problem:** Unicode symbols not displaying correctly

**Solution:** Force ASCII mode:

```python
from research_toolkit import ReportFormatter

# Force ASCII
formatter = ReportFormatter(use_unicode_console=False)
```

### File Encoding

**Problem:** Saved files show garbled text

**Solution:** Specify UTF-8 encoding explicitly:

```python
from research_toolkit import SafeOutput

with SafeOutput.safe_file_output('output.txt', 'w', 'utf-8') as f:
    f.write("Content with symbols: α β μ\n")
```

---

## Testing

Run the package info command:

```bash
research-toolkit-info
```

Expected output shows:
- Package version (2.0.0)
- Available features
- Quick start examples

---

## Further Documentation

- **Research Guidelines:** `guidelines/RESEARCH_TYPES_INDEX.md`
- **Quick Reference:** `guidelines/QUICK_REFERENCE.md`
- **Examples:** `examples/` directory
- **Project Overview:** `README.md`

---

## License

Apache License 2.0 - See LICENSE file for details.
