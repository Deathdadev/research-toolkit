# Research Toolkit

A comprehensive Python library for conducting research across multiple methodologies with proper scientific notation, encoding-safe output, APA 7 referencing, and AI integration.

**Version:** 2.0.0

---

## Overview

The Research Toolkit provides everything needed to conduct rigorous research across empirical and non-empirical methodologies:

- **Empirical Research**: Correlational, comparative, time series, observational, meta-analysis
- **Non-Empirical Research**: Simulation, methodological, theoretical
- **APA 7 References**: Complete bibliography management system
- **Statistical Formatting**: APA-compliant result reporting
- **MCP Server**: AI model integration for research guidance
- **Encoding Safety**: Unicode/ASCII handling for cross-platform compatibility

---

## Installation

```bash
uv pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from research_toolkit import ReportFormatter, APA7ReferenceManager, StatisticalFormatter

# Create professional reports
formatter = ReportFormatter()
formatter.print_section("RESEARCH RESULTS")

# Format statistics (APA 7 style)
print(StatisticalFormatter.format_correlation(0.456, 0.012, 100))
# Output: r(98) = .46, p = .012

print(StatisticalFormatter.format_t_test(2.34, 48, 0.023))
# Output: t(48) = 2.34, p = .023

# Manage references
manager = APA7ReferenceManager()
key = manager.add_reference(
    'journal',
    author='Smith, J.; Jones, K.',
    year='2023',
    title='Research methods in psychology',
    journal='Journal of Psychology',
    volume='10',
    pages='123-145',
    doi='10.1234/jp.2023.123'
)

# Format reference (APA 7)
print(manager.format_reference(key))
# Smith, J., & Jones, K. (2023). Research methods in psychology. 
# *Journal of Psychology*, *10*, 123-145. https://doi.org/10.1234/jp.2023.123

# Generate citations
print(manager.get_in_text_citation([key]))
# (Smith & Jones, 2023)
```

### For AI Models (MCP Server)

```python
from research_toolkit import EmpiricalResearchMCPServer

server = EmpiricalResearchMCPServer()

# Select appropriate research type
result = server.call_tool("select_research_type", {
    "research_question": "Is there a relationship between X and Y?",
    "data_availability": "real_data_available",
    "goal": "correlate"
})

# Validate data sources
result = server.call_tool("check_data_requirements", {
    "research_type": "correlational",
    "proposed_data_source": "survey data"
})

# Format APA 7 references
result = server.call_tool("format_apa7_reference", {
    "reference_type": "journal",
    "author": "Smith, J.",
    "year": "2023",
    "title": "Research methods",
    "additional_fields": {"journal": "Journal", "volume": "10", "pages": "1-10"}
})
```

---

## Key Features

### 1. Core Utilities

- **SafeOutput**: Encoding-safe console/file output (Unicode → ASCII fallback)
- **ScientificNotation**: Greek symbols (α, β, μ) with ASCII fallback
- **ReportFormatter**: Professional report generation
- **StatisticalFormatter**: APA 7 statistical result formatting

### 2. Statistical Formatting

Complete APA 7 compliant formatters:
- `format_p_value()` - p-values (omits leading zero)
- `format_ci()` - Confidence intervals
- `format_mean_sd()` - Descriptive statistics
- `format_correlation()` - Correlation results with df
- `format_t_test()` - t-test results
- `format_f_test()` - F-test results
- `format_chi_square()` - Chi-square results
- `format_effect_size()` - Effect sizes with interpretation

### 3. APA 7 Reference Management

- **10 reference types**: journal, book, chapter, website, report, conference, dataset, software, dissertation, government
- **Advanced name parsing** (NEW in v2.0.0): Handles multiple author name formats
- **Multiple citations**: Cite multiple sources together
- **20+ authors**: Proper handling (first 19 ... last)
- **BibTeX export**: Convert to BibTeX format
- **Validation**: Check completeness and format

#### Name Parsing Examples (v2.0.0)

```python
# Parse author names in any format
parsed = APA7ReferenceManager.parse_author_name("Smith, John Michael")
# Returns: {'last': 'Smith', 'first': 'John', 'middle': 'Michael', 
#           'initials': 'J. M.', 'formatted_citation': 'Smith',
#           'formatted_reference': 'Smith, J. M.'}

# Supported formats:
manager.add_reference('journal',
    author='Smith, J. M.',          # ✅ Last, Initials
    author='Smith, John Michael',   # ✅ Last, First Middle
    author='John Michael Smith',    # ✅ First Middle Last
    author='Smith',                 # ✅ Last only
    author='J. M. Smith',          # ✅ Initials Last
    # All formats work and auto-convert to proper APA 7!
    ...
)
```

### 4. MCP Server (AI Integration)

7 research tools for AI models:
1. `select_research_type` - Choose methodology
2. `format_apa7_reference` - Format references
3. `generate_citation` - Create in-text citations
4. `validate_reference` - Check APA 7 compliance
5. `check_data_requirements` - Validate data sources
6. `get_methodology_guidance` - Research guidance
7. `validate_interpretation` - Check claim appropriateness

### 5. Research Examples

9 complete working examples:
- `00_synthetic_example` - What NOT to do
- `01_correlational_study` - Air quality research
- `02_comparative_study` - Group comparisons
- `03_time_series_analysis` - Temporal trends
- `04_observational_study` - GitHub repositories
- `05_meta_analysis` - Study synthesis
- `06_simulation_study` - SIR epidemic model
- `07_methodological_study` - Power analysis
- `08_theoretical_model` - Information diffusion theory

---

## Project Structure

```
research-toolkit/
├── src/
│   └── research_toolkit/       # Main package
│       ├── core/               # Core utilities
│       ├── references/         # APA 7 system
│       └── research/           # MCP server
├── examples/                   # 9 research examples
├── guidelines/                 # Research methodology guides
├── docs/                       # Documentation
│   ├── LIBRARY_GUIDE.md        # Complete library guide
│   ├── MCP_INTEGRATION_GUIDE.md # AI integration
│   ├── AI_MODEL_PROMPTS.md     # AI model prompts (NEW!)
│   └── FUTURE_ROADMAP.md       # Future plans
├── templates/                  # Reusable templates
├── README.md                   # This file
├── setup.py                    # Installation script
└── requirements.txt            # Dependencies
```

---

## Research Types Covered

### Empirical (Real Data Required)
1. **Correlational Study** - Examine relationships between variables
2. **Comparative Study** - Compare groups
3. **Time Series Analysis** - Analyze temporal patterns
4. **Observational Study** - Describe phenomena
5. **Meta-Analysis** - Synthesize existing research

### Non-Empirical (Conditional/Theoretical)
6. **Simulation Study** - Explore models (IF-THEN claims)
7. **Methodological Study** - Test research methods
8. **Theoretical Model** - Develop theory (requires empirical testing)

---

## Documentation

- **Library Guide**: `docs/LIBRARY_GUIDE.md` - Complete API reference
- **MCP Integration**: `docs/MCP_INTEGRATION_GUIDE.md` - AI model integration with educational prompts
- **Research Guidelines**: `guidelines/` - Methodology guides for all research types
- **Examples**: `examples/` - 9 working examples with full code

### AI Model Prompts (NEW!)

The MCP server now includes **4 educational prompts** that teach AI models proper research methodology:

1. **research_methodology_primer** - Teaches core principles: verifiable, reproducible, falsifiable, transparent
2. **apa_formatting_guide** - Teaches APA 7 formatting using MCP tools
3. **claim_validation_guide** - Teaches matching claims to evidence by research design
4. **data_source_guide** - Teaches data requirements (when real data is needed vs synthetic)

**Educational Approach:** Instead of step-by-step instructions, these prompts teach principles and point AI models to:
- `guidelines/AI_RESEARCH_GUIDELINES.md` - Complete methodology guide
- `templates/research_template.py` - Structured workflow template
- `examples/` directory - Working examples for each research type

Each prompt explains decision-making, shows how to use MCP validation tools, and encourages studying examples. See `docs/MCP_INTEGRATION_GUIDE.md` for details.

---

## Requirements

- Python 3.8+ (I have pinned the python version to 3.10)
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- requests
- python-dotenv
- scikit-learn
- statsmodels (optional, for time series)

---

## Console Command

After installation:

```bash
research-toolkit-info
```

Shows package information, features, and quick start guide.

---

## Contributing

This toolkit is designed for:
- Researchers conducting studies
- Students learning research methods
- AI models performing research
- Educators teaching methodology

---

## License

MIT License

---

## Version History

### v2.0.0 (Current)
- Renamed to "Research Toolkit" (broader scope)
- Enhanced statistical formatters (9 methods)
- Enhanced APA 7 manager (10 reference types, multiple citations)
- Professional package structure (src/ layout)
- Clean project organization
- Updated documentation

### v1.0.0
- Initial release as "Empirical Research Toolkit"
- Basic utilities and APA 7 support
- 9 research examples
- MCP server foundation

---

## Contact

For questions, issues, or contributions, please refer to the documentation in the `docs/` directory.
