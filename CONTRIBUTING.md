# Contributing to Research Toolkit

Thank you for your interest in contributing to the Research Toolkit! This project is designed to advance rigorous, reproducible research methodologies across multiple disciplines. Your contributions help improve research quality for researchers, students, educators, and AI models worldwide.

## Why Contribute to the Main Repository?

### The Power of Collaborative Research

**Contributing upstream creates shared value for the entire research community.** Here's why contributing to the main repository benefits everyone:

#### 1. **Unified Standards**
- **Consistency**: All researchers benefit from consistent APA 7 formatting, statistical reporting, and research methodologies
- **Quality Assurance**: Community review ensures robust, scientifically sound implementations
- **Best Practices**: Shared standards evolve and improve through collective expertise

#### 2. **Amplified Impact**
- **Broader Reach**: Your improvements immediately benefit thousands of researchers using the toolkit
- **Cross-Pollination**: Research methods improvements help diverse fields (psychology, sociology, environmental science, etc.)
- **Educational Value**: Students and AI models learn from the most current, peer-reviewed methods

#### 3. **Sustained Development**
- **Longevity**: Community contributions ensure the project remains actively maintained
- **Innovation**: Diverse perspectives lead to more creative solutions and methodologies
- **Reliability**: Multiple contributors catch edge cases and ensure robust implementations

### Forking vs. Contributing Upstream

| Aspect | Forking | Contributing Upstream |
|--------|---------|----------------------|
| **Impact** | Limited to your use case | Benefits entire research community |
| **Maintenance** | You maintain your fork | Community maintains improvements |
| **Standards** | May diverge over time | Evolves with community consensus |
| **Collaboration** | Isolated development | Builds on others' expertise |
| **Learning** | Self-directed | Community knowledge sharing |

**We strongly encourage upstream contributions** because this toolkit serves as **research infrastructure** - improvements should benefit all users, not just individual implementations.

---

## How to Contribute

### 1. Getting Started

1. **Read the Documentation**
   - [Library Guide](docs/LIBRARY_GUIDE.md) - Complete API reference
   - [Research Guidelines](guidelines/AI_RESEARCH_GUIDELINES.md) - Methodology standards
   - [MCP Integration Guide](docs/MCP_INTEGRATION_GUIDE.md) - AI model integration

2. **Explore Examples**
   - Study the 9 research examples in `examples/`
   - Understand different research methodologies (correlational, comparative, time series, etc.)
   - Learn proper data handling and statistical reporting

3. **Set Up Development Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/Deathdadev/research-toolkit.git
   cd research-toolkit

   # Install in development mode using uv (recommended)
   uv pip install -e .

   # Or using pip
   pip install -e .

   # Install with example dependencies
   uv pip install -e ".[examples]"
   ```

   **Requirements:**
   - Python >= 3.10
   - Dependencies: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, statsmodels, requests, python-dotenv

### 2. Contribution Workflow

1. **Choose an Issue**
   - Browse existing [Issues](../../issues)
   - Look for "good first issue" or "help wanted" labels
   - Create a new issue if you have a unique idea

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Make Your Changes**
   - Follow code standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Test specific functionality
   python examples/your-example.py

   # Run the package info command
   research-toolkit-info

   # Check code style with ruff
   ruff check src/research_toolkit/

   # Auto-fix issues
   ruff check src/research_toolkit/ --fix

   # Check for functional errors only
   ruff check src/research_toolkit/ --select F,E9
   ```

5. **Submit a Pull Request**
   - Ensure your branch is up to date with main
   - Provide clear description of changes
   - Reference related issues
   - Include screenshots for UI changes (if applicable)

---

## Code Standards

### Python Style Guidelines

We follow [PEP 8](https://peps.python.org/pep-0008/) with the following specifications:

#### 1. **Type Hints**
All functions must include type hints:
```python
from typing import Dict, List, Optional, Tuple

def analyze_correlation(data: pd.DataFrame, variables: List[str]) -> Dict[str, float]:
    """Analyze correlations between specified variables."""
    # Implementation
    pass
```

#### 2. **Docstrings**
Use Google-style docstrings for all public functions:
```python
def format_statistical_result(
    statistic: str,
    value: float,
    p_value: float,
    df: int,
    decimals: int = 3
) -> str:
    """
    Format statistical results according to APA 7 guidelines.

    This function ensures consistent, publication-ready statistical reporting
    across all research outputs.

    Args:
        statistic: Name of the statistic (e.g., 'r', 't', 'F')
        value: The statistical value
        p_value: The p-value for significance testing
        df: Degrees of freedom
        decimals: Number of decimal places for formatting

    Returns:
        Formatted string following APA 7 conventions

    Example:
        >>> format_statistical_result('r', 0.456, 0.012, 98)
        'r(98) = .46, p = .012'
    """
    pass
```

#### 3. **Imports**
Organize imports following PEP 8 standards:
```python
# Standard library imports
import os
from typing import Dict, List

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats

# Local imports
from .core import SafeOutput
from .formatting import ScientificNotation
```

#### 4. **Error Handling**
Use appropriate exceptions with informative messages:
```python
def validate_data_source(data_source: str) -> bool:
    """Validate that data source meets empirical research requirements."""
    if not data_source:
        raise ValueError("Data source cannot be empty")

    if data_source == "synthetic":
        raise ValueError(
            "Synthetic data cannot be used for empirical claims. "
            "See guidelines/AI_RESEARCH_GUIDELINES.md for acceptable data sources."
        )

    # Implementation
    return True
```

#### 5. **Scientific Accuracy**
- **Statistical Correctness**: Ensure all statistical implementations are mathematically sound
- **APA 7 Compliance**: Follow current APA 7 formatting guidelines exactly
- **Reproducibility**: All functions must produce identical results given identical inputs
- **Documentation**: Include clear examples and edge cases

### 6. **File Organization**
```
research-toolkit/
├── src/research_toolkit/
│   ├── core/           # Core utilities (formatting, output, statistics)
│   ├── references/     # APA 7 reference management
│   ├── research/       # MCP server and research tools
│   └── __init__.py     # Package initialization (v2.0.0)
├── examples/           # 9 research methodology examples
├── guidelines/         # Research methodology guidelines
├── docs/              # Library and integration guides
├── templates/         # Research template for AI agents
├── pyproject.toml     # Package configuration
└── LICENSE            # Apache License 2.0
```

---

## Testing Requirements

### Test Categories

#### 1. **Unit Tests**
- Test individual functions and methods
- Mock external dependencies
- Cover edge cases and error conditions

```python
import pytest
from research_toolkit.core.formatting import StatisticalFormatter

def test_format_correlation():
    """Test correlation formatting with various inputs."""
    # Test normal case
    result = StatisticalFormatter.format_correlation(0.456, 0.012, 100)
    assert result == "r(98) = .46, p = .012"

    # Test perfect correlation
    result = StatisticalFormatter.format_correlation(1.0, 0.001, 50)
    assert result == "r(48) = 1.00, p = .001"

    # Test non-significant result
    result = StatisticalFormatter.format_correlation(0.123, 0.456, 25)
    assert result == "r(23) = .12, p = .456"
```

#### 2. **Integration Tests**
- Test complete workflows
- Verify end-to-end functionality
- Test with real data sources

```python
def test_research_workflow():
    """Test complete research workflow from data to publication."""
    # Test data collection
    # Test statistical analysis
    # Test APA formatting
    # Test reference management
    pass
```

#### 3. **Scientific Validation Tests**
- Verify statistical correctness
- Test against known results
- Validate APA 7 formatting accuracy

```python
def test_statistical_accuracy():
    """Verify statistical calculations against known values."""
    # Test correlation calculation
    # Test p-value computation
    # Test confidence intervals
    pass
```

### Test Data Requirements

#### 1. **Synthetic Test Data**
- Use for methodological testing
- Document data generation process
- Ensure reproducibility with random seeds

```python
def test_methodological_function():
    """Test functions that use synthetic data."""
    np.random.seed(42)  # Ensure reproducibility
    synthetic_data = np.random.normal(0, 1, 100)

    # Test your function
    result = your_function(synthetic_data)
    assert result is not None
```

#### 2. **Real Data for Validation**
- Use public datasets for integration testing
- Document data sources clearly
- Ensure proper attribution

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_formatting.py

# Run with coverage
python -m pytest --cov=src/research_toolkit

# Run only unit tests
python -m pytest -m unit

# Run integration tests
python -m pytest -m integration
```

---

## Documentation Guidelines

### Documentation Structure

#### 1. **Function Documentation**
Every public function must have:
- Complete docstring with description
- Args section with types and descriptions
- Returns section with type and description
- Example usage

#### 2. **Module Documentation**
Each module must have:
- Module-level docstring explaining purpose
- Brief overview of functionality
- Usage examples

#### 3. **Research Examples**
New research methodologies should include:
- Complete working example in `examples/`
- Explanation of research design decisions
- Data source documentation
- Interpretation guidelines

### Documentation Updates

When adding new features:
1. **Update Library Guide** (`docs/LIBRARY_GUIDE.md`)
2. **Add Examples** (if new research methodology)
3. **Update MCP Integration** (if affects AI tools)
4. **Update README** (for major features)

### Documentation Standards

#### 1. **Clarity**
- Use clear, concise language
- Define technical terms
- Provide context for complex concepts

#### 2. **Completeness**
- Document all parameters
- Include return values
- Explain exceptions and error conditions

#### 3. **Examples**
- Provide practical, copy-pasteable examples
- Show both simple and advanced usage
- Include expected outputs

#### 4. **Scientific Accuracy**
- Use correct statistical terminology
- Follow APA 7 formatting in examples
- Acknowledge limitations and assumptions

---

## Contribution Areas

### High-Impact Contributions

#### 1. **Research Methodologies**
- New statistical methods and formatters
- Improved research workflows
- Enhanced data validation
- Support for additional statistical tests (e.g., non-parametric tests)

#### 2. **APA 7 Formatting**
- New reference types (currently 10 supported)
- Updated formatting rules
- Improved author name parsing algorithms
- Enhanced citation formatting

#### 3. **MCP Server Tools**
- New AI research guidance tools
- Enhanced validation functions
- Better error messages and guidance
- Additional research type support

#### 4. **Core Utilities (v2.0 Focus)**
- Expanded scientific notation system (122 symbols in v2.0!)
- Better Unicode/ASCII fallback handling
- Generalized unit formatting system
- New convenience functions (concentrations, percentages, etc.)
- Performance improvements

### Documentation Contributions

#### 1. **Research Guidelines**
- Clarifications to existing guidelines
- New research methodology guides
- Common pitfalls and solutions

#### 2. **Examples**
- Additional research examples
- Alternative approaches to existing problems
- Edge case demonstrations

#### 3. **Educational Content**
- AI model prompts
- Student tutorials
- Educator resources

---

## Review Process

### Pull Request Review

1. **Automated Checks**
   - Code style (PEP 8 compliance)
   - Type hints validation
   - Test coverage requirements
   - Documentation completeness

2. **Manual Review**
   - Scientific accuracy verification
   - APA 7 compliance checking
   - Documentation quality assessment
   - Example validation

3. **Community Feedback**
   - Other contributors may provide input
   - Domain experts may review specialized areas
   - Users may test and provide feedback

### Review Criteria

- **Functionality**: Does it work as intended?
- **Scientific Soundness**: Are statistical methods correct?
- **Documentation**: Is it well-documented with examples?
- **Testing**: Are there adequate tests?
- **Standards Compliance**: Does it follow project conventions?

---

## Community Guidelines

### Respectful Collaboration

1. **Constructive Feedback**
   - Focus on code quality and functionality
   - Provide specific, actionable suggestions
   - Acknowledge good work

2. **Inclusive Communication**
   - Use welcoming, professional language
   - Be patient with newcomers
   - Respect diverse backgrounds and perspectives

3. **Scientific Integrity**
   - Maintain high standards for research quality
   - Be honest about limitations and uncertainties
   - Prioritize reproducible, verifiable results

### Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check existing docs before asking questions
- **Examples**: Study existing examples for guidance

---

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

Copyright 2025 Deathdadev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## Recognition

Contributors are recognized for their work through:
- **GitHub Contributors** list and graph
- **Release Notes** mentioning major contributions
- **Documentation Credits** for significant improvements
- **Community Thanks** in discussions and issues

---

## Final Thoughts

Your contributions to this project have real impact on research quality worldwide. By contributing upstream, you're helping to:

- **Improve Research Standards**: Better tools lead to better research
- **Educate Future Researchers**: Students learn from robust, well-documented methods
- **Support AI Research**: AI models get better guidance for conducting valid research
- **Advance Science**: Higher quality research accelerates scientific progress

Thank you for being part of this mission to improve research quality and accessibility for everyone!

---

*For questions or suggestions, please open an issue or start a discussion. We're here to help you contribute effectively!*