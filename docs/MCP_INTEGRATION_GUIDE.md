# MCP Server Integration Guide

## Overview

The **Research Toolkit MCP Server** exposes research methodology tools and APA 7 formatting to AI models through the Model Context Protocol (MCP). This enables AI models to conduct proper research directly within conversational workflows.

**Version:** 2.0.0  
**MCP Protocol:** https://modelcontextprotocol.io/

---

## What is MCP?

The Model Context Protocol (MCP) is a standard for connecting AI models to external tools and resources. It allows:

- **Tools**: Callable functions that AI models can use
- **Resources**: Documents and data that AI models can access
- **Prompts**: Reusable prompt templates
- **Sampling**: AI model invocation

**For AI Models:** MCP provides a standardized way to access external capabilities  
**For Developers:** MCP provides a standard interface for exposing functionality

---

## Server Architecture

### Components

```
Research Toolkit MCP Server
│
├── Tools (7 available)
│   ├── select_research_type       # Determine appropriate methodology
│   ├── format_apa7_reference      # Format bibliographic references
│   ├── generate_citation          # Create in-text citations
│   ├── validate_reference         # Check APA 7 compliance
│   ├── check_data_requirements    # Validate data sources
│   ├── get_methodology_guidance   # Step-by-step research guidance
│   └── validate_interpretation    # Check claims for appropriateness
│
├── Prompts (4 available - Educational Guides)
│   ├── research_methodology_primer      # Teaches research principles
│   ├── apa_formatting_guide             # Teaches APA 7 formatting
│   ├── claim_validation_guide           # Teaches claim validation
│   └── data_source_guide                # Teaches data requirements
│
├── Resources (3 available)
│   ├── guidelines/all             # Complete methodology guidelines
│   ├── examples/all               # 9 working research examples
│   └── templates/research         # Reusable research template
│
└── APA 7 Manager
    └── Complete reference management system
```

---

## Available Prompts (Educational Guides)

The MCP server provides **4 educational prompts** that teach AI models proper research methodology by pointing them to the comprehensive guidelines, templates, and examples in the repository.

**Philosophy:** Rather than providing step-by-step instructions, these prompts educate AI models about research principles and direct them to detailed resources for learning.

### 1. research_methodology_primer

**Purpose:** Core primer on conducting proper research

**What it teaches:**
- The four principles: Verifiable, Reproducible, Falsifiable, Transparent
- Critical question: "Am I making claims about the real world?"
- When real data is required vs when synthetic is acceptable
- How to use the MCP server tools
- Research workflow

**Points to:**
- `guidelines/AI_RESEARCH_GUIDELINES.md` - Complete methodology guide
- `templates/research_template.py` - Structured template
- `examples/` directory - Working examples for each research type

**Usage:**
```python
prompt = server.get_prompt("research_methodology_primer")
# Returns educational content teaching research principles
```

---

### 2. apa_formatting_guide

**Purpose:** Guide to APA 7 formatting using MCP tools

**What it teaches:**
- Core principle: Every claim needs a source
- How to use format_apa7_reference, generate_citation, validate_reference
- Parenthetical vs narrative citations
- When to include page numbers

**Points to:**
- `examples/01_correlational_study.py` - See reference management
- `examples/02_comparative_study.py` - See in-text citations

**Usage:**
```python
prompt = server.get_prompt("apa_formatting_guide")
# Returns guide on using MCP tools for citations
```

---

### 3. claim_validation_guide

**Purpose:** Guide to validating research claims match evidence

**What it teaches:**
- Design limitations by research type
- What each design can and cannot claim
- Correlational: Association only, never causation
- Experimental: Only design supporting causal claims
- How to use validate_interpretation tool

**Points to:**
- `examples/01_correlational_study.py` - Proper correlational language
- `examples/02_comparative_study.py` - Acknowledging confounds
- `examples/06_simulation_study.py` - Conditional claims

**Usage:**
```python
prompt = server.get_prompt("claim_validation_guide")
# Returns guide on matching claims to evidence
```

---

### 4. data_source_guide

**Purpose:** Guide to acceptable data sources by research type

**What it teaches:**
- Decision tree: Real world claims need real data
- When synthetic data is acceptable (methods/theory)
- How to use check_data_requirements tool
- Documentation requirements

**Points to:**
- `examples/01_correlational_study.py` - Real API data usage
- `examples/07_methodological_study.py` - Synthetic for methods
- `examples/00_synthetic_example_what_not_to_do.py` - Common mistakes

**Usage:**
```python
prompt = server.get_prompt("data_source_guide")
# Returns guide on data requirements
```

---

## Key Difference from Procedural Prompts

**These prompts are NOT step-by-step instructions.** Instead, they:
- Teach principles and decision-making
- Point to comprehensive resources
- Show how to use validation tools
- Encourage reading guidelines and studying examples

The AI model should:
1. Read the educational prompt
2. Study the referenced guidelines
3. Examine relevant examples
4. Use MCP tools for validation throughout

This approach teaches research methodology rather than just providing a checklist.

---

## Available Tools

### 1. select_research_type

**Purpose:** Determine the appropriate research type based on question and data availability

**Input:**
```json
{
  "research_question": "Is there a relationship between X and Y?",
  "data_availability": "real_data_available",
  "goal": "correlate"
}
```

**Output:**
```json
{
  "status": "success",
  "research_type": "correlational",
  "data_requirement": "Real data showing relationship between variables",
  "methodology_summary": "Measure two variables, calculate correlation...",
  "key_limitations": ["Cannot establish causation", "..."],
  "next_steps": ["Review guidelines", "Verify data", "..."]
}
```

**Use Case:** AI model needs to determine which research methodology to use

---

### 2. format_apa7_reference

**Purpose:** Format bibliographic reference in APA 7 style

**Input:**
```json
{
  "reference_type": "journal",
  "author": "Smith, J. M.",
  "year": "2023",
  "title": "Research methods",
  "additional_fields": {
    "journal": "Journal of Science",
    "volume": "10",
    "pages": "123-145",
    "doi": "10.1234/js.2023.123"
  }
}
```

**Output:**
```json
{
  "status": "success",
  "formatted_reference": "Smith, J. M. (2023). Research methods. *Journal of Science*, *10*, 123-145. https://doi.org/10.1234/js.2023.123",
  "citation_key": "ref1",
  "is_valid": true,
  "validation_issues": []
}
```

**Use Case:** AI model needs to cite sources in APA 7 format

---

### 3. generate_citation

**Purpose:** Generate in-text citation

**Input:**
```json
{
  "author": "Smith, J. M.; Jones, K.",
  "year": "2023",
  "page": "45",
  "narrative": false
}
```

**Output:**
```json
{
  "status": "success",
  "citation": "(Smith & Jones, 2023, p. 45)",
  "citation_type": "parenthetical"
}
```

**Use Case:** AI model needs inline citation within text

---

### 4. validate_reference

**Purpose:** Check reference for APA 7 compliance

**Input:**
```json
{
  "reference_type": "journal",
  "fields": {
    "author": "Smith, J.",
    "year": "2023",
    "title": "Title",
    "journal": "Journal",
    "volume": "10",
    "pages": "1-10"
  }
}
```

**Output:**
```json
{
  "status": "success",
  "is_valid": true,
  "issues": [],
  "field_validation": {
    "author": {"valid": true, "message": "Valid"},
    "year": {"valid": true, "message": "Valid year"},
    "doi": {"valid": true, "message": "No DOI provided (optional)"}
  }
}
```

**Use Case:** AI model wants to verify reference correctness before using

---

### 5. check_data_requirements

**Purpose:** Validate whether data source is appropriate for research type

**Input:**
```json
{
  "research_type": "correlational",
  "proposed_data_source": "randomly generated synthetic data"
}
```

**Output:**
```json
{
  "status": "success",
  "research_type": "correlational",
  "requires_real_data": true,
  "minimum_data": "Paired observations of at least 2 variables",
  "synthetic_acceptable": false,
  "reason": "Empirical research requires verifiable real-world data",
  "assessment": "INVALID",
  "assessment_message": "Synthetic data is NOT acceptable for correlational research..."
}
```

**Use Case:** AI model wants to verify data source before starting research

---

### 6. get_methodology_guidance

**Purpose:** Get step-by-step methodology for research type

**Input:**
```json
{
  "research_type": "comparative"
}
```

**Output:**
```json
{
  "status": "success",
  "research_type": "comparative",
  "methodology_summary": "Compare groups using appropriate test...",
  "key_steps": [...],
  "common_pitfalls": [...],
  "statistical_tests": [...]
}
```

**Use Case:** AI model needs guidance on conducting specific research type

---

### 7. validate_interpretation

**Purpose:** Check if proposed claim is appropriate for research design

**Input:**
```json
{
  "research_type": "correlational",
  "proposed_claim": "Study hours cause higher test scores",
  "has_control_group": false,
  "has_random_assignment": false
}
```

**Output:**
```json
{
  "status": "success",
  "is_appropriate": false,
  "has_causal_language": true,
  "can_claim_causation": false,
  "issues": ["Causal language detected but study design does not support causal claims"],
  "warnings": [],
  "suggested_revision": "Study hours are associated with higher test scores"
}
```

**Use Case:** AI model wants to verify interpretation before making claims

---

## Integration with AI Models

### Using Prompts for Guided Workflows

The MCP server provides **4 educational prompts** that teach AI models proper research methodology rather than providing step-by-step instructions.

**Educational Philosophy:**

These prompts teach principles and point to comprehensive resources:
- `guidelines/AI_RESEARCH_GUIDELINES.md` - Complete methodology guide
- `templates/research_template.py` - Structured workflow template
- `examples/` directory - Working examples for each research type

**Quick Start:**

```python
from research_toolkit.research.mcp_server import ResearchToolkitMCPServer

server = ResearchToolkitMCPServer()

# List available prompts
prompts = server.list_prompts()
for prompt in prompts:
    print(f"{prompt['name']}: {prompt['description']}")

# Get an educational prompt
prompt_result = server.get_prompt("research_methodology_primer")
# Returns educational content teaching research principles
```

**Available Prompts:**

1. **research_methodology_primer** - Teaches core research principles
2. **apa_formatting_guide** - Teaches APA 7 formatting with MCP tools
3. **claim_validation_guide** - Teaches matching claims to evidence
4. **data_source_guide** - Teaches data requirements by research type

Each prompt:
- Explains principles and decision-making
- Points to detailed guidelines and examples
- Shows how to use MCP validation tools
- Encourages studying examples for learning

**Example: Using a Prompt**

```python
# Step 1: Get the educational prompt
prompt_result = server.get_prompt("data_source_guide")

# Step 2: Read and understand the principles taught
# The prompt teaches:
# - Decision tree for data requirements
# - When real data is required
# - When synthetic is acceptable
# - How to use check_data_requirements tool

# Step 3: Follow the guidance to study resources
# - Read guidelines/AI_RESEARCH_GUIDELINES.md
# - Study examples/01_correlational_study.py (real data)
# - Study examples/07_methodological_study.py (synthetic OK)

# Step 4: Use MCP tools for validation
data_check = server.call_tool("check_data_requirements", {
    "research_type": "correlational",
    "proposed_data_source": "OpenWeatherMap API data"
})
# Returns: "ACCEPTABLE" - real data source
```

---

### Workflow Example

```
User: "Is there a relationship between study hours and test scores?"

AI Model:
1. Calls select_research_type
   - research_question: "relationship between study hours and test scores"
   - data_availability: "real_data_obtainable"
   - goal: "correlate"
   
2. Receives: "correlational study with real data required"

3. Calls check_data_requirements
   - research_type: "correlational"
   - proposed_data_source: "student survey data"
   
4. Receives: "ACCEPTABLE - real data source"

5. Proceeds with correlational study methodology

6. After analysis, calls validate_interpretation
   - proposed_claim: "Study hours lead to better scores"
   
7. Receives: "INAPPROPRIATE - remove causal language"

8. Revises to: "Study hours are positively correlated with scores"

9. Calls format_apa7_reference for each source

10. Generates properly formatted research response
```

---

## Server Setup

### Installation

```bash
pip install -e .
```

### Running the Server

```python
from mcp_server import EmpiricalResearchMCPServer

# Initialize server
server = EmpiricalResearchMCPServer()

# List available tools
tools = server.list_tools()

# Call a tool
result = server.call_tool("select_research_type", {
    "research_question": "...",
    "data_availability": "real_data_available",
    "goal": "correlate"
})
```

### MCP Configuration

For integration with MCP-compatible AI systems, create `mcp_config.json`:

```json
{
  "name": "research-toolkit",
  "version": "1.0.0",
  "description": "Research methodology and APA 7 formatting tools",
  "server": {
    "command": "python",
    "args": ["mcp_server.py"]
  },
  "tools": [
    {
      "name": "select_research_type",
      "description": "Determine appropriate research methodology"
    },
    {
      "name": "format_apa7_reference",
      "description": "Format references in APA 7 style"
    },
    ...
  ]
}
```

---

## Benefits for AI Models

### 1. Research Methodology Guidance

**Without MCP:**
- AI model might suggest inappropriate research design
- May not check data requirements
- Could make invalid claims

**With MCP:**
- AI model selects appropriate methodology
- Validates data sources automatically
- Checks interpretations before presenting

### 2. Proper Citation Management

**Without MCP:**
- Inconsistent citation formats
- Manual APA 7 formatting prone to errors
- No validation of reference completeness

**With MCP:**
- Consistent APA 7 formatting
- Automatic validation
- Complete reference management

### 3. Quality Control

**Without MCP:**
- AI model responsible for all validation
- Prone to overclaiming
- May use synthetic data inappropriately

**With MCP:**
- Automatic validation at each step
- Prevents inappropriate claims
- Enforces research quality standards

---

## Use Cases

### 1. Academic Research Assistant

```
Student: "Help me design a study on sleep and academic performance"

AI with MCP:
1. select_research_type → Determines correlational study
2. check_data_requirements → Requires real data
3. Guides student to collect actual sleep/grade data
4. Helps analyze with proper methods
5. validate_interpretation → Ensures no causal claims
6. format_apa7_reference → Formats all sources
7. Produces valid research report
```

### 2. Literature Review

```
Researcher: "Summarize research on X and provide references"

AI with MCP:
1. Searches literature
2. format_apa7_reference → Formats each source consistently
3. generate_citation → Creates in-text citations
4. Produces properly formatted review with reference list
```

### 3. Research Design Consultation

```
User: "I want to prove that meditation reduces stress"

AI with MCP:
1. select_research_type → Notes need for experimental design
2. check_data_requirements → Real intervention data required
3. Advises on control group, random assignment
4. validate_interpretation → Checks if "prove" is appropriate
5. Guides toward valid causal study design
```

---

## API Reference

### Tool Call Format

```python
result = server.call_tool(tool_name, arguments)
```

**Returns:**
```python
{
    "status": "success" | "error",
    "message": "...",  # If error
    # ... tool-specific fields
}
```

### Error Handling

All tools return structured error messages:

```python
{
    "status": "error",
    "message": "Missing required field: author"
}
```

### Validation Results

Tools that validate return consistent structure:

```python
{
    "status": "success",
    "is_valid": true | false,
    "issues": ["issue 1", "issue 2"],
    "warnings": ["warning 1"]
}
```

---

## Advanced Features

### Batch Operations

```python
# Format multiple references
references = [
    {"type": "journal", "author": "...", ...},
    {"type": "book", "author": "...", ...}
]

for ref in references:
    result = server.call_tool("format_apa7_reference", ref)
    print(result['formatted_reference'])
```

### Research Pipeline

```python
# Complete research workflow
def conduct_research(question, data_source):
    # 1. Select type
    type_result = server.call_tool("select_research_type", {
        "research_question": question,
        "data_availability": "real_data_available",
        "goal": "correlate"
    })
    
    # 2. Check data
    data_result = server.call_tool("check_data_requirements", {
        "research_type": type_result['research_type'],
        "proposed_data_source": data_source
    })
    
    if data_result['assessment'] != "ACCEPTABLE":
        return "Invalid data source"
    
    # 3. Conduct analysis (user code)
    # ...
    
    # 4. Validate interpretation
    interp_result = server.call_tool("validate_interpretation", {
        "research_type": type_result['research_type'],
        "proposed_claim": "...",
        ...
    })
    
    return interp_result
```

---

## Testing

### Unit Tests

```python
def test_select_research_type():
    server = EmpiricalResearchMCPServer()
    result = server.call_tool("select_research_type", {
        "research_question": "test",
        "data_availability": "real_data_available",
        "goal": "correlate"
    })
    assert result['status'] == 'success'
    assert result['research_type'] == 'correlational'
```

### Integration Tests

```python
def test_full_workflow():
    server = EmpiricalResearchMCPServer()
    
    # Select type
    result1 = server.call_tool("select_research_type", {...})
    assert result1['status'] == 'success'
    
    # Check data
    result2 = server.call_tool("check_data_requirements", {
        "research_type": result1['research_type'],
        ...
    })
    assert result2['status'] == 'success'
    
    # Format reference
    result3 = server.call_tool("format_apa7_reference", {...})
    assert result3['is_valid'] == True
```

---

## Troubleshooting

### Common Issues

**Issue:** Tool returns "Unknown tool" error  
**Solution:** Check tool name spelling, use `server.list_tools()` to see available tools

**Issue:** Missing required argument  
**Solution:** Check tool's inputSchema, ensure all required fields provided

**Issue:** Reference validation fails  
**Solution:** Use `validate_reference` tool to see specific issues

**Issue:** MCP connection fails  
**Solution:** Verify server is running, check MCP configuration file

---

## Future Enhancements

See `FUTURE_ROADMAP.md` for planned features:

- Statistical analysis tools
- Data visualization tools
- Automated literature search
- Peer review simulation
- Export to multiple formats (LaTeX, Word, etc.)
- Collaboration features
- Version control integration

---

## Support and Documentation

- **Full API:** See `mcp_server.py` docstrings
- **Examples:** See `examples/` directory
- **Guidelines:** See `guidelines/` directory
- **Library Guide:** See `LIBRARY_GUIDE.md`

---

## License

Apache License 2.0 - See LICENSE file for details
