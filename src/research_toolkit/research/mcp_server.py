"""
Model Context Protocol (MCP) Server for Research Toolkit

This MCP server exposes research methodology tools and APA 7 formatting
capabilities to AI models, enabling them to conduct proper research
directly within their conversational workflows.

Supports all 8 research types and 10 APA 7 reference types with advanced
name parsing (v2.0.0 features).

MCP Specification: https://modelcontextprotocol.io/
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..references.apa7 import APA7ReferenceManager


class ResearchType(Enum):
    """Types of research supported by the toolkit."""
    CORRELATIONAL = "correlational"
    COMPARATIVE = "comparative"
    TIME_SERIES = "time_series"
    OBSERVATIONAL = "observational"
    META_ANALYSIS = "meta_analysis"
    SIMULATION = "simulation"
    METHODOLOGICAL = "methodological"
    THEORETICAL = "theoretical"


@dataclass
class Tool:
    """MCP Tool specification."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class MCPResource:
    """MCP Resource specification."""
    uri: str
    name: str
    description: str
    mimeType: str


class ResearchToolkitMCPServer:
    """
    MCP Server for Research Toolkit.
    
    Provides tools for:
    - Research type selection and validation (8 types)
    - APA 7 reference formatting (10 types with advanced name parsing)
    - Citation generation
    - Research methodology guidance
    - Data requirement checking
    - Interpretation validation
    """

    def __init__(self):
        """Initialize MCP server."""
        self.name = "research-toolkit"
        self.version = "2.0.0"
        self.apa_manager = APA7ReferenceManager()

        # Define tools
        self.tools = [
            self._define_research_type_selector(),
            self._define_apa7_formatter(),
            self._define_citation_generator(),
            self._define_reference_validator(),
            self._define_data_requirement_checker(),
            self._define_methodology_advisor(),
            self._define_interpretation_validator()
        ]

        # Define resources
        self.resources = [
            self._define_guidelines_resource(),
            self._define_examples_resource(),
            self._define_templates_resource()
        ]

    def _define_research_type_selector(self) -> Tool:
        """Tool: Select appropriate research type based on research question."""
        return Tool(
            name="select_research_type",
            description=(
                "Determines the appropriate research type based on research question, "
                "available data, and research goals. Returns guidance on methodology, "
                "data requirements, and potential limitations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "research_question": {
                        "type": "string",
                        "description": "The research question to be answered"
                    },
                    "data_availability": {
                        "type": "string",
                        "enum": ["real_data_available", "real_data_obtainable", "no_real_data", "model_only"],
                        "description": "Availability of real empirical data"
                    },
                    "goal": {
                        "type": "string",
                        "enum": ["describe", "compare", "correlate", "predict", "explain", "theorize"],
                        "description": "Primary research goal"
                    }
                },
                "required": ["research_question", "data_availability", "goal"]
            }
        )

    def _define_apa7_formatter(self) -> Tool:
        """Tool: Format reference in APA 7 style."""
        return Tool(
            name="format_apa7_reference",
            description=(
                "Formats a bibliographic reference according to APA 7th edition guidelines. "
                "Supports 10 reference types (v2.0.0) with advanced name parsing that handles "
                "5+ author name formats automatically: 'Last, F. M.', 'First Last', 'Last, First Middle', etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "reference_type": {
                        "type": "string",
                        "enum": ["journal", "book", "chapter", "website", "report",
                                "conference", "dataset", "software", "dissertation", "government"],
                        "description": "Type of reference (10 types supported in v2.0.0)"
                    },
                    "author": {
                        "type": "string",
                        "description": "Author(s) in any format: 'Last, F. M.', 'First Last', 'Last, First Middle' (v2.0.0 auto-parses 5+ formats)"
                    },
                    "year": {
                        "type": "string",
                        "description": "Publication year or 'n.d.' for no date"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of work"
                    },
                    "additional_fields": {
                        "type": "object",
                        "description": "Additional required fields (journal name, publisher, DOI, etc.)",
                        "additionalProperties": True
                    }
                },
                "required": ["reference_type", "author", "year", "title"]
            }
        )

    def _define_citation_generator(self) -> Tool:
        """Tool: Generate in-text citation."""
        return Tool(
            name="generate_citation",
            description=(
                "Generates APA 7 style in-text citation from reference information. "
                "Supports both parenthetical and narrative citations with optional page numbers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "author": {
                        "type": "string",
                        "description": "Author(s) in format 'Last, F. M.'"
                    },
                    "year": {
                        "type": "string",
                        "description": "Publication year"
                    },
                    "page": {
                        "type": "string",
                        "description": "Optional page number(s)"
                    },
                    "narrative": {
                        "type": "boolean",
                        "description": "True for narrative citation, False for parenthetical",
                        "default": False
                    }
                },
                "required": ["author", "year"]
            }
        )

    def _define_reference_validator(self) -> Tool:
        """Tool: Validate APA 7 reference completeness."""
        return Tool(
            name="validate_reference",
            description=(
                "Validates a reference for APA 7 compliance and completeness. "
                "Checks required fields, format, DOI, and provides actionable feedback."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "reference_type": {
                        "type": "string",
                        "enum": ["journal", "book", "chapter", "website", "report",
                                "conference", "dataset", "software", "dissertation", "government"]
                    },
                    "fields": {
                        "type": "object",
                        "description": "Reference fields to validate",
                        "additionalProperties": True
                    }
                },
                "required": ["reference_type", "fields"]
            }
        )

    def _define_data_requirement_checker(self) -> Tool:
        """Tool: Check data requirements for research type."""
        return Tool(
            name="check_data_requirements",
            description=(
                "Determines what data is required for a specific research type and "
                "validates whether synthetic data is acceptable. Provides clear guidance "
                "on empirical vs non-empirical requirements."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "research_type": {
                        "type": "string",
                        "enum": ["correlational", "comparative", "time_series", "observational",
                                "meta_analysis", "simulation", "methodological", "theoretical"],
                        "description": "Type of research"
                    },
                    "proposed_data_source": {
                        "type": "string",
                        "description": "Description of proposed data source"
                    }
                },
                "required": ["research_type"]
            }
        )

    def _define_methodology_advisor(self) -> Tool:
        """Tool: Get methodology guidance for research type."""
        return Tool(
            name="get_methodology_guidance",
            description=(
                "Provides step-by-step methodology guidance for a specific research type. "
                "Includes required steps, statistical tests, validation procedures, and "
                "common pitfalls to avoid."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "research_type": {
                        "type": "string",
                        "enum": ["correlational", "comparative", "time_series", "observational",
                                "meta_analysis", "simulation", "methodological", "theoretical"]
                    }
                },
                "required": ["research_type"]
            }
        )

    def _define_interpretation_validator(self) -> Tool:
        """Tool: Validate interpretation of results."""
        return Tool(
            name="validate_interpretation",
            description=(
                "Validates whether a proposed interpretation or claim is appropriate "
                "for the given research type and results. Helps prevent overclaiming "
                "and ensures proper causal language."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "research_type": {
                        "type": "string",
                        "enum": ["correlational", "comparative", "time_series", "observational",
                                "meta_analysis", "simulation", "methodological", "theoretical"]
                    },
                    "proposed_claim": {
                        "type": "string",
                        "description": "The claim or interpretation to validate"
                    },
                    "has_control_group": {
                        "type": "boolean",
                        "description": "Whether study had control group",
                        "default": False
                    },
                    "has_random_assignment": {
                        "type": "boolean",
                        "description": "Whether subjects were randomly assigned",
                        "default": False
                    }
                },
                "required": ["research_type", "proposed_claim"]
            }
        )

    def _define_guidelines_resource(self) -> MCPResource:
        """Resource: Research guidelines."""
        return MCPResource(
            uri="research-toolkit://guidelines/all",
            name="Research Guidelines",
            description="Complete research methodology guidelines covering all 8 research types",
            mimeType="text/markdown"
        )

    def _define_examples_resource(self) -> MCPResource:
        """Resource: Research examples."""
        return MCPResource(
            uri="research-toolkit://examples/all",
            name="Research Examples",
            description="8 complete working examples demonstrating proper research methodology (all use research_toolkit library)",
            mimeType="application/x-python"
        )

    def _define_templates_resource(self) -> MCPResource:
        """Resource: Research templates."""
        return MCPResource(
            uri="research-toolkit://templates/research",
            name="Research Template",
            description="Reusable template for conducting research (uses research_toolkit library v2.0.0)",
            mimeType="application/x-python"
        )

    # Tool Implementations

    def select_research_type(
        self,
        research_question: str,
        data_availability: str,
        goal: str
    ) -> Dict[str, Any]:
        """Implement research type selection."""

        # Decision logic
        if data_availability == "no_real_data" or data_availability == "model_only":
            if goal == "theorize":
                research_type = "theoretical"
                data_requirement = "None - pure theoretical work"
            elif goal == "predict":
                research_type = "simulation"
                data_requirement = "Model parameters (synthetic data acceptable with caveats)"
            else:
                return {
                    "status": "error",
                    "message": "Real data is required for descriptive, comparative, or correlational research",
                    "suggested_action": "Obtain real data or switch to theoretical/simulation research"
                }
        else:
            # Real data available
            if goal == "correlate":
                research_type = "correlational"
                data_requirement = "Real data showing relationship between variables"
            elif goal == "compare":
                research_type = "comparative"
                data_requirement = "Real data from distinct groups"
            elif goal == "describe":
                research_type = "observational"
                data_requirement = "Real observational data from target population"
            elif goal == "predict":
                research_type = "time_series"
                data_requirement = "Real temporal data with multiple time points"
            else:
                research_type = "observational"
                data_requirement = "Real data from phenomenon of interest"

        # Get methodology guidance
        methodology = self._get_methodology_summary(research_type)
        limitations = self._get_research_limitations(research_type)

        return {
            "status": "success",
            "research_type": research_type,
            "data_requirement": data_requirement,
            "methodology_summary": methodology,
            "key_limitations": limitations,
            "next_steps": [
                f"Review guidelines for {research_type} research",
                "Verify data meets requirements",
                "Design study following methodology",
                "Plan statistical analysis",
                "Prepare for limitations in interpretation"
            ]
        }

    def format_apa7_reference(
        self,
        reference_type: str,
        author: str,
        year: str,
        title: str,
        additional_fields: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Implement APA 7 reference formatting."""
        try:
            fields = {"author": author, "year": year, "title": title}
            if additional_fields:
                fields.update(additional_fields)

            key = self.apa_manager.add_reference(reference_type, **fields)
            formatted = self.apa_manager.format_reference(key)
            is_valid, issues = self.apa_manager.validate_reference(key)

            return {
                "status": "success",
                "formatted_reference": formatted,
                "citation_key": key,
                "is_valid": is_valid,
                "validation_issues": issues
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def generate_citation(
        self,
        author: str,
        year: str,
        page: Optional[str] = None,
        narrative: bool = False
    ) -> Dict[str, Any]:
        """Implement citation generation."""
        # Add temporary reference
        try:
            key = self.apa_manager.add_reference(
                'journal',  # Type doesn't matter for citation
                author=author,
                year=year,
                title='Temporary',
                journal='Temp',
                volume='1',
                pages='1'
            )

            citation = self.apa_manager.get_in_text_citation(key, page=page, narrative=narrative)

            return {
                "status": "success",
                "citation": citation,
                "citation_type": "narrative" if narrative else "parenthetical"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def validate_reference(
        self,
        reference_type: str,
        fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement reference validation."""
        try:
            # Temporarily add reference
            key = self.apa_manager.add_reference(reference_type, **fields)
            is_valid, issues = self.apa_manager.validate_reference(key)

            # Simple field validation
            field_validation = {}

            # Validate author
            author = fields.get('author', '')
            author_valid = bool(author and len(author) > 0)
            field_validation['author'] = {
                "valid": author_valid,
                "message": "OK" if author_valid else "Author field is required"
            }

            # Validate year
            year = fields.get('year', '')
            year_valid = bool(year and (year == 'n.d.' or year.isdigit()))
            field_validation['year'] = {
                "valid": year_valid,
                "message": "OK" if year_valid else "Year must be a 4-digit year or 'n.d.'"
            }

            # Validate DOI if present
            doi = fields.get('doi', '')
            doi_valid = True if not doi else (doi.startswith('10.') or doi.startswith('http'))
            field_validation['doi'] = {
                "valid": doi_valid,
                "message": "OK" if doi_valid else "DOI should start with '10.' or be a URL"
            }

            return {
                "status": "success",
                "is_valid": is_valid and author_valid and year_valid and doi_valid,
                "issues": issues,
                "field_validation": field_validation
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def check_data_requirements(
        self,
        research_type: str,
        proposed_data_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Implement data requirement checking."""

        empirical_types = {
            "correlational": {
                "requires_real_data": True,
                "minimum_data": "Paired observations of at least 2 variables from real-world phenomena",
                "synthetic_acceptable": False,
                "reason": "Empirical research requires verifiable real-world data"
            },
            "comparative": {
                "requires_real_data": True,
                "minimum_data": "Data from at least 2 distinct groups",
                "synthetic_acceptable": False,
                "reason": "Group comparisons must use real observed differences"
            },
            "time_series": {
                "requires_real_data": True,
                "minimum_data": "Multiple time points of real observations (minimum 30 recommended)",
                "synthetic_acceptable": False,
                "reason": "Temporal patterns must be from actual data"
            },
            "observational": {
                "requires_real_data": True,
                "minimum_data": "Real observations from population of interest",
                "synthetic_acceptable": False,
                "reason": "Descriptive research describes real phenomena"
            },
            "meta_analysis": {
                "requires_real_data": True,
                "minimum_data": "Multiple published studies with effect sizes",
                "synthetic_acceptable": False,
                "reason": "Meta-analysis synthesizes real research findings"
            },
            "simulation": {
                "requires_real_data": False,
                "minimum_data": "Model parameters (can be synthetic)",
                "synthetic_acceptable": True,
                "reason": "Simulations explore model behavior, BUT claims must be conditional",
                "warning": "Results are IF-THEN only. Must state 'According to model...' or 'IF assumptions hold...'"
            },
            "methodological": {
                "requires_real_data": False,
                "minimum_data": "Synthetic data appropriate for testing methods",
                "synthetic_acceptable": True,
                "reason": "Testing methods on controlled data is valid",
                "warning": "Claims are about the METHOD, not about real-world phenomena"
            },
            "theoretical": {
                "requires_real_data": False,
                "minimum_data": "No data required",
                "synthetic_acceptable": True,
                "reason": "Pure theory development",
                "warning": "Generates hypotheses that MUST be tested empirically before claiming validity"
            }
        }

        if research_type not in empirical_types:
            return {
                "status": "error",
                "message": f"Unknown research type: {research_type}"
            }

        requirements = empirical_types[research_type]

        response = {
            "status": "success",
            "research_type": research_type,
            **requirements
        }

        # Assess proposed data source if provided
        if proposed_data_source:
            is_synthetic = any(word in proposed_data_source.lower() for word in
                             ['synthetic', 'simulated', 'generated', 'random', 'fake'])

            if is_synthetic and not requirements['synthetic_acceptable']:
                response['assessment'] = "INVALID"
                response['assessment_message'] = (
                    f"Synthetic data is NOT acceptable for {research_type} research. "
                    f"{requirements['reason']}"
                )
            elif is_synthetic and requirements['synthetic_acceptable']:
                response['assessment'] = "ACCEPTABLE WITH CAVEATS"
                response['assessment_message'] = requirements.get('warning', '')
            else:
                response['assessment'] = "ACCEPTABLE"
                response['assessment_message'] = "Real data source appears appropriate"

        return response

    def _get_methodology_summary(self, research_type: str) -> str:
        """Get brief methodology summary."""
        summaries = {
            "correlational": "Measure two variables, calculate correlation, test significance, interpret strength and direction",
            "comparative": "Compare groups using appropriate test (t-test, ANOVA), calculate effect size, check assumptions",
            "time_series": "Collect temporal data, test stationarity, analyze trends, forecast if appropriate",
            "observational": "Systematically observe and describe, calculate descriptive statistics, identify patterns",
            "meta_analysis": "Systematically search literature, extract effect sizes, calculate pooled effects, assess heterogeneity",
            "simulation": "Define model, set parameters, run simulations, analyze outcomes WITH conditional claims",
            "methodological": "Design method test, apply to controlled data, evaluate performance, compare alternatives",
            "theoretical": "Define constructs, state axioms, derive propositions, generate testable hypotheses"
        }
        return summaries.get(research_type, "Unknown")

    def _get_research_limitations(self, research_type: str) -> List[str]:
        """Get key limitations for research type."""
        limitations = {
            "correlational": ["Cannot establish causation", "Third variables may explain relationship"],
            "comparative": ["Cannot establish causation without random assignment", "Groups may differ in unmeasured ways"],
            "time_series": ["Past patterns may not continue", "Cannot establish causation from temporal patterns"],
            "observational": ["Cannot establish causation", "Limited to description only"],
            "meta_analysis": ["Quality depends on primary studies", "Publication bias possible"],
            "simulation": ["Results depend entirely on model assumptions", "Requires empirical validation"],
            "methodological": ["Claims about method only, not phenomena", "Results may not generalize"],
            "theoretical": ["Requires empirical testing", "No claims about reality without validation"]
        }
        return limitations.get(research_type, [])

    def validate_interpretation(
        self,
        research_type: str,
        proposed_claim: str,
        has_control_group: bool = False,
        has_random_assignment: bool = False
    ) -> Dict[str, Any]:
        """Implement interpretation validation."""

        # Check for causal language
        causal_words = ['cause', 'causes', 'caused', 'effect of', 'due to', 'because of', 'leads to', 'results in']
        has_causal_language = any(word in proposed_claim.lower() for word in causal_words)

        # Determine if causal claims are acceptable
        can_claim_causation = has_control_group and has_random_assignment

        # Check for appropriate conditional language in non-empirical research
        non_empirical = research_type in ['simulation', 'methodological', 'theoretical']
        has_conditional = any(phrase in proposed_claim.lower() for phrase in
                            ['according to', 'if', 'under the assumption', 'the model suggests', 'theoretically'])

        issues = []
        warnings = []

        if has_causal_language and not can_claim_causation:
            issues.append("Causal language detected but study design does not support causal claims")
            issues.append("Remove words like 'cause', 'effect', 'due to' or redesign as experimental study")

        if non_empirical and not has_conditional:
            warnings.append("Non-empirical research should use conditional language (e.g., 'According to the model...')")

        if research_type == "correlational" and "correlation" not in proposed_claim.lower():
            warnings.append("Correlational studies should explicitly state findings are correlational")

        is_appropriate = len(issues) == 0

        return {
            "status": "success",
            "is_appropriate": is_appropriate,
            "has_causal_language": has_causal_language,
            "can_claim_causation": can_claim_causation,
            "issues": issues,
            "warnings": warnings,
            "suggested_revision": self._suggest_revision(proposed_claim, research_type, issues) if not is_appropriate else None
        }

    def _suggest_revision(self, claim: str, research_type: str, issues: List[str]) -> str:
        """Suggest revised claim."""
        revised = claim

        # Remove causal language
        causal_replacements = {
            'causes': 'is associated with',
            'caused': 'was associated with',
            'cause': 'associate with',
            'effect of': 'relationship with',
            'due to': 'associated with',
            'because of': 'correlated with',
            'leads to': 'is related to',
            'results in': 'correlates with'
        }

        for causal, correlational in causal_replacements.items():
            revised = revised.replace(causal, correlational)

        # Add conditional prefix for non-empirical
        if research_type in ['simulation', 'theoretical']:
            revised = f"According to the {research_type}, {revised}"

        return revised

    # MCP Protocol Methods

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools (MCP protocol)."""
        return [asdict(tool) for tool in self.tools]

    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available resources (MCP protocol)."""
        return [asdict(resource) for resource in self.resources]

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompt templates for AI models (MCP protocol)."""
        return [
            {
                "name": "research_methodology_primer",
                "description": "Core primer on conducting proper research - teaches principles and points to guidelines/examples",
                "arguments": []
            },
            {
                "name": "apa_formatting_guide",
                "description": "Guide to APA 7 formatting using MCP tools - teaches citation principles",
                "arguments": []
            },
            {
                "name": "claim_validation_guide",
                "description": "Guide to validating research claims match evidence - teaches limitations by design",
                "arguments": []
            },
            {
                "name": "data_source_guide",
                "description": "Guide to acceptable data sources by research type - teaches data requirements",
                "arguments": []
            }
        ]

    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """Get a specific prompt template."""
        prompts_content = {
            "research_methodology_primer": self._get_research_primer(),
            "apa_formatting_guide": self._get_apa_guide(),
            "claim_validation_guide": self._get_claim_validation_guide(),
            "data_source_guide": self._get_data_source_guide()
        }

        if prompt_name not in prompts_content:
            return {
                "status": "error",
                "message": f"Prompt not found: {prompt_name}"
            }

        prompts = self.list_prompts()
        prompt_info = next((p for p in prompts if p["name"] == prompt_name), None)

        return {
            "status": "success",
            "prompt_name": prompt_name,
            "description": prompt_info["description"],
            "prompt": prompts_content[prompt_name]
        }

    def _get_research_primer(self) -> str:
        """Get the research methodology primer text."""
        return """# Understanding Research Methodology

You are an AI model capable of conducting rigorous, verifiable research. This primer teaches you the core principles.

## Core Principle

**EMPIRICAL RESEARCH = MAKING VERIFIABLE CLAIMS ABOUT THE OBSERVABLE WORLD**

Before starting any research, understand:
1. **Verifiable** - Others must be able to check your work
2. **Reproducible** - Others must be able to repeat your methods  
3. **Falsifiable** - Your claims can be tested and potentially disproven
4. **Transparent** - All methods, data, and limitations must be documented

## The Critical Question: Do I Need Real Data?

Ask yourself: "Am I making claims about the REAL WORLD?"

**YES → You MUST use real, verifiable data**
- Examples: "Is population correlated with pollution?" "Do countries with higher GDP have better health?"
- Wrong: Using `np.random.normal()` to generate data
- Right: Using OpenWeatherMap API, World Bank data, government statistics

**NO → Then what AM I claiming?**
- About a METHOD? → Methodological research (synthetic data OK, see examples/07)
- About a THEORY? → Theoretical research (state as IF-THEN, see examples/08)
- About a MODEL? → Simulation research (validate against reality, see examples/06)

## Resources to Study

**BEFORE conducting research, read these files in this repository:**

1. **guidelines/AI_RESEARCH_GUIDELINES.md** - Complete methodology guide
   - Decision tree for research types
   - Data source requirements
   - Documentation standards
   - Common pitfalls to avoid

2. **templates/research_template.py** - Template structure for your research
   - Shows proper workflow
   - Includes all required sections
   - Demonstrates research_toolkit usage

3. **examples/** directory - Study examples relevant to your research type:
   - examples/00_synthetic_example_what_not_to_do.py ← Learn what NOT to do
   - examples/01_correlational_study.py ← Real data, finding relationships
   - examples/02_comparative_study.py ← Real data, comparing groups
   - examples/03_time_series_analysis.py ← Real data, temporal patterns
   - examples/04_observational_study.py ← Real data, describing phenomena
   - examples/05_meta_analysis.py ← Synthesizing existing research
   - examples/06_simulation_study.py ← Model-based exploration (conditional claims)
   - examples/07_methodological_study.py ← Testing methods (synthetic OK here)
   - examples/08_theoretical_model.py ← Theory development (IF-THEN claims)

## Using the MCP Server Tools

This server provides tools to help you conduct proper research:

**select_research_type** - Determines appropriate methodology for your question
- Use this FIRST to understand what type of research you should conduct
- Provides data requirements and limitations

**check_data_requirements** - Validates your data source is appropriate
- Use BEFORE starting analysis
- Will tell you if synthetic data invalidates your research

**validate_interpretation** - Checks if your claims match your design
- Use BEFORE making any claims
- Prevents over-claiming (e.g., claiming causation from correlation)

**format_apa7_reference** - Formats citations properly
- Use for ALL sources
- Supports 10 reference types with advanced name parsing

**generate_citation** - Creates in-text citations
- Narrative: "Smith (2023) found..."
- Parenthetical: "Research shows... (Smith, 2023)"

## Research Workflow

1. **Understand Your Question**
   - What am I claiming about?
   - Real world → Need real data
   - Methods/theory → Different approach

2. **Select Methodology**
   - Call select_research_type with your question
   - Read the returned methodology_summary and key_limitations
   - Study the relevant example file

3. **Validate Data Source**
   - Call check_data_requirements
   - If "INVALID" → Either get real data or change research type

4. **Follow the Template**
   - Use templates/research_template.py structure
   - Document everything transparently
   - Save raw data for verification

5. **Validate Claims**
   - Call validate_interpretation for EVERY claim
   - If "inappropriate" → Use the suggested_revision
   - Never claim more than your design supports

6. **State Limitations**
   - Be explicit about what you CANNOT conclude
   - Acknowledge study design limitations
   - Suggest future research

## Critical Reminders

❌ **NEVER DO THIS:**
- Make empirical claims with synthetic data
- Claim causation from correlation
- Hide non-significant results
- Overgeneralize from small samples
- Skip documentation or verification

✅ **ALWAYS DO THIS:**
- Use real data for empirical claims
- Match claims to evidence
- Report all results (including non-significant)
- State limitations explicitly
- Provide verification path

## Next Steps

1. Read: guidelines/AI_RESEARCH_GUIDELINES.md (complete guide)
2. Study: Relevant example files for your research type
3. Use: templates/research_template.py as your structure  
4. Validate: Use MCP tools throughout your workflow

Remember: It's better to say "I cannot answer this without real data" than to make unverifiable claims.
"""

    def _get_apa_guide(self) -> str:
        """Get the APA formatting guide text."""
        return """# APA 7 Formatting Guide

## Core Principle
**Every claim needs a source. Every source needs proper formatting.**

## Using MCP Tools

**format_apa7_reference** - Formats complete reference entries (10 types supported)
**generate_citation** - Creates in-text citations (narrative or parenthetical)
**validate_reference** - Checks reference completeness

## Quick Examples

Parenthetical: `generate_citation({"author": "Smith, J.", "year": "2023", "narrative": False})`  
Returns: (Smith, 2023)

Narrative: `generate_citation({"author": "Smith, J.", "year": "2023", "narrative": True})`  
Returns: Smith (2023)

With page: `generate_citation({"author": "Smith, J.", "year": "2023", "page": "45", "narrative": False})`  
Returns: (Smith, 2023, p. 45)

## Study These Examples
- examples/01_correlational_study.py - See how references are added
- examples/02_comparative_study.py - See in-text citation usage

Remember: Consistency is key. Use the tools for ALL citations.
"""

    def _get_claim_validation_guide(self) -> str:
        """Get the claim validation guide text."""
        return """# Validating Research Claims

## Core Principle
**Your claims must match what your research design can actually support.**

## Design Limitations

**Correlational (no control, no random assignment):**
- ✅ Can say: "X is associated with Y", "X correlates with Y"
- ❌ Cannot say: "X causes Y", "X leads to Y"

**Experimental (has control + random assignment):**
- ✅ Can say: "X caused Y", "X affects Y"
- This is the ONLY design supporting causal claims

**Simulation/Theoretical:**
- ✅ Can say: "According to the model...", "IF assumptions hold, THEN..."
- ❌ Cannot say: Direct claims about reality

## Using validate_interpretation

For EVERY claim:
```
validate_interpretation({
    "research_type": "correlational",
    "proposed_claim": "Your claim here",
    "has_control_group": False,
    "has_random_assignment": False
})
```

If `is_appropriate: False`, use the `suggested_revision`.

## Study These Examples
- examples/01_correlational_study.py - Notes cannot establish causation
- examples/02_comparative_study.py - Notes potential confounds
- examples/06_simulation_study.py - Makes only conditional claims

Remember: Over-claiming damages credibility. Be honest about limitations.
"""

    def _get_data_source_guide(self) -> str:
        """Get the data source requirements guide text."""
        return """# Data Source Requirements

## Core Principle
**The type of claims you make determines what data you need.**

## Decision Tree

Am I making claims about the REAL WORLD?
├─ YES → MUST use REAL, VERIFIABLE data
│         (APIs, government data, scientific repositories)
└─ NO → What am I claiming about?
          ├─ A METHOD → Synthetic OK (you're testing the method)
          ├─ A THEORY → Synthetic OK (but state as IF-THEN)
          └─ A MODEL → Synthetic for exploration (validate for real claims)

## Using check_data_requirements

Before analysis:
```
check_data_requirements({
    "research_type": "correlational",
    "proposed_data_source": "Description of your data"
})
```

Returns "ACCEPTABLE", "ACCEPTABLE WITH CAVEATS", or "INVALID"

## Documentation

For REAL data, save:
```python
data.to_csv('raw_research_data.csv')
# Document: source URL, access date, method, parameters
```

## Study These Examples
- examples/01_correlational_study.py - Uses real API data, saves raw data
- examples/07_methodological_study.py - Uses synthetic (testing method)
- examples/00_synthetic_example_what_not_to_do.py - What NOT to do

Remember: Real-world claims require real-world data.
"""

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool (MCP protocol)."""
        tool_methods = {
            "select_research_type": self.select_research_type,
            "format_apa7_reference": self.format_apa7_reference,
            "generate_citation": self.generate_citation,
            "validate_reference": self.validate_reference,
            "check_data_requirements": self.check_data_requirements,
            "validate_interpretation": self.validate_interpretation
        }

        if tool_name not in tool_methods:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        try:
            return tool_methods[tool_name](**arguments)
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Backward compatibility alias
EmpiricalResearchMCPServer = ResearchToolkitMCPServer


if __name__ == "__main__":
    print("=" * 70)
    print("Research Toolkit - MCP Server")
    print("=" * 70)

    server = ResearchToolkitMCPServer()

    print(f"\nServer: {server.name} v{server.version}")
    print(f"Tools available: {len(server.tools)}")
    print(f"Resources available: {len(server.resources)}")
    print(f"Prompts available: {len(server.list_prompts())}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION: MCP Server Capabilities")
    print("=" * 70)

    # Demo: List available prompts
    print("\n[0] Available AI Model Prompts (Educational Guides):")
    prompts = server.list_prompts()
    for i, prompt in enumerate(prompts, 1):
        print(f"    {i}. {prompt['name']}")
        print(f"       {prompt['description']}")

    print("\n" + "=" * 70)
    print("AVAILABLE TOOLS")
    print("=" * 70)
    for tool in server.tools:
        print(f"\n{tool.name}")
        print(f"  {tool.description}")

    print("\n" + "=" * 70)
    print("TOOL DEMONSTRATION")
    print("=" * 70)

    # Demo: Select research type
    print("\n[1] Selecting research type...")
    result = server.call_tool("select_research_type", {
        "research_question": "Is there a relationship between study hours and test scores?",
        "data_availability": "real_data_available",
        "goal": "correlate"
    })
    print(f"Research type: {result['research_type']}")
    print(f"Data requirement: {result['data_requirement']}")

    # Demo: Format APA 7 reference (v2.0.0 with advanced name parsing)
    print("\n[2] Formatting APA 7 reference...")
    print("    (Note: v2.0.0 supports 5+ author name formats)")
    result = server.call_tool("format_apa7_reference", {
        "reference_type": "journal",
        "author": "John Michael Smith",  # v2.0.0: Full name format works!
        "year": "2023",
        "title": "Research methods in psychology",
        "additional_fields": {
            "journal": "Journal of Psychology",
            "volume": "10",
            "pages": "123-145",
            "doi": "10.1234/jp.2023.123"
        }
    })
    print(f"Formatted:\n{result['formatted_reference']}")

    # Demo: Check data requirements
    print("\n[3] Checking data requirements...")
    result = server.call_tool("check_data_requirements", {
        "research_type": "correlational",
        "proposed_data_source": "randomly generated synthetic data"
    })
    print(f"Assessment: {result['assessment']}")
    print(f"Message: {result['assessment_message']}")

    # Demo: Validate interpretation
    print("\n[4] Validating interpretation...")
    result = server.call_tool("validate_interpretation", {
        "research_type": "correlational",
        "proposed_claim": "Study hours cause higher test scores",
        "has_control_group": False,
        "has_random_assignment": False
    })
    print(f"Appropriate: {result['is_appropriate']}")
    if not result['is_appropriate']:
        print(f"Issues: {', '.join(result['issues'])}")
        print(f"Suggested: {result['suggested_revision']}")

    # Demo: Get a prompt
    print("\n[5] Getting an Educational Prompt...")
    prompt_result = server.get_prompt("research_methodology_primer")
    if prompt_result['status'] == 'success':
        print(f"Prompt: {prompt_result['prompt_name']}")
        print(f"Description: {prompt_result['description']}")
        print(f"Content length: {len(prompt_result['prompt'])} characters")
        print("First 300 chars:")
        print(f"  {prompt_result['prompt'][:300]}...")

    print("\n[OK] MCP Server demonstration complete!")
    print("\nKey Features (v2.0.0):")
    print("  - 8 research types supported")
    print("  - 10 APA 7 reference types")
    print("  - Advanced name parsing (5+ formats)")
    print("  - Automatic initials generation")
    print("  - Cross-platform compatibility")
    print("  - 4 AI model educational prompts")
    print("  - Comprehensive validation tools")
