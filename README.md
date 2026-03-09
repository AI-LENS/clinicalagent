# clinicalagent

**AI-powered agent framework for clinical trials intelligence.**

clinicalagent is a lightweight, flexible Python framework for building and orchestrating AI agents tailored for clinical trial workflows. It is designed to run locally, with a tool definition system closely aligned with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), making it simple to integrate clinical data sources, trial registries, and biomedical knowledge bases.

## Features

- **Clinical Trial Intelligence**: Purpose-built for clinical trial analysis — eligibility screening, protocol summarization, adverse event detection, and more.
- **Local-First**: Run agents securely in your local environment, keeping sensitive patient and trial data under your control.
- **MCP Integration**: Seamlessly integrate with MCP servers to connect trial registries (e.g., ClinicalTrials.gov), EHR systems, and lab data tools.
- **Thoughtful Agents**: Optional "thought" process visibility for debugging, auditing, and regulatory transparency.
- **Highly Configurable**: Easy configuration via YAML or environment variables to adapt to different therapeutic areas and trial phases.
- **Streaming**: Built-in support for streaming agent responses in real time.
- **Extensible**: Create custom environments to control tool execution, clinical context management, and domain-specific logic.

## Installation

You can install clinicalagent directly from PyPI:

```bash
pip install clinicalagent
```

## Quick Start

Here is a simple example of how to create an agent that can look up clinical trial information and assess patient eligibility.

### 1. Define Tools & Environment

Create a file named `main.py`:

```python
import asyncio
from fastmcp import Client, FastMCP
from clinicalagent import DefaultEnvironment, Agent
from clinicalagent.types import BaseToolModel, CallToolRequestParams
from clinicalagent.utils import mcp_tools

# 1. Define clinical trial tools using FastMCP
mcp = FastMCP()

@mcp.tool
def search_trials(condition: str, phase: str = "") -> str:
    """Search for active clinical trials by medical condition and optional trial phase (e.g., 'Phase 2')."""
    # In production, this would query ClinicalTrials.gov or an internal registry
    return f"Found 12 active trials for '{condition}'" + (f" in {phase}" if phase else "")

@mcp.tool
def check_eligibility(trial_id: str, patient_age: int, diagnosis: str) -> str:
    """Check whether a patient meets the eligibility criteria for a specific clinical trial."""
    # In production, this would evaluate inclusion/exclusion criteria
    return (
        f"Patient (age {patient_age}, diagnosis: {diagnosis}) "
        f"meets eligibility criteria for trial {trial_id}."
    )

@mcp.tool
def get_trial_summary(trial_id: str) -> str:
    """Retrieve a summary of a clinical trial including phase, sponsor, endpoints, and status."""
    return (
        f"Trial {trial_id}: Phase 3, Randomized, Double-Blind study "
        f"evaluating Drug X vs Placebo in Advanced NSCLC. "
        f"Primary endpoint: Progression-Free Survival. Status: Recruiting."
    )

# 2. Create an MCP client
client = Client(mcp)

# 3. Define the Environment
# The environment handles tool execution and clinical context management
class ClinicalTrialEnvironment[T: BaseToolModel](DefaultEnvironment[T]):
    async def call_tool(self, action: CallToolRequestParams[T]) -> str | None:
        async with client:
            result = await client.call_tool(
                name=action.tool_name,
                arguments=action.arguments.model_dump()
            )
        return "\n".join(res.text for res in result.content if res.type == "text")

# 4. Initialize Agent
async def main():
    # Initialize environment with clinical trial tools from the MCP client
    env = ClinicalTrialEnvironment(tools=mcp_tools(client))

    # Create the agent
    agent = Agent(
        environment=env,
        disable_thought=False  # Set to False to see the agent's reasoning process
    )

    # Add a clinical query to the history
    env.history.add_message(
        role="user",
        content="Find active Phase 3 clinical trials for non-small cell lung cancer and check if a 58-year-old patient with stage IIIB NSCLC is eligible."
    )

    # Run the agent
    async for event in agent.stream():
        print(event.model_dump_json(indent=2))
        print("------------")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Configure LLM

Create a `clinicalagent-config.yaml` file in your project root to configure your LLM provider:

```yaml
llm_model_name: "gpt-4o"
llm_api_key: "your-api-key-here"
# Optional: Base URL for other compatible providers
# llm_base_url: "https://api.openai.com/v1"
```

### 3. Run

```bash
python main.py
```

## Configuration

clinicalagent uses `pydantic-settings` for configuration. You can configure it using a `clinicalagent-config.yaml` file, a `.env` file, or environment variables.

| Setting                | Description                                    | Default |
| ---------------------- | ---------------------------------------------- | ------- |
| `llm_model name`       | The name of the LLM model to use.              | `None`  |
| `llm_api_key`          | API key for the LLM provider.                  | `None`  |
| `llm_base_url`         | Base URL for the LLM API.                      | `None`  |
| `max_agent_iterations` | Maximum number of loops the agent can perform. | `7`     |
| `max_history_length`   | Maximum number of messages to keep in history. | `11`    |
| `llm_api_extra_kw`     | Extra keyword arguments for the LLM API call.  | `{}`    |

## Use Cases

clinicalagent can be applied across the clinical trial lifecycle:

- **Trial Discovery & Matching**: Search trial registries and match patients to eligible studies based on diagnosis, demographics, and biomarkers.
- **Protocol Analysis**: Summarize and compare trial protocols, identify inclusion/exclusion criteria, and flag potential issues.
- **Adverse Event Monitoring**: Analyze safety reports, detect signals, and classify adverse events by severity and causality.
- **Regulatory Document Assistance**: Draft or review sections of regulatory submissions (e.g., IND, CSR) with AI-assisted writing.
- **Site Feasibility Assessment**: Evaluate potential trial sites based on patient population, historical enrollment rates, and infrastructure.
- **Literature Review**: Search and synthesize relevant biomedical literature to support trial design and evidence generation.

## Advanced Usage

### Custom Environments

The `Environment` class is the heart of clinicalagent's extensibility. By subclassing `DefaultEnvironment` or implementing the `Environment` protocol, you can:

- **Integrate Clinical Data Sources**: Connect to EHR/EMR systems, CTMS platforms, lab data feeds, and trial registries.
- **Customize Tool Execution**: Handle tool calls locally, remotely, or via complex clinical data pipelines.
- **Manage Clinical Context**: Control how patient history, trial data, and conversation context are stored and presented to the LLM.
- **Implement Termination Logic**: Define custom criteria for when the agent should stop — e.g., after eligibility is confirmed or an adverse event is flagged.

### Thought Process

clinicalagent can expose the agent's internal "thought" process. When `disable_thought=False` is passed to the `Agent` constructor, the agent will generate a thought trace before taking actions or answering. This is particularly valuable in clinical settings for:

- **Audit Trails**: Understanding why the agent made specific clinical recommendations.
- **Debugging**: Tracing the reasoning behind trial matching or eligibility decisions.
- **Regulatory Transparency**: Providing explainable AI outputs for compliance and review.

## License

[MIT](LICENSE)
