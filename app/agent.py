# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
import os
import re
import google.auth
from dotenv import load_dotenv

from google.adk.workflow import Workflow, node, Edge
from google.adk.agents import LlmAgent
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables first
load_dotenv()

# Initialize Google Cloud Project details with fallback to prevent crashes on import
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not project_id:
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    except Exception:
        # Fallback if credentials/ADC not configured yet
        pass

if "GOOGLE_CLOUD_LOCATION" not in os.environ:
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

if "GOOGLE_GENAI_USE_VERTEXAI" not in os.environ:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        query: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


# --- Security Helper Functions ---

def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Scrubs SSNs and Credit Card numbers from text and tracks redacted categories."""
    redacted_categories = []
    
    # SSN patterns: XXX-XX-XXXX or XXXXXXXXX
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    ssn_raw_pattern = r'\b\d{9}\b'
    
    # Credit Card pattern: 13 to 16 digits with optional spaces or dashes
    cc_pattern = r'\b(?:\d[ -]*?){13,16}\b'
    
    scrubbed = text
    if re.search(ssn_pattern, scrubbed) or re.search(ssn_raw_pattern, scrubbed):
        scrubbed = re.sub(ssn_pattern, '[REDACTED SSN]', scrubbed)
        scrubbed = re.sub(ssn_raw_pattern, '[REDACTED SSN]', scrubbed)
        redacted_categories.append("SSN")
        
    if re.search(cc_pattern, scrubbed):
        scrubbed = re.sub(cc_pattern, '[REDACTED CREDIT CARD]', scrubbed)
        redacted_categories.append("Credit Card")
        
    return scrubbed, redacted_categories


def detect_prompt_injection(text: str) -> bool:
    """Defends against prompt injection by scanning for instruction-override keywords."""
    injection_keywords = [
        "ignore previous instructions",
        "ignore all previous",
        "system override",
        "override system",
        "bypass rules",
        "bypass configuration",
        "force auto-approval",
        "force auto approval",
        "approve automatically",
        "auto-approve this",
        "ignore rules",
    ]
    text_lower = text.lower()
    for kw in injection_keywords:
        if kw in text_lower:
            return True
    return False


# --- ADK 2.0 Workflow API Implementation ---

@node
def security_checkpoint(ctx: Context, node_input: object) -> Event:
    """Security checkpoint to scrub PII and defend against prompt injections."""
    text = ""
    if isinstance(node_input, str):
        text = node_input
    elif hasattr(node_input, "parts") and node_input.parts:
        text = "".join(part.text for part in node_input.parts if part.text)
    elif isinstance(node_input, list):
        parts_text = []
        for item in node_input:
            if isinstance(item, str):
                parts_text.append(item)
            elif hasattr(item, "parts") and item.parts:
                parts_text.append("".join(part.text for part in item.parts if part.text))
            elif hasattr(item, "text") and item.text:
                parts_text.append(item.text)
        text = "".join(parts_text)
    elif hasattr(node_input, "text") and getattr(node_input, "text"):
        text = node_input.text
    elif node_input:
        text = str(node_input)
    
    scrubbed_desc, redacted_categories = scrub_pii(text)
    
    state_delta = {}
    if redacted_categories:
        state_delta["redacted_categories"] = redacted_categories
    
    # Check for prompt injection
    if detect_prompt_injection(scrubbed_desc):
        payload = {
            "response": "Security Alert: Potential prompt injection attempt blocked. Force-routed to human review.",
            "needs_confirmation": True,
            "is_security_event": True,
            "description": scrubbed_desc,
        }
        state_delta["security_flagged"] = True
        state_delta["original_description_scrubbed"] = scrubbed_desc
        return Event(output=payload, route="suspicious", state=state_delta)
    
    state_delta["original_description_scrubbed"] = scrubbed_desc
    # Clean flow: proceed to LLM reviewer with the scrubbed description
    return Event(output=scrubbed_desc, route="clean", state=state_delta)


class AgentOutput(BaseModel):
    response: str = Field(description="The response to display to the user.")
    needs_confirmation: bool = Field(
        description="Set to True if the request requires user confirmation (e.g. contains the word 'confirm' or requests sensitive action)."
    )


assistant_node = LlmAgent(
    name="assistant",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a helpful AI assistant designed to provide accurate and useful information. "
        "Use the weather or time tools when needed. "
        "If the user asks to perform an action containing the word 'confirm' or a sensitive action, "
        "set needs_confirmation to True. Otherwise, set it to False."
    ),
    tools=[get_weather, get_current_time],
    output_schema=AgentOutput,
)


@node
def route_request(node_input: AgentOutput) -> Event:
    """Routes the request based on whether confirmation is needed."""
    if node_input.needs_confirmation:
        return Event(output=node_input.model_dump(), route="needs_confirm")
    return Event(output=node_input.response, route="direct")


@node
async def human_confirmation(ctx: Context, node_input: dict):
    """Requests confirmation from the user (human-in-the-loop)."""
    if not ctx.resume_inputs or "approve" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="approve",
            message="Do you approve executing this request? (yes/no)"
        )
        return

    user_approval = ctx.resume_inputs["approve"]
    
    is_sec = False
    response = ""
    if isinstance(node_input, dict):
        is_sec = node_input.get("is_security_event", False)
        response = node_input.get("response", "")
    else:
        response = str(node_input)
        
    prefix = "[SECURITY WARNING: Flagged Event] " if is_sec else ""
    if user_approval.lower() in ["yes", "y", "confirm"]:
        yield Event(output=f"{prefix}{response}", route="approved")
    else:
        yield Event(output="Operation cancelled by user.", route="cancelled")


@node
def format_final_output(node_input: str):
    """Emits the UI content and returns the output value."""
    yield Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=node_input)]
        )
    )
    yield Event(output=node_input)


root_agent = Workflow(
    name="root_agent",
    edges=[
        ('START', security_checkpoint),
        (security_checkpoint, {
            "clean": assistant_node,
            "suspicious": human_confirmation,
        }),
        (assistant_node, route_request),
        (route_request, {
            "needs_confirm": human_confirmation,
            "direct": format_final_output,
        }),
        Edge(
            from_node=human_confirmation,
            to_node=format_final_output,
            route=["approved", "cancelled"],
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
