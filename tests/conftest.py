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

import os
from unittest.mock import MagicMock
import google.auth

# Load dummy credentials if default credentials are not set up,
# to prevent import-time crashes in tests.
try:
    google.auth.default()
except Exception:
    dummy_credentials = MagicMock()
    dummy_credentials.token = None
    dummy_credentials.valid = False
    google.auth.default = lambda *args, **kwargs: (
        dummy_credentials,
        os.environ.get("GOOGLE_CLOUD_PROJECT", "dummy-project"),
    )

import pytest
from unittest.mock import patch
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.genai import types

@pytest.fixture(autouse=True)
def mock_gemini_generate_content():
    async def mock_generate_content_async(self, llm_request, stream=False):
        # We can construct the mocked response text
        response_text = '{"response": "Mocked response", "needs_confirmation": false}'
        
        # If the user prompt specifically asks for something that needs confirmation:
        prompt = ""
        if llm_request.contents:
            for content in llm_request.contents:
                if content.parts:
                    prompt += " ".join(part.text for part in content.parts if part.text)
        
        if "confirm" in prompt.lower():
            response_text = '{"response": "Confirming the action.", "needs_confirmation": true}'
        
        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=response_text)]
            )
        )
        yield response

    with patch.object(Gemini, "generate_content_async", mock_generate_content_async):
        yield

