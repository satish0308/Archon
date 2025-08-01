from __future__ import annotations as _annotations

import logfire
import os
import sys
from pydantic_ai import Agent
from dotenv import load_dotenv
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client
from pydantic_ai.providers.openai import OpenAIProvider

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import get_env_var
from archon.agent_prompts import prompt_refiner_prompt

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

reasoner_llm_model_name = get_env_var('PRIMARY_MODEL') or 'o3-mini'
is_anthropic = provider == "Anthropic"
is_openai = provider == "OpenAI"
is_ollama = provider == "Ollama"

model = (
    AnthropicModel(reasoner_llm_model_name, api_key=api_key)
    if is_anthropic else
    OpenAIModel(
        model_name=reasoner_llm_model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key)
    )
    if is_openai else
    OpenAIModel(
        model_name=reasoner_llm_model_name,
        provider=OpenAIProvider(base_url=base_url, api_key="adaad")
    )
)
logfire.configure(send_to_logfire='if-token-present')

prompt_refiner_agent = Agent(
    model,
    system_prompt=prompt_refiner_prompt
)