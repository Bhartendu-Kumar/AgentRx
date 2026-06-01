import os
import time
from dotenv import load_dotenv

load_dotenv()
from azure.ai.inference import ChatCompletionsClient
from openai.lib.azure import AzureOpenAI
from openai import RateLimitError
import agentrx.pipeline.globals as g
from agentrx.llm_clients import observed_snapshot
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)

MAX_RETRIES = 5
INITIAL_BACKOFF = 4  # seconds
class LLMAgent:
    def __init__(
            self,
            api_version,
            model_name,
            model_version,
            deployment_name):
        self.scope = os.getenv("SCOPE", "")
        self.credential = get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
        self.scope,
    )
        self.model_name = g.TRAPI_DEPLOYMENT_NAME
        self.instance = g.TRAPI_INSTANCE
        self.api_version = api_version
        self.endpoint = f"{g.TRAPI_ENDPOINT_PREFIX}{self.instance}"
        self.llm_client = AzureOpenAI(
        azure_endpoint=self.endpoint,
        azure_ad_token_provider=self.credential,
        api_version=self.api_version,
    )

    def get_llm_response(self, messages):
        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                observed_snapshot.record(
                    served_model=getattr(response, "model", None),
                    system_fingerprint=getattr(response, "system_fingerprint", None),
                )
                return response
            except RateLimitError as e:
                if attempt < MAX_RETRIES - 1:
                    wait = INITIAL_BACKOFF * (2 ** attempt)
                    print(f"[TRAPI] Rate limited, retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    raise
    
    @staticmethod
    def trapi_mk_client() -> AzureOpenAI:
        scope = os.getenv("SCOPE", "")
        credential = get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
            scope,
        )
        return AzureOpenAI(
            azure_endpoint=f"{g.TRAPI_ENDPOINT_PREFIX}{g.TRAPI_INSTANCE}",
            azure_ad_token_provider=credential,
            api_version=g.TRAPI_API_VERSION,
        )

