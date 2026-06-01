import json
import os
import time
from datetime import datetime
import agentrx.pipeline.globals as g
import agentrx.reports.metrics as metrics
from agentrx.llm_clients import observed_snapshot

from openai import AzureOpenAI
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
    DefaultAzureCredential
)

class LLMAgent:
    def __init__(
            self,
            api_version,  
            model_name,  
            model_version, 
            deployment_name,
    ):     
        self.api_version = api_version
        self.model_name = model_name
        self.model_version = model_version
        self.deployment_name = deployment_name
        self.endpoint = g.ENDPOINT
        
        # Initialize telemetry tracking
        self.last_call_telemetry = None

        # Authenticate with Azure using Managed Identity Credential
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(managed_identity_client_id=g.CLIENT_ID),
            "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider
        )

    def get_llm_response(self, messages):
        # Capture timing before the call
        start_timestamp = datetime.now()
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        # Capture timing after the call
        end_time = time.perf_counter()
        end_timestamp = datetime.now()
        execution_time_sec = round(end_time - start_time, 4)

        # Record what the server actually served (D1: drift diagnosability).
        observed_snapshot.record(
            served_model=getattr(response, "model", None),
            system_fingerprint=getattr(response, "system_fingerprint", None),
        )

        # Extract token usage from response
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(response, "usage") and response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            total_tokens = response.usage.total_tokens or 0

        # Create telemetry objects
        token_usage = metrics.TokenUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens
        )

        time_info = metrics.TimingInfo(
            start_time=start_timestamp,
            end_time=end_timestamp,
            execution_time_sec=execution_time_sec
        )

        self.last_call_telemetry = metrics.LLMCallTelemetry(
            tokens=token_usage,
            time=time_info,
            model_name=self.model_name,
            instance=self.endpoint
        )

        return response

    @staticmethod
    def azure_mk_client() -> AzureOpenAI:
        """Create an Azure OpenAI client using credentials from globals/.env."""
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(managed_identity_client_id=g.CLIENT_ID),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            api_version=g.API_VERSION,
            azure_endpoint=g.ENDPOINT,
            azure_ad_token_provider=token_provider,
        )