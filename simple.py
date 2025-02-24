from agent import Swarm, ProviderType, pretty_print_messages
import os
from dotenv import load_dotenv
_ = load_dotenv()

def run_with_provider(provider_type: ProviderType, message: str):
    print(f"\n{'='*50}")
    print(f"Using {provider_type.value.upper()}")
    print('='*50)
    
    swarm = Swarm(provider=provider_type)
    response = swarm.run(
        messages=[{"role": "user", "content": message}]
    )
    
    # Print the conversation
    pretty_print_messages(response.messages, provider=provider_type)
    
    # Print agent info (optional)
    if response.agent:
        print("\nAgent Configuration:")
        print(f"- Name: {response.agent.name}")
        print(f"- Model: {response.agent.model}")
        print(f"- Provider: {response.agent.provider.value}")

# Test each provider
run_with_provider(
    ProviderType.OPENAI,
    "Hello! I'm using OpenAI."
)

run_with_provider(
    ProviderType.AZURE_OPENAI,
    "Hello! I'm using Azure OpenAI."
)

run_with_provider(
    ProviderType.BEDROCK_ANTHROPIC,
    "Hello! I'm using Bedrock Anthropic."
)