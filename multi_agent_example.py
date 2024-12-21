import argparse
from dotenv import load_dotenv
_ = load_dotenv()

from agent import ProviderType, pretty_print_messages, Agent, Swarm

# Initialize Swarm with telemetry
client = Swarm()
parser = argparse.ArgumentParser()
parser.add_argument('--provider', type=str, default='oai', 
                   choices=['oai', 'bant'],
                   help='Provider to use (default: oai)')
args = parser.parse_args()

provider = ProviderType.OPENAI if args.provider == 'oai' else ProviderType.BEDROCK_ANTHROPIC
print(f"Using provider: {provider}")


def process_refund(item_id, reason="NOT SPECIFIED"):
    """Process a refund for a specific item.
    
    This function handles refund requests by processing the refund for a given item.
    The item_id must be in the format 'item_XXX'. User confirmation should be obtained
    before processing the refund.
    
    Args:
        item_id (str): The ID of the item to refund, must start with 'item_'
        reason (str, optional): The reason for the refund. Defaults to "NOT SPECIFIED"
    
    Returns:
        str: Success message if refund is processed
    """
    print(f"[mock] Refunding item {item_id} because {reason}...")
    return "Success!"

def apply_discount():
    """Apply a standard discount to the user's current shopping cart.
    
    This function applies an 11% discount to all items currently in the user's
    shopping cart. The discount is applied immediately and reflected in the total.
    
    Returns:
        str: Message confirming the discount percentage applied
    """
    print("[mock] Applying discount...")
    return "Applied discount of 11%"


triage_agent = Agent(
    name="Triage Agent",
    provider=provider,
    instructions="""Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.
    - For purchases, pricing, discounts and product inquiries -> Sales Agent
    - For refunds, returns and complaints -> Refunds Agent
    Never handle requests directly - always transfer to the appropriate specialist.""",
)
sales_agent = Agent(
    name="Sales Agent",
    provider=provider,
    instructions="Be super enthusiastic about selling bees.",
)

refunds_agent = Agent(
    name="Refunds Agent",
    provider=provider,
    instructions="Help the user with a refund. If the reason is that it was too expensive, offer the user a refund code. If they insist, then process the refund.",
    functions=[process_refund, apply_discount],
)


def transfer_back_to_triage():
    """Transfer the conversation back to the triage agent. Call this function if a user is asking about a topic that is not handled by the current agent."""
    return triage_agent


def transfer_to_sales():
    """Transfer the conversation to the sales agent. Call this function for inquiries about purchases, pricing, discounts and product information."""
    return sales_agent


def transfer_to_refunds():
    """Transfer the conversation to the refunds agent. Call this function for handling refunds, returns and customer complaints."""
    return refunds_agent


triage_agent.functions = [transfer_to_sales, transfer_to_refunds]
sales_agent.functions.append(transfer_back_to_triage)
refunds_agent.functions.append(transfer_back_to_triage)

print("Starting Multiple Agents - Triage Agent, Refunds Agent and Bee Sales Agent")

messages = []
agent = triage_agent

while True:
    user_input = input("\033[90mUser\033[0m: ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(agent=agent, messages=messages)
    
    pretty_print_messages(response.messages, provider=provider)
    messages.extend(response.messages)
    agent = response.agent