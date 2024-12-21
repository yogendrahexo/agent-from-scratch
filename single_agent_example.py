from dotenv import load_dotenv
_ = load_dotenv()

import json
import argparse
from agent import ProviderType, pretty_print_messages, Agent, Swarm


def get_weather(location, time="now"):
    """Get the current weather in a given location. Location MUST be a city.
    :param location: The city to get weather for
    :param time: The time to get weather for, defaults to 'now'
    """
    return json.dumps({"location": location, "temperature": "65", "time": time})


def send_email(recipient, subject, body):
    """Send an email to a recipient.
    :param recipient: Email address of the recipient
    :param subject: Subject line of the email
    :param body: Main content/body of the email
    """
    return f"Sent! email to {recipient} with the subject: {subject} and body: {body}"


parser = argparse.ArgumentParser()
parser.add_argument('--provider', type=str, default='oai', 
                   choices=['oai', 'bant'],
                   help='Provider to use (default: oai)')
args = parser.parse_args()

provider = ProviderType.OPENAI if args.provider == 'oai' else ProviderType.BEDROCK_ANTHROPIC
print(f"Using provider: {provider}")

weather_agent = Agent(
    name="Weather Agent",
    provider=provider,
    instructions="You are a helpful agent.",
    functions=[get_weather, send_email],
)

client = Swarm()
print("Starting Single Agent - Weather Agent")
print('Ask me how is the weather today in Brussels?')

messages = []
agent = weather_agent

while True:
    user_input = input("\033[90mUser\033[0m: ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(agent=agent, messages=messages)
    pretty_print_messages(response.messages, provider=provider)

    messages.extend(response.messages)
    agent = response.agent