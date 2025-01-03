# agent-from-scratch

agent-from-scratch is a Python-based repository for developers and researchers to explore the fundamentals of single and multi-agent systems.

It is a fork of OpenAI's Swarm, which is already straightforward. However, agent-from-scratch is even simpler, making it easier to quickly start and understand single and multi-agent systems.

# Getting started

1. Clone or fork the repository: `git clone https://github.com/hexo-ai/agent-from-scratch.git`
2. To set up the conda environment, run the following command: `conda env create -f environment.yml`. Alternatively, you can use a virtual environment.
3. Create a `.env` file by copying the structure from `.env.template`.
4. Add your environment variables to the `.env` file.
5. Activate the conda environment using `conda activate agent-from-scratch` or your virtual environment.
6. Install the requirements using `pip install -r requirements.txt`.
7. To run the single agent example, execute `python single_agent_example.py`. This script implements a weather agent with capabilities to send emails.
8. To run the multi-agent example, execute `python multi_agent_example.py`. This script implements sales and refund agents with capabilities to apply discounts and process refunds.

Arguments:

- `--provider`: Choose the AI provider (default: 'oai')
  - 'oai': OpenAI
  - 'bant': Bedrock Anthropic

## Environment Variables

Required environment variables in your `.env` file:

- For OpenAI: `OPENAI_API_KEY`
- For Amazon Bedrock: AWS credentials configured via:
  - AWS CLI: Run `aws configure` and enter your credentials
  - Environment variables:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_DEFAULT_REGION` (recommended: us-west-2)

## Running Examples

### Single Agent

```bash
python single_agent_example.py --provider [oai|bant]
```

### Multi Agent

```bash
python multi_agent_example.py --provider [oai|bant]
```

Multiple agents working together:

- Triage Agent: Routes requests
- Sales Agent: Handles purchases
- Refunds Agent: Processes returns
