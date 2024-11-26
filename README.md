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

# Going deeper
1. [Planning and Reasoning with LLMs](https://hexoai.notion.site/Planning-and-Reasoning-with-LLMs-09ed06fe3a3b45f494760d606c4f285b?pvs=74)
2. [Cognitive Architectures for Language Agents](https://arxiv.org/pdf/2309.02427v3)
