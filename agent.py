import json
import copy
import inspect
from enum import Enum
import time
import random
from botocore.exceptions import ClientError
import os

from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel
from typing_extensions import Literal
from typing import Union, Callable, List, Optional
import boto3



class ProviderType(Enum):
    OPENAI = "openai"
    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    AZURE_OPENAI = "azure_openai"

def pretty_print_messages(messages, provider: ProviderType = "openai") -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        if provider == ProviderType.BEDROCK_ANTHROPIC:
            # Handle Bedrock format
            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if "text" in content:
                        print(content["text"])
                    elif "toolUse" in content:
                        tool = content["toolUse"]
                        name = tool["name"]
                        args = json.dumps(tool["input"]).replace(":", "=")
                        print(f"\033[95m{name}\033[0m({args[1:-1]})")
        else:
            # Handle OpenAI format
            if message.get("content"):
                print(message["content"])

            # print tool calls in purple, if any
            tool_calls = message.get("tool_calls") or []
            if len(tool_calls) > 1:
                print()
            for tool_call in tool_calls:
                f = tool_call["function"]
                name, args = f["name"], f["arguments"]
                arg_str = json.dumps(json.loads(args)).replace(":", "=")
                print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")

def function_to_json(func) -> dict:
    """
    Sample Input:
    def add_two_numbers(a: int, b: int) -> int:
        # Adds two numbers together
        return a + b
    
    Sample Output:
    {
        'type': 'function',
        'function': {
            'name': 'add_two_numbers',
            'description': 'Adds two numbers together',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'integer'},
                    'b': {'type': 'integer'}
                },
                'required': ['a', 'b']
            }
        }
    }
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

AgentFunction = Callable[[], Union[str, "Agent", dict]]



class Agent(BaseModel):
    name: str = "Agent"
    provider: ProviderType = ProviderType.OPENAI
    model: str = "gpt-4o"  # default for OpenAI
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[Callable] = []
    tool_choice: Optional[str] = None
    parallel_tool_calls: bool = True
    # Azure OpenAI specific configs
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: str = "2024-02-15-preview"

    def __init__(self, **data):
        super().__init__(**data)
        # Set default model based on provider if not explicitly specified
        if 'model' not in data:
            if self.provider == ProviderType.BEDROCK_ANTHROPIC:
                self.model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
            if self.provider == ProviderType.AZURE_OPENAI:
                self.model = os.getenv("AZURE_DEPLOYMENT_NAME")
                self.api_version = os.getenv("AZURE_API_VERSION")
                self.azure_endpoint = os.getenv("AZURE_ENDPOINT_URL")

            
            
            
            
            

class Response(BaseModel):
    # Response is used to encapsulate the entire conversation output
    messages: List = []
    agent: Optional[Agent] = None
    
class Function(BaseModel):
    arguments: str
    name: str

class ChatCompletionMessageToolCall(BaseModel):
    id: str # The ID of the tool call
    function: Function # The function that the model called
    type: Literal["function"] # The type of the tool. Currently, only `function` is supported

class Result(BaseModel):
    # Result is used to encapsulate the return value of a single function/tool call
    value: str = "" # The result value as a string.
    agent: Optional[Agent] = None # The agent instance, if applicable.


class Swarm:
    def __init__(self, client=None, provider: ProviderType = ProviderType.OPENAI):
        self.openai_client = None
        self.bedrock_client = None
        self.azure_client = None
        self.default_provider = provider
        
        if client:
            if isinstance(client, AzureOpenAI):
                self.azure_client = client
            elif isinstance(client, OpenAI):
                self.openai_client = client
            else:
                self.bedrock_client = client
    
    def get_client(self, provider: ProviderType):
        if provider == ProviderType.OPENAI:
            if not self.openai_client:
                self.openai_client = OpenAI()
            return self.openai_client
        elif provider == ProviderType.AZURE_OPENAI:
            if not self.azure_client:
                self.azure_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT_URL"),
                    api_version=os.getenv('AZURE_API_VERSION', "2024-02-15-preview"),
                )
            return self.azure_client
        else:
            if not self.bedrock_client:
                self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")
            return self.bedrock_client

    def function_to_json(self, func, provider: ProviderType) -> dict:
        """Convert function to provider-specific format"""
        if provider == ProviderType.OPENAI:
            return self._openai_function_to_json(func)
        elif provider == ProviderType.AZURE_OPENAI:
            return self._azure_function_to_json(func)
        else:
            return self._bedrock_function_to_json(func)

    def _openai_chat_completion(
        self,
        agent: Agent,
        history: List,
        model_override: str
    ):
        client = self.get_client(ProviderType.OPENAI)
        messages = [{"role": "system", "content": agent.instructions}] + history
        tools = [self.function_to_json(f, agent.provider) for f in agent.functions]
        
        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
        }
        
        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls
        
        return client.chat.completions.create(**create_params)

    def _bedrock_chat_completion(self, agent: Agent, history: List, model_override: str = None):
        client = self.get_client(ProviderType.BEDROCK_ANTHROPIC)
        system_prompt = [{"text": agent.instructions}]
        tools = [self.function_to_json(f, agent.provider) for f in agent.functions]
        
        formatted_messages = []
        for msg in history:
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"] if isinstance(msg["content"], list) else [{"text": msg["content"]}]
            }
            formatted_messages.append(formatted_msg)

        inference_config = {
            "temperature": 0,
            "maxTokens": 2048,
            "topP": 0,
        }

        # Create base parameters
        params = {
            "modelId": model_override or agent.model,
            "messages": formatted_messages,
            "system": system_prompt,
            "inferenceConfig": inference_config,
        }

        # Only add toolConfig if there are tools
        if tools:
            params["toolConfig"] = {"tools": tools}

        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                return client.converse(**params)
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    if attempt == max_retries - 1:
                        raise
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 0.1)
                    print(f"\033[93mRate limited by Bedrock, retrying in {delay:.1f} seconds...\033[0m")
                    time.sleep(delay)
                    continue
                raise

    def _azure_chat_completion(
        self,
        agent: Agent,
        history: List,
        model_override: str
    ):
        client = self.get_client(ProviderType.AZURE_OPENAI)
        messages = [{"role": "system", "content": agent.instructions}] + history
        tools = [self.function_to_json(f, agent.provider) for f in agent.functions]
        
        create_params = {
            "model": model_override or agent.azure_deployment,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
        }
        
        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls
        
        return client.chat.completions.create(**create_params)

    def get_chat_completion(self, agent: Agent, history: List, model_override: str = None):
        if agent.provider == ProviderType.OPENAI:
            return self._openai_chat_completion(agent, history, model_override)
        elif agent.provider == ProviderType.AZURE_OPENAI:
            return self._azure_chat_completion(agent, history, model_override)
        else:
            return self._bedrock_chat_completion(agent, history, model_override)

    def handle_function_result(self, result) -> Result:
        match result:
            case Result() as result:
                return result
            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    raise TypeError(e)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction]
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None)
        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            raw_result = function_map[name](**args)
            print(f'Called function {name} with args: {args} and obtained result: {raw_result}')
            print('#############################################')
            result: Result = self.handle_function_result(raw_result)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            if result.agent:
                partial_response.agent = result.agent

        return partial_response
    
    def run(
        self,
        agent: Optional[Agent] = None,
        messages: List = None,
        model_override: str = None,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        # Create default messages list if none provided
        if messages is None:
            messages = []
        
        # Create default agent if none provided
        if agent is None:
            agent = Agent(
                name="Assistant",
                provider=self.default_provider,
                instructions="You are a helpful assistant."
            )
        
        active_agent = agent
        history = copy.deepcopy(messages)
        init_len = len(messages)
        # client = self.get_client(agent.provider)

        while len(history) - init_len < max_turns and active_agent:
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                model_override=model_override
            )

            if active_agent.provider == ProviderType.OPENAI or active_agent.provider == ProviderType.AZURE_OPENAI:
                message = completion.choices[0].message
                message.sender = active_agent.name
                history.append(json.loads(message.model_dump_json()))

                if not message.tool_calls or not execute_tools:
                    break
                
                partial_response = self._handle_openai_tool_calls(message.tool_calls, active_agent.functions)
            else:
                output_message = completion.get("output", {}).get("message", {})
                stop_reason = completion.get("stopReason")
                
                output_message["sender"] = active_agent.name
                history.append(output_message)
                
                if stop_reason != "tool_use" or not execute_tools:
                    break
                    
                tools_requested = output_message.get("content", [])
                partial_response = self._handle_bedrock_tool_calls(tools_requested, active_agent.functions)

            history.extend(partial_response.messages)
            
            if partial_response.agent:
                active_agent = partial_response.agent
                if active_agent.provider == ProviderType.OPENAI:
                    message.sender = active_agent.name
                else:
                    output_message["sender"] = active_agent.name

        return Response(
            messages=history[init_len:],
            agent=active_agent,
        )

    def _handle_openai_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction]
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None)
        
        for tool_call in tool_calls:
            name = tool_call.function.name
            if name not in function_map:
                partial_response.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": f"Error: Tool {name} not found."
                })
                continue

            args = json.loads(tool_call.function.arguments)
            raw_result = function_map[name](**args)
            result = self.handle_function_result(raw_result)
            
            partial_response.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value,
            })
            
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def _handle_bedrock_tool_calls(self, tools_requested: List, functions: List[Callable]) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None)
        
        for content in tools_requested:
            if 'toolUse' not in content:
                continue
            
            tool_use = content['toolUse']
            tool_use_id = tool_use['toolUseId']
            name = tool_use['name']
            tool_input = tool_use['input']
            
            if name not in function_map:
                partial_response.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "tool_name": name,
                    "content": f"Error: Tool {name} not found."
                })
                continue

            raw_result = function_map[name](**tool_input)
            result = self.handle_function_result(raw_result)
            
            # Handle both JSON and plain string results
            result_content = result.value
            if isinstance(result.value, str):
                try:
                    result_content = json.loads(result.value)
                except json.JSONDecodeError:
                    # If it's not JSON, wrap it in a text structure
                    result_content = {"text": result.value}
            
            tool_result = {
                "toolUseId": tool_use_id,
                "content": [{"json": result_content}]
            }
            
            partial_response.messages.append({
                "role": "user",
                "content": [{"toolResult": tool_result}]
            })
            
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def _openai_function_to_json(self, func) -> dict:
        """Convert function to OpenAI format"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        signature = inspect.signature(func)
        parameters = {}
        for param in signature.parameters.values():
            param_type = type_map.get(param.annotation, "string")
            parameters[param.name] = {"type": param_type}

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect._empty
        ]

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                },
            },
        }

    def _bedrock_function_to_json(self, func) -> dict:
        """Convert function to Bedrock format"""
        type_map = {
            str: "string",
            int: "number",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        signature = inspect.signature(func)
        properties = {}
        for param in signature.parameters.values():
            param_type = type_map.get(param.annotation, "string")
            properties[param.name] = {
                "type": param_type,
                "description": f"Parameter {param.name} for function {func.__name__}"
            }

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect._empty
        ]

        return {
            "toolSpec": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
        }

    def _azure_function_to_json(self, func) -> dict:
        """Convert function to Azure OpenAI format"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        signature = inspect.signature(func)
        parameters = {}
        for param in signature.parameters.values():
            param_type = type_map.get(param.annotation, "string")
            parameters[param.name] = {"type": param_type}

        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect._empty
        ]

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                },
            },
        }
