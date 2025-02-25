import boto3
import json

def simple_bedrock_chat():
    # Initialize the Bedrock client
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")
    
    # Define the model ID for Claude 3.5 Sonnet
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Create a simple conversation
    messages = [
        {
            "role": "user",
            "content": [{"text": "What are the three most interesting applications of AI in healthcare?"}]
        }
    ]
    
    # Set up the request parameters
    params = {
        "modelId": model_id,
        "messages": messages,
        "system": [{"text": "You are a helpful AI assistant specializing in healthcare technology."}],
        "inferenceConfig": {
            "temperature": 0.7,
            "maxTokens": 1000,
            "topP": 0.9,
        }
    }
    
    # Make the API call
    try:
        response = bedrock_client.converse(**params)
        
        # Extract and print the response
        output_message = response.get("output", {}).get("message", {})
        content_list = output_message.get("content", [])
        
        print("Claude's response:")
        print("-----------------")
        for content in content_list:
            if "text" in content:
                print(content["text"])
        
    except Exception as e:
        print(f"Error calling Bedrock: {str(e)}")

if __name__ == "__main__":
    simple_bedrock_chat()