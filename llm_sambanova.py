import llm
import llm.default_plugins.openai_models
import requests
import json
import httpx

def get_sambanova_models():
    """
    Fetch available models from SambaNova API with caching
        
    Returns:
        List of model dictionaries with 'id' key
    """
        
    # Fetch from API
    api_key = llm.get_key("", "sambanova", "LLM_SAMBANOVA_KEY")
    if not api_key:
        # Fallback to hardcoded models if no API key
        print("Warning: No SambaNova API key found, returning empty model list.")
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = httpx.get(
            "https://api.sambanova.ai/v1/models",
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
        data = response.json()
        models = [{"id": model["id"]} for model in data["data"]]
                
        return models
        
    except Exception as e:
        # Log error and fallback to hardcoded models
        print(f"Warning: Failed to fetch models from SambaNova API: {e}")
        return []


class SambaNovaChat(llm.default_plugins.openai_models.Chat):
    needs_key = "sambanova"
    key_env_var = "SAMBANOVA_KEY"

    def __str__(self):
        return "SambaNova: {}".format(self.model_id)

class SambaNovaCompletion(llm.default_plugins.openai_models.Completion):
    needs_key = "sambanova"
    key_env_var = "SAMBANOVA_KEY"

    def execute(self, prompt, stream, response, conversation=None):
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.append(prev_response.prompt.prompt)
                messages.append(prev_response.text())
        messages.append(prompt.prompt)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_key()}",
            **self.headers
        }

        data = {
            "model": self.model_name,
            "prompt": "\n".join(messages),
            "stream": stream,
            **self.build_kwargs(prompt, stream)  # modified: pass stream as argument
        }

        api_response = requests.post(
            f"{self.api_base}/completions",
            headers=headers,
            json=data,
            stream=stream
        )
        api_response.raise_for_status()

        if stream:
            for line in api_response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            line_text = line_text[6:]
                            if line_text.strip() == '[DONE]':
                                break
                            chunk = json.loads(line_text)
                            text = chunk['choices'][0].get('text')
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue
        else:
            response_json = api_response.json()
            yield response_json['choices'][0]['text']

    def __str__(self):
        return "SambaNova: {}".format(self.model_id)

@llm.hookimpl
def register_models(register):
    # Only do this if the sambanova key is set
    key = llm.get_key("", "sambanova", "LLM_SAMBANOVA_KEY")
    if not key:
        return

    models = get_sambanova_models()

    for model_definition in models:
        chat_model = SambaNovaChat(
            model_id=f"sambanova/{model_definition['id']}",
            model_name=model_definition['id'],
            api_base="https://api.sambanova.ai/v1"
        )
        register(chat_model)

        # Add completion model (without AsyncCompletion since it doesn't exist)
        completion_model = SambaNovaCompletion(
            model_id=f"sambanova/{model_definition['id']}-completion",
            model_name=model_definition['id'],
            api_base="https://api.sambanova.ai/v1"
        )
        register(completion_model)
