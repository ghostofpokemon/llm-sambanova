import llm
from llm.default_plugins.openai_models import Chat, Completion, SharedOptions
import json
import requests

# Hardcoded models for now
def get_sambanova_models():
    return [
        {"id": "Meta-Llama-3.2-1B-Instruct"},
        {"id": "Meta-Llama-3.2-3B-Instruct"},
        {"id": "Meta-Llama-3.1-8B-Instruct"},
        {"id": "Meta-Llama-3.1-8B-Instruct-8k"},
        {"id": "Meta-Llama-3.1-70B-Instruct"},
        {"id": "Meta-Llama-3.1-70B-Instruct-8k"},
        {"id": "Meta-Llama-3.1-405B-Instruct"},
        {"id": "Meta-Llama-3.1-405B-Instruct-8k"},
    ]

class SambaNovaChat(Chat):
    needs_key = "sambanova"
    key_env_var = "SAMBANOVA_KEY"

    def __str__(self):
        return "SambaNova: {}".format(self.model_id)

class SambaNovaCompletion(Completion):
    needs_key = "sambanova"
    key_env_var = "SAMBANOVA_KEY"

    def execute(self, prompt, stream, response, conversation=None, **kwargs):
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
            **self.build_kwargs(prompt)
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
            model_id="sambanova/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            api_base="https://api.sambanova.ai/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(chat_model)

    for model_definition in models:
        completion_model = SambaNovaCompletion(
            model_id="sambanovacompletion/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            api_base="https://api.sambanova.ai/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(completion_model)

class DownloadError(Exception):
    pass
