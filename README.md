# llm-sambanova

[![PyPI](https://img.shields.io/pypi/v/llm-sambanova.svg)](https://pypi.org/project/llm-sambanova/)
[![Changelog](https://img.shields.io/github/v/release/ghostofpokemon/llm-sambanova?include_prereleases&label=changelog)](https://github.com/ghostofpokemon/llm-sambanova/releases)
[![Tests](https://github.com/ghostofpokemon/llm-sambanova/workflows/Test/badge.svg)](https://github.com/ghostofpokemon/llm-sambanova/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ghostofpokemon/llm-sambanova/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for models hosted by [SambaNova](https://sambanova.ai/)

## Installation

First, [install the LLM command-line utility](https://llm.datasette.io/en/stable/setup.html).

Now install this plugin in the same environment as LLM.
```bash
llm install llm-sambanova
```

## Configuration

You will need an API key from SambaNova. You can [obtain one here](https://sambanova.ai/keys).

You can set that as an environment variable called `OPENROUTER_KEY`, or add it to the `llm` set of saved keys using:

```bash
llm keys set sambanova
```
```
Enter key: <paste key here>
```

## Usage

To list available models, run:
```bash
llm models -q sambanova
```
You should see a list that looks something like this:
```
SambaNova: sambanova/DeepSeek-R1-0528
SambaNova: sambanova/DeepSeek-R1-0528-completion
SambaNova: sambanova/DeepSeek-R1-Distill-Llama-70B
SambaNova: sambanova/DeepSeek-R1-Distill-Llama-70B-completion
SambaNova: sambanova/DeepSeek-V3-0324
SambaNova: sambanova/DeepSeek-V3-0324-completion
SambaNova: sambanova/DeepSeek-V3.1
SambaNova: sambanova/DeepSeek-V3.1-completion
SambaNova: sambanova/E5-Mistral-7B-Instruct
SambaNova: sambanova/E5-Mistral-7B-Instruct-completion
SambaNova: sambanova/Llama-3.3-Swallow-70B-Instruct-v0.4
SambaNova: sambanova/Llama-3.3-Swallow-70B-Instruct-v0.4-completion
...
```
To update the model list, reinstall the plugin:
```bash
llm install llm-sambanova -U
```

To run a prompt against a model, pass its full model ID to the `-m` option, like this:
```bash
llm -m sambanova/Meta-Llama-3.2-1B-Instruct "Five spooky names for a pet tarantula"
```
You can set a shorter alias for a model using the `llm aliases` command like so:
```bash
llm aliases set 405 sambanova/Meta-Llama-3.1-405B-Instruct
```
Now you can prompt Llama using:
```bash
cat llm_sambanova.py | llm -m 405 -s 'write some pytest tests for this'
```
## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-sambanova
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```
