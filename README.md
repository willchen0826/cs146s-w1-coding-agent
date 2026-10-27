# W1 Coding Agent

A minimal coding agent built from scratch for CS146S, Week 1. It runs a terminal chat loop against an LLM (served via the Groq API) and lets the model request file operations through a simple plain-text tool-calling protocol.

## How it works

The model doesn't use native function calling. Instead, the system prompt tells it to write tool requests as plain text:

```
tool: TOOL_NAME({"arg": "value"})
```

The agent loop parses that line out of the model's reply, runs the matching Python function, and feeds the result back into the conversation as `tool_result(...)` so the model can continue.

## Available tools

- `read_file(filename)` — read a file's full contents
- `list_files(path)` — list files/directories at a path
- `edit_file(path, old_str, new_str)` — replace the first occurrence of `old_str` with `new_str`, or create the file if `old_str` is empty

## Setup

```bash
pip install openai python-dotenv
```

Create a `.env` file in this folder with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

## Run

```bash
python3 prac.py
```

Then just chat with it, e.g. `List the files in the current directory.`

## Notes

- Built as a learning exercise — the tool-calling protocol is hand-rolled, not using the LLM API's native tool-use feature.
- `.env` is git-ignored; never commit API keys.
