# tau

A minimal, extensible AI coding agent for the terminal, written in Python and
modeled after [pi](https://pi.dev).

tau replicates pi's core architecture — the agent loop, streaming events,
session persistence, tool execution, and context-file discovery — in a few
hundred lines of dependency-light Python.

## Install

```bash
cd tau
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set credentials (any OpenAI-compatible endpoint works):

```bash
export OPENAI_API_KEY=sk-...        # or TAU_API_KEY
export TAU_MODEL=gpt-4.1            # optional
export TAU_BASE_URL=...             # optional, for OpenRouter/ollama/etc.
```

### Providers

Any OpenAI-compatible endpoint works. Configure with `TAU_BASE_URL`,
`TAU_API_KEY`, and `TAU_MODEL` (or the `--base-url` / `--api-key` / `-m`
flags, which take precedence).

**Z.ai (GLM models)** — [z.ai](https://z.ai) has two OpenAI-compatible
endpoints with **separate billing**. Pick the one matching where your
balance lives, or you will get "Insufficient balance" errors:

- **GLM Coding Plan subscription** → `https://api.z.ai/api/coding/paas/v4/`
- **Pay-as-you-go platform balance** → `https://api.z.ai/api/paas/v4/`

```bash
# GLM Coding Plan
export TAU_BASE_URL=https://api.z.ai/api/coding/paas/v4/
export TAU_API_KEY=your-zai-key
export TAU_MODEL=glm-4.6
```

```bash
# Pay-as-you-go
export TAU_BASE_URL=https://api.z.ai/api/paas/v4/
export TAU_API_KEY=your-zai-key
export TAU_MODEL=glm-4.6
```

**DeepSeek** — DeepSeek's API is OpenAI-compatible. Current model names are
`deepseek-flash` and `deepseek-v4-pro` (older deployments may still accept
`deepseek-chat` / `deepseek-reasoner`).

```bash
export TAU_BASE_URL=https://api.deepseek.com
export TAU_API_KEY=your-deepseek-key      # DEEPSEEK_API_KEY also works for this URL
export TAU_MODEL=deepseek-flash
tau
```

DeepSeek enables thinking mode by default. tau streams `reasoning_content`
to the terminal and persists it with the assistant message; it is sent back
on later requests, which DeepSeek requires whenever tools are present. For
DeepSeek-compatible proxies whose URL does not contain `deepseek`, set
`TAU_PASS_REASONING_CONTENT=1` to enable this explicitly.

Other examples: OpenRouter (`https://openrouter.ai/api/v1`),
Ollama (`http://localhost:11434/v1`, key `ollama`).
Switch models at runtime with `/model <name>`.

## Use

```bash
tau                     # interactive REPL in the current directory
tau "fix the failing tests"   # start with an initial prompt
tau -p "what does this repo do?"   # print mode: run once, write final response
tau --resume 2          # resume a previous session
```

Slash commands inside the REPL:

```
/help /model /tools /system /new /sessions /resume /usage /quit
```

Ctrl-C aborts the active run (steering between turns). Ctrl-D exits.

## How it works

A session is tau's record of a conversation. The pieces mirror pi:

- **Agent loop** — a submitted message is added to the active branch; tau
  builds a model request from the system prompt, active branch, and tool
  definitions; the provider streams an assistant response containing text
  and/or tool calls; tau executes each tool call and records the results.
  Tool results trigger another turn; otherwise the run ends.
- **Events** — subscribe with `session.subscribe(callback)` to observe
  message start/end, text deltas, tool execution, and run lifecycle.
- **Sessions** — append-only JSONL under `~/.tau/sessions/<date>/`. Each
  entry has an id and a parent, so history forms a tree whose active branch
  supplies model context (pointer lines record the active leaf).
- **System prompt** — base instructions plus the nearest `AGENTS.md`
  discovered from the working directory upward.
- **Tools** — `read`, `bash`, `edit`, `write`, bound to the working
  directory. Tools run with the permissions of the tau process.

## SDK usage

Like pi's `createAgentSession()`:

```python
from tau import create_agent_session

session = create_agent_session()
unsubscribe = session.subscribe(lambda event: print(getattr(event, "delta", ""), end=""))
session.prompt("What files are in the current directory?")
unsubscribe()
print(session.last_assistant_text())
session.dispose()
```

Custom tools implement the `Tool` interface:

```python
from tau.tools import Tool, ToolResult

class GitStatusTool(Tool):
    name = "git_status"
    description = "Show the working tree status."
    parameters = {"type": "object", "properties": {}}

    def execute(self, **_):
        import subprocess
        out = subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout
        return ToolResult(output=out or "clean")
```

Pass `tools=[GitStatusTool(), ...]` to `create_agent_session`.

## Layout

```
src/tau/
├── agent.py     # AgentSession: the agent loop
├── cli.py       # interactive REPL + print mode
├── events.py    # session lifecycle events
├── messages.py  # message types (user, assistant, toolResult)
├── prompts.py   # system prompt + AGENTS.md discovery
├── provider.py  # streaming OpenAI-compatible runtime (DeepSeek/GLM ready)
├── session.py   # JSONL session persistence (tree-shaped)
└── tools/       # read, bash, edit, write
```

## Not implemented (ideas for extension)

Branching UI over the session tree, compaction of old messages, skills,
prompt templates, extensions, MCP support, multi-line editor input.

## License

MIT
