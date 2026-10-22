# tau

A minimal, extensible AI coding agent for the terminal, written in Python and
modeled after [pi](https://pi.dev).

tau replicates pi's core architecture — the agent loop, streaming events,
session persistence, tool execution, and context-file discovery — in roughly
3,500 lines of dependency-free Python across 21 modules (standard library only;
HTTP and SSE are `urllib` plus a line loop). The test suite is about 2,000
lines over 9 files.

## Install

Requires Python 3.12+. There are no runtime dependencies — installation pulls
nothing beyond the standard library.

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
/help /model /tools /system /plain /thinking /new /sessions /resume /compact /quit
```

Ctrl-C aborts the active run (steering between turns). Ctrl-D exits.

### Terminal output

Assistant text is rendered as lightweight Markdown (headings, bullets, inline
code, fenced code blocks), tool calls are shown as compact `╭─` / `╰─` blocks,
and `edit` results include a colorized unified diff. Reasoning streams behind a
dim gutter and can be hidden with `/thinking`; `/plain` disables Markdown
rendering.

Styling follows the terminal's ANSI palette and is disabled automatically for
non-TTY output, `NO_COLOR`, or `TERM=dumb`, so `tau -p` stays scriptable.
`FORCE_COLOR=1` overrides detection.

In print mode the final response goes to stdout, provider failures go to stderr
prefixed with `tau:`, and the exit status is non-zero on failure.

## How it works

A session is tau's record of a conversation. The pieces mirror pi:

- **Agent loop** — a submitted message is added to the active branch; tau
  builds a model request from the system prompt, active branch, and tool
  definitions; the provider streams an assistant response containing text
  and/or tool calls; tau executes each tool call and records the results.
  Tool results trigger another turn; otherwise the run ends.
- **Provider** — one `chat/completions` request per turn, streamed over
  `urllib` and parsed line by line as SSE, so there is no client SDK to depend
  on. Text, reasoning, and tool-call deltas accumulate into the finished
  message; `transport=` swaps the HTTP layer out (the tests inject a stand-in).
  Transient failures (429, 5xx, dropped streams, server overload) are retried up
  to three times with 1/2/4s backoff; permanent ones (bad key, bad request,
  content filtered) fail immediately. Ctrl-C aborts between chunks.
- **Events** — subscribe with `session.subscribe(callback)` to observe
  message start/end, text deltas, tool execution, and run lifecycle.
- **Sessions** — append-only JSONL under `~/.tau/sessions/<date>/`. Each
  entry has an id and a parent, so history forms a tree whose active branch
  supplies model context (pointer lines record the active leaf).
- **System prompt** — base instructions plus the nearest `AGENTS.md`
  discovered from the working directory upward.
- **Tools** — `read`, `grep`, `glob`, `bash`, `edit`, `write`, `task`, bound
  to the working directory. Tools run with the permissions of the tau process.
- **Search** — `grep` and `glob` walk the tree in Python (no ripgrep
  dependency), prune `.git`, `node_modules`, `.venv`, `__pycache__` and build
  output, skip binaries, and cap their output, so a broad query cannot flood
  the transcript. `grep` returns ripgrep-shaped `path:line: text` lines and
  supports `glob` filters, context lines, and content/count/file modes.
- **Sub-agents** — a `task` call runs a separate `AgentSession` with a fresh
  history and a read-only tool set (`read`, `grep`, `glob`, `bash`). Only its
  final report returns to the parent, so exploration does not accumulate in
  the main context. Sub-agents share the parent's provider (and therefore
  `/model`) and its abort flag.
- **Compaction** — when the active branch passes a token threshold, the oldest
  turns are summarized into one message and the most recent turns are kept
  verbatim. Cuts land on user-message boundaries, so tool calls and their
  results are never split apart. Compaction only moves the active leaf; the
  original entries stay in the JSONL file.

## Long sessions

Compaction kicks in automatically, or on demand with `/compact`. The defaults
suit a ~200k-token context window:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TAU_COMPACT` | `1` | Set to `0` to disable automatic compaction |
| `TAU_COMPACT_THRESHOLD` | `100000` | Estimated tokens in the branch that trigger it |
| `TAU_COMPACT_KEEP_TURNS` | `4` | Recent user turns kept verbatim |
| `TAU_COMPACT_INPUT_TOKENS` | `30000` | Cap per summarization request (longer histories are folded in batches) |

Lower the threshold when running a smaller-context model. Compaction is
conservative: if the summary would not be smaller than the turns it replaces
(short branches, or one huge tool result), it is skipped and the verbatim
history is kept. Failures are non-fatal too — the summary request fails,
history is left untouched, and the run continues. Neither case is retried
within the same run.

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
├── agent.py       # AgentSession: the agent loop
├── cli.py         # interactive REPL + print mode
├── compaction.py  # summarize old turns when the branch gets long
├── events.py      # session lifecycle events
├── messages.py    # message types (user, assistant, toolResult)
├── prompts.py     # system prompt + AGENTS.md discovery
├── provider.py    # streaming OpenAI-compatible runtime, urllib + SSE
├── session.py     # JSONL session persistence (tree-shaped)
└── tools/         # read, grep, glob, bash, edit, write, task
```

## Not implemented (ideas for extension)

Branching UI over the session tree, parallel tool execution, diagnostics
after edits, skills, prompt templates, extensions, MCP support, multi-line
editor input.

## License

MIT
