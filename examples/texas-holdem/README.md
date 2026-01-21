# No-Limit Texas Hold'em Demo

LLM-powered poker agents vs bot opponents. Demonstrates scout-otel
context isolation. All LLM calls are recorded as Scout transcripts with
the agent's context ID by way of OTEL spans.

The example runs individual hands via the `texas-holdem` command. Each
hand is associated with a table. By default tables are located in the
[`./tables`](./tables/) directory but you can reference any table config
file using its path.

The default table is `openai.toml`, which runs a single OpenAI Agents
based agent against a single bot.

## Provide credentials

Model access requires the applicable API key or token as environment
variables. You can define one or more variables in `.env`, which is read
and applied automatically.

## Setup

These instructions assume you're running commands from the project
(scout-otel) root directory. They use `uv run ...` but you can run the
commands directly in activated virtual environments.

Install dependencies.

```bash
uv sync --package texas-holdem  # or --all-packages
```

Run the default table.

```bash
uv run texas-holdem
```

Use `scout-otel` to view logged spans and transcripts.

To view spans:

```bash
uv run scout-otel spans <run-dir>/spans
```

To view transcripts:

```bash
uv run scout-otel transcripts <run-dir>/transcripts
```

See [TODO.md](TODO.md).
