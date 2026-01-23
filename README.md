# Scout OTEL

Bridge OpenLLMetry (Traceloop) instrumentation to Scout transcript
Parquet files.

## Goal

Automatic Scout transcript generation from OTEL-instrumented code.

## Usage

See [API](./docs/api.md) for API overview and usage.

## Examples

- [simple-case](./examples/simple-case/README.md)
- [texas-holdem](./examples/texas-holdem/README.md)

Each example is a workspace project. Install support for an example by
running:

```bash
uv sync --package <example>
```

Install support for all packages:

```bash
uv sync --all-package
```

Run example commands using `uv run <cmd>` or in an activated virtual
environment directly with `<cmd>`.

### Example help

Use `--help` to show example command help.

### Example runs

Each example generates runs in `./runs` by default. To specify a
different directory, use `--runs-dir` option.

A run generally conists of `spans` and `transcripts` directories. Refer
to each example for details.

### Example provider credentials (API keys/tokens)

Examples require provider credentials. Specify those using environment
variables. Variables can be defined in `.env`.

## scout-otel CLI

Use `scout-otel` to view spans and transcripts.

- **`scount-otel spans <dir>`** - show spans for a directory
- **`scout-otel transcripts <dir>`** - show Scout transcripts for a
  directory

## Related

### OpenTelemetry GenAI Working Group

**Primary**

- [Semantic conventions for generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
  --- the official spec
- [GenAI Spans conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
  --- span-specific details

**On GitHub**

- [Issue #327: Introduce semantic conventions for modern AI](https://github.com/open-telemetry/semantic-conventions/issues/327)
  --- original LLM conventions issue
- [Issue #2664: Semantic Conventions for Agentic Systems](https://github.com/open-telemetry/semantic-conventions/issues/2664)
  --- newer agent-specific proposal

**Context**

- [AI Agent Observability blog post](https://opentelemetry.io/blog/2025/ai-agent-observability/)
  --- evolving standards overview
- [OpenTelemetry for GenAI and OpenLLMetry](https://horovits.medium.com/opentelemetry-for-genai-and-the-openllmetry-project-81b9cea6a771)
  --- background on OpenLLMetry becoming part of OTEL
