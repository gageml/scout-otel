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
