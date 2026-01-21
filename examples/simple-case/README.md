# Span to Message

This example demonstrates various traces generated from libraries and
frameworks. Each case is implemented in a `case_xxx.py` module.

Each case generates traces (span trees) and their associated Scout
transcripts according to the current scout_otel span-to-transcript
algorithm.

These cases are used to demonstrate a few things:

- Differences across frameworks in span generation
- Challenge of inferring transcript messages from spans
- The current state of "correct" transcript generation given simple
  cases

Generating correct transcripts from spans is an ongoing problem. We have
yet to see a full enough range of spans across various frameworks. It's
also not clear if our algoritm generalizes correctly in all cases. This
example serves as a test bed for testing cases and making the
appropriate adjustments to the transcript generation algorithm.

## Cases

This example is really a set of _cases_. Each case demonstrates a
"simple agent" that focuses on a narrow problem.

Most cases follow this pattern:

- Single framework (case specific)
- Run a simple model call with one tool

In general, cases generate a common transcript, though from different
underlying traces (span trees).

- **`openai`** - OpenAI Agents
- **`langchain`** - Langchain
- **`inspect`** - Inspect AI (NOTE this is not spans based but rather
  uses Inspect AI logs to generate "gold standard" Scount transcripts

## Run a case

From the project root, install the project dependencies.

```bash
uv sync --package simple-case  # or --all-packages
```

Run a case using the case name (see above). These correspond to
`src/simple_case/case_xxx.py` files.

```bash
uv run simple-case openai
```

The case is run, generating output in a runs directory, which is shown
after the case finishes.

## View case output

Most cases generate `spans` and `transcripts` directories for the run.
However, `inspect` generates `logs` (Inspect logs) and `transcripts`
(corresponding Scount transcripts). `transcripts`. `inspect` is used to
establish a baseline "gold standard" transcript that can be used to
compare other cases to.

To view spans for a run:

```bash
uv run scout-otel spans <run-dir>/spans
```

To view transcripts for a run:

```bash
uv run scout-otel spans <run-dir>/spans
```

Note that `spans` and `transcripts` commands support progressive levels
of detail by way of a `-v, --verbose` option. For example, to show more
detail for spans, run `scout-otel spans <dir> -v` --- use `-vv` and
`-vvv` (max level) for more detail.

## Highlight span content

This example supports careful reasoning about spans and how spans can be
inferred to generate transcript messages and events. It's not always
clear where transcript content is captured in span trees, or if it's
available at all.

`scout-otel` supports content highlighting with one or more
`H, --highight` values.

Here's a command that highlights content that is a strong candidate for
transcript messages.

```bash
uv run scout-otel spans <run-dir>/spans -H 'attributes/gen_ai.prompt.*.content' -H 'attributes/gen_ai.completion.*.content' -H 'attributes/gen_ai.completion.*.tool_calls.*' -H 'attributes/gen_ai.prompt.*.tool_call*' -H 'attributes/traceloop.entity.input' -H 'attributes/traceloop.entity.output'
```

Using `-H, --highlight` automatically selects `-vvv` (level 3) to show
span JSON. Matching content is highlighted in yellow.

Usage pattern for highlighting:

- Examine the span JSON to identify likely content for transcripts
- Collect applicable content key paths/patterns to highlight
- View the spans with highlights
- Use the highlights as a higher level view to the content to reason
  about possible mappings to transcript content

## Validating transcripts

Use the `spec_check` scanner to verify transcripts have the expected
message structure.

```bash
uv run scout scan examples/simple-case/scanner.py -T <run-dir>/transcripts
```

The scanner checks that each transcript contains this sequence:

1. system message
2. user message
3. assistant message with tool call
4. tool result
5. assistant response

If a transcript doesn't match, the explanation shows which positions
diverge.

**NOTE:** This scanner uses a hard-coded spec to test a case transcript.
Currently all of the cases are designed to generate transcripts that
match this spec. This hard-coding may at some point be generalized and
defined by each case.
