# Walkthrough

- API overview

- Basic case example
  - What is it
  - What it generates
  - What it shows

- CLI - views to spans and transcripts Emphasis on connecting recorded
  data to human narrative/understanding

- Scanner - check spec in msg

- Examples give us transcripts (to improve correctness and to get
  complex/interesting traces)

- Correctness
  - Hard problem given the wide end of the funnel --- tackle through
    FAFO

- More spans
  - Simulations (agents doing various things)
  - Frameworks/libraries
  - From the field (???)

## Gist

```
API
-> use with different frameworks and scenarios
-> rich set of traces
-> correctness check
-> sample scanners
```

Bulk of work is in views to data for "get head around" topics. Concern
is _correctness_. Getting something wrong has big consequences. It can
be hard to identify errors. Upstream libraries are varied and may be
incomplete.

Poker simulation to detect cheating as target. Tables with a variety of
players exercise all manner of frameworks and models. Good for
generating traces. Also, different transcript scopes (per hand, per
game, per player)

Explore using OpenLLMetry internal tests to generate spans for a quick
set of broad spans (all supported frameworks)

Emerging concept of a transcript _spec_ for tests/validation (this could
be literal Scout reference transcripts with normalized attrs)

Laters, focus on scanner validation. This is probably out of scope for
this lib it's a useful book end.
