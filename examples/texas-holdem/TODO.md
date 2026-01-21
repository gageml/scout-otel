# To do

The goal of this project is to drive interesting transcripts, in
particular transcripts that contain important but hard-to-spot signals
(e.g. cheating).

The example is currently suited to generate a variety of transcripts per
player type by way of different frameworks. Agent behavior, however, is
relatively simple.

Future work therefore falls into these categories:

- Traces from a **wider variety of frameworks** --- this tests both our
  use of various frameworks and collects a wider variety of traces)

- Traces that capture a **wider range of agent behavior** --- i.e. more
  complex and nuanced traces

## Road map

- More player types (supported frameworks)

- True agentic behavior - the current implementation is refers to
  "agents" but these are stateful abstractions used to make LLM and tool
  calls rather than independent agent

- More complete board info (e.g. list other players, position, button
  location)

- Player strategies or dispositions (aggressive, canonical, novice,
  etc.) - can be prompted or affected programmatically

- Cheat scenarios - with more independent acting agent behavior, we can
  make cheat opportunities available (peeking, pot shorting, collusion,
  etc.) and write scanners to detect different behaviors

- Multi-hand support (transcript capture over games/tournaments)
