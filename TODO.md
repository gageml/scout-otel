# To do

## Design topics / discussion

- [ ] "Correctness" strategy
  - Spans -> transcript events -> Spans (prove lossless)
  - Traceloop tests -> span breadth (speculative)
  - Transcript "specs" -> verify transcript intergrity (via scanners)
  - Trace generation (e.g. simulators)
  - Adoption and feedback

## Near term issues

- [ ] Code review / cleanup (esp span -> transcript algo)

## Longer term

- [ ] External systems integration (related to Deeper Scount integration
      topic)
- [ ] Performance topics (memory, performance)

## Speculative

- [ ] Span start/end transcript events (lossless span info) (Update: not
      a clear need for lossless repro of original spans but it's
      feasible given the spant event structures)

## Enhancements to Examples

- [Texas holdem](./examples/texas-holdem/TODO.md)
- [Simple case](./examples/simple-case/TODO.md)
