# Governance: stop conditions and handoff

This is the most portable thing in the repository. It is not specific to motifs.

## The four-step protocol

When a preregistered stop condition fires:

1. **Stop on trigger.** Do not start the next task. "The next step is cheap",
   "the failure is obviously bounded" and "let me characterise it first" are not
   exceptions.
2. **Write the record.** The triggering condition verbatim, the section that
   defines it, a timestamp, the evidence in hand at trigger time, and what is
   still undetermined.
3. **Report at the top.** The halt status appears in the first line of output,
   before any results. A governance disclosure at the end of a long report is not
   a disclosure.
4. **Wait for a decision.** Bounded characterisation of the failure is allowed
   while halted, but characterising is not permission to continue; conclusions
   produced under a halt are labelled as such.

**Timestamp check.** No artifact of a *subsequent* task may exist between the halt
and the human decision. If one does, the halt failed regardless of that task's
result.

## The legality test for changing a frozen rule

> If the same change could have been written without knowing the result, it is
> legitimate.

Tolerance corrections and definition clarifications usually pass. Crossing a stop
condition does not.

## Two precedents, pointing opposite ways

Both are worth keeping, because the contrast is the lesson.

**The crossing.** A stop condition fired; the executor judged the failure bounded
and continued through six further phases; the failing status surfaced only at the
end. The important part is what came next: this was *not* filed as simple
executor negligence. The specification said "stop" and **never defined who decides
whether to continue**. The gap in the rule was the thing to fix, and defining this
handoff is that fix.

**The halt.** Later, a stop condition fired on a result that would have revised a
published conclusion. The run halted and asked. That is the only reason the line
did not end up reporting that a second cell line had overturned an earlier
finding — the authorised follow-up showed the apparent revision came from the
choice of comparator, not from the data.

## A third, smaller precedent: conflicting instructions

During closeout, one instruction required adding a comment above a specific
source line while a standing instruction required that file to remain unmodified.
The executor did **not** pick one. It reported the conflict and waited.

Separately, having started to insert the comment, it found that **21 documents
cited that file by line number**, several of them sealed records — so inserting a
line would silently invalidate all of them. It reverted and attached the comment
to the existing line instead, preserving the numbering.

Neither judgement was heroic. Both are what the protocol looks like when it is
working: surface the conflict, and prefer the change that does not silently break
someone else's reference.

## Grades of selection provenance

When a tuning choice is recorded, record *how* it was made:

| grade | meaning |
|---|---|
| `PROGRAMMATIC_RULE` | an executable rule that selects the value without seeing results |
| `EYEBALLED` | a human chose from a comparison; the criterion was never formalised |
| `MODEL_SELECTED_NO_TRANSCRIPT` | an automated agent chose, no criterion retained, and the information visible to it cannot be reconstructed |
| `PROVENANCE_UNRECOVERABLE` | nobody can recall and nothing was retained |

`MODEL_SELECTED_NO_TRANSCRIPT` is **stricter** than `EYEBALLED`, which is
counter-intuitive and deliberate: a human selector can at least testify afterwards
to what they looked at. An agent's conditioning set cannot be reconstructed, and in
particular it cannot be ruled out that it had already seen downstream information.
Later questioning cannot repair that.

**A fabricated rationale is worse than an admitted gap**, because a future reader
will treat it as a basis. When the criterion is gone, record that it is gone — then
test whether the conclusion depends on the choice at all. That test, not the
recovered reason, is what removes the risk.
