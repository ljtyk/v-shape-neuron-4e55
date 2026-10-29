# AI-rater verification log

All batches pass two checks (scripts/verify_rater_batch.py): every annotated image was actually viewed
(from the agent's real transcript), and main_object names match the hidden class key for >= 75%.

## Exceptions

- **Rater A, batch 06: accepted at 74% object match (below the 75% bar).**
  - First run: 74%, rejected and redone.
  - Redo: 74%, with nearly the same mismatching names ("robot frame", "boy", "woman in racing suit", "red hot rod", "land rover").
  - Both runs: 50/50 annotated images confirmed viewed, 0 blocked reads.
  - Reading: the batch contains photos whose main subject isn't the labelled class, not invented labels. The redo is the one accepted.
- **Rater B, batch 06: accepted at 74% object match**, the same image set as the rater A exception above.
  - A third independent agent, on a different model (Opus), gives the same 74% and the same mismatching names.
  - 50/50 annotated images confirmed viewed, 0 blocked reads.
  - Not rerun: three agents on two models agree that this batch's photos often aren't mainly of the labelled class.

## Rejections (redone; the rejected output is not used)

- **Rater A, batch 14, first run.** One annotated image was never viewed. Mismatches also suggest descriptions attached to the wrong IDs ("tabby → wooden box", "Japanese spaniel → wooden chest", "chest → dog", "Pekinese → cat"). Redone with a fresh agent.
