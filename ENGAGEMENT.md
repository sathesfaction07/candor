# Engagement readout: Candor

How this project would be scoped, delivered and handed over if it were a customer engagement.
The customer is a composite: an operations team whose procedures live in long SOP documents, not
a named company. Every claim below points at a test in this repository; anything not measured is
labelled as such.

## The customer and what they asked for

New analysts spend their first months asking senior colleagues where a procedure is written down.
The senior colleagues are the bottleneck, and their answers never make it back into the SOPs.

**The ask:** "A chatbot over our SOPs."

## What discovery found

| Observed | What it meant |
|:--|:--|
| A wrong answer about a procedure costs more than no answer | The assistant has to be allowed to refuse, and refusing must be the designed path, not an error |
| An earlier internal attempt scored its own confidence from response length and formatting | Confidence has to come from evidence (retrieval support, citation coverage, agreement between models), and the breakdown has to be visible to a reviewer |
| The real knowledge is in experts' heads, not the documents | Every refusal should become a precise question for an expert, and the expert's answer should make the next identical question succeed |
| The customer will ask "how accurate is it?" | A threshold on an evidence score is not a probability. Claiming "90% confident" requires calibration on labelled outcomes, which do not exist on day one |

**What was deliberately not built:** a polished chat UI. The CLI and a small API carry the loop;
the customer's existing chat tool is where a UI belongs, as an integration.

## What was delivered

- Ingestion of PDF (with page numbers), text and markdown, sentence-aware chunking.
- Three retrieval backends behind one interface: local keyword search with no dependencies,
  ChromaDB, and a managed knowledge base.
- A confidence breakdown on every answer, with "not measured" modelled as `None` rather than 0.
- An abstention gate, an expert-question generator, a knowledge-gap ledger, and the loop that
  folds an expert's answer back into the index.
- Calibration machinery (reliability bins, expected calibration error, Brier score) ready for the
  day labelled outcomes exist.

## Proof that it works

| Claim | Evidence | Reproduce |
|:--|:--|:--|
| The loop closes: abstain, ask the expert, answer next time | End-to-end test of the whole path | `pytest tests/e2e/test_workflow.py` |
| Unmeasured agreement is never treated as disagreement | Test asserts it scores strictly higher than measured disagreement | `pytest tests/unit/test_confidence.py` |
| A probability-style threshold on a raw score would refuse almost everything | The end-to-end test shows a strict gate still refusing after the expert's answer, and the shipped evidence gate answering | `pytest tests/e2e/test_workflow.py` |
| The classifier refuses to train on refusals alone | A ledger of gaps is one class; fitting raises and says both classes are needed | `pytest tests/unit/test_cli.py` |

The whole suite runs with no cloud credentials: `pytest`.

## What is not yet proven, and how the engagement would prove it

**Answer accuracy on the customer's own questions has not been measured.** That is the first task
of phase 1 below, not a footnote: build a set of real questions with expert-approved answers, then
report accuracy among the questions answered and the rate of abstention, and pick the threshold
from that trade-off together with the customer.

## Rollout plan

| Phase | What changes | Gate to move on |
|:--|:--|:--|
| 1. Shadow | Experts receive every question with Candor's answer or refusal attached; they answer as usual and mark Candor right or wrong | A few hundred labelled outcomes; calibration error measured; threshold agreed with the customer |
| 2. Assist | New analysts see Candor's answer with its evidence breakdown; refusals go to experts as precise questions | Expert time per question falls; no wrong answer above the agreed threshold goes unflagged |
| 3. Self-serve | Answers above the calibrated threshold are shown without an expert in the loop | Accuracy above threshold holds on a fresh sample each month |

## Risks and open items

- **Labelled outcomes are the real dependency.** Without accepted or rejected answers, the gate
  stays uncalibrated. Collecting them is a workflow change for the experts and has to be agreed
  in phase 1.
- **Citation coverage is self-fulfilling on the extractive path.** It becomes a real
  hallucination check only when a generative model writes the answer; the composite already
  reweights over the signals that were measured.

## Feedback for the platform team

"Learn from what the assistant could not answer" is a common request, and a ledger of refusals
alone cannot train anything: it holds negatives only. A managed answer-feedback capture (accepted
or rejected, tied to the retrieval trace) would turn a frequent customer request into something
that works by default.
