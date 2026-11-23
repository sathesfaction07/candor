import logging

from sme_bridge.core.ledger import GapLedger
from sme_bridge.models import ConfidenceBreakdown

logger = logging.getLogger(__name__)


def build_training_examples(
    ledger: GapLedger,
) -> list[tuple[ConfidenceBreakdown, bool]]:
    examples = []

    # Uses the ledger interface rather than reaching for private attributes: sniffing for
    # _cache/gaps silently returned [] for any other implementation, and an empty training
    # set produces a classifier that looks fitted while having learned nothing.
    all_gaps = ledger.list_all()

    for gap in all_gaps:
        if gap.status == "open":
            continue

        if gap.status == "answered" and gap.sme_answer:
            is_correct = False
            examples.append((gap.confidence, is_correct))
            logger.debug(
                f"Gap {gap.gap_id} was answered by SME (system was unsure); labeled as incorrect"
            )

    return examples
