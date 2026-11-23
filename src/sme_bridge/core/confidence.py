import logging

from sme_bridge.models import ConfidenceBreakdown

logger = logging.getLogger(__name__)

# Weights over the three evidence components. They are renormalised over whichever
# components are actually available, so a missing component withholds its weight rather
# than contributing a default value.
WEIGHT_RETRIEVAL = 0.35
WEIGHT_ENSEMBLE = 0.30
WEIGHT_CITATION = 0.35


def compute_confidence(
    retrieval_support: float,
    citation_coverage: float,
    ensemble_agreement: float | None,
) -> ConfidenceBreakdown:
    components = {
        "retrieval_support": retrieval_support,
        "citation_coverage": citation_coverage,
    }
    weights = {
        "retrieval_support": WEIGHT_RETRIEVAL,
        "citation_coverage": WEIGHT_CITATION,
    }
    if ensemble_agreement is not None:
        components["ensemble_agreement"] = ensemble_agreement
        weights["ensemble_agreement"] = WEIGHT_ENSEMBLE

    total_weight = sum(weights.values())
    composite = sum(components[k] * weights[k] for k in components) / total_weight
    composite = max(0.0, min(1.0, composite))

    return ConfidenceBreakdown(
        retrieval_support=retrieval_support,
        ensemble_agreement=ensemble_agreement,
        citation_coverage=citation_coverage,
        composite=composite,
        calibrated=False,
        components={
            "weights_used": {k: weights[k] / total_weight for k in components},
            "components_available": list(components),
        },
    )
