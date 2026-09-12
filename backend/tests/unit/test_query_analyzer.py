"""Query analysis and adaptive retrieval profiles (spec sections 14, 16, 76).

The classifier's job is to be *explainable and stable*, not clever: a rule that
fires unpredictably would make every adaptive retrieval result impossible to
attribute.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from core.retrieval.adaptive import (
    DEFAULT_PROFILES,
    RetrievalProfile,
    RuleBasedRetrievalPolicy,
)
from core.retrieval.base import RetrievalQuery
from core.routing.query_analyzer import RuleBasedQueryAnalyzer
from core.types import QueryType

pytestmark = pytest.mark.unit

analyzer = RuleBasedQueryAnalyzer()


# --- classification -------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("why do coral reefs bleach", QueryType.CONCEPTUAL),
        ("explain how photosynthesis works", QueryType.CONCEPTUAL),
        ("ERR-5521", QueryType.EXACT_ENTITY),
        ("what does GISTEMP measure", QueryType.EXACT_ENTITY),
        ("version 4.2.1 requirements", QueryType.EXACT_ENTITY),
        ('find the "exact phrase" in the report', QueryType.EXACT_ENTITY),
        ("how many degrees of warming", QueryType.NUMERIC),
        ("what is the average emissions rate", QueryType.NUMERIC),
        ("what changed since 2020", QueryType.TEMPORAL),
        ("compare dense versus keyword retrieval", QueryType.COMPARATIVE),
        ("explain bleaching and then describe recovery", QueryType.MULTI_HOP),
        ("the endpoint returned a timeout", QueryType.TECHNICAL),
        ("reefs", QueryType.AMBIGUOUS),
    ],
)
def test_queries_are_classified(query: str, expected: QueryType) -> None:
    assert analyzer.analyze(query).query_type is expected


def test_classification_is_deterministic() -> None:
    """An analyzer that varied between runs would make every adaptive result
    irreproducible."""
    first = analyzer.analyze("compare ERR-5521 and ERR-9310 since 2020")
    second = analyzer.analyze("compare ERR-5521 and ERR-9310 since 2020")

    assert first == second


def test_an_empty_query_is_ambiguous_not_an_error() -> None:
    analysis = analyzer.analyze("   ")

    assert analysis.query_type is QueryType.AMBIGUOUS
    assert analysis.entities == ()


# --- entities and signals -------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ERR-5521 failed", "ERR-5521"),
        ("call get_user_by_id first", "get_user_by_id"),
        ("upgrade to 4.2.1", "4.2.1"),
        ("the NASA dataset", "NASA"),
        ('search for "exact words" now', "exact words"),
    ],
)
def test_exact_match_tokens_are_extracted(query: str, expected: str) -> None:
    """These are the tokens where a dense embedding can land on a topically
    similar passage naming the wrong thing."""
    analysis = analyzer.analyze(query)

    assert expected in analysis.entities
    assert analysis.requires_exact_match is True


def test_an_acronym_inside_an_identifier_is_not_a_separate_entity() -> None:
    """Reporting both `ERR` and `ERR-5521` would pad the explanation without
    adding information."""
    analysis = analyzer.analyze("ERR-5521")

    assert analysis.entities == ("ERR-5521",)


def test_a_bare_year_is_temporal_not_an_identifier() -> None:
    analysis = analyzer.analyze("what changed since 2020")

    assert analysis.entities == ()
    assert "temporal_marker" in analysis.signals


def test_numeric_phrasing_outranks_a_bare_figure() -> None:
    """ "How many ERR-5521 events" is a counting task, not a lookup."""
    analysis = analyzer.analyze("how many ERR-5521 events occurred")

    assert analysis.query_type is QueryType.NUMERIC
    assert analysis.requires_numeric_reasoning is True
    # The identifier is still recorded — the retriever may still want it.
    assert "ERR-5521" in analysis.entities


def test_signals_name_the_rules_that_fired() -> None:
    """A classification that cannot say what triggered it explains nothing."""
    analysis = analyzer.analyze("compare ERR-5521 and ERR-9310")

    assert "identifier_token" in analysis.signals
    assert "comparative_phrase" in analysis.signals
    assert f"type:{analysis.query_type}" in analysis.signals


def test_difficulty_rises_with_complexity() -> None:
    simple = analyzer.analyze("why do reefs bleach")
    complex_query = analyzer.analyze(
        "compare bleaching rates and then explain recovery across both regions"
    )

    assert complex_query.difficulty > simple.difficulty


def test_a_very_short_query_is_marked_ambiguous() -> None:
    assert analyzer.analyze("reefs").ambiguity >= 0.8


# --- retrieval profiles ---------------------------------------------------


def test_every_query_type_has_a_profile() -> None:
    """A missing type would silently fall back and make the policy's behaviour
    depend on an omission."""
    assert set(DEFAULT_PROFILES) == set(QueryType)


@pytest.mark.parametrize(
    ("query", "leans"),
    [
        ("why do coral reefs bleach", "semantic"),
        ("ERR-5521", "keyword"),
    ],
)
def test_the_profile_leans_the_way_the_spec_describes(query: str, leans: str) -> None:
    """Spec section 16: conceptual -> semantic-heavy, exact entity ->
    keyword-heavy. These are heuristics to be tested, not findings."""
    profile = RuleBasedRetrievalPolicy().profile_for(analyzer.analyze(query))

    if leans == "semantic":
        assert profile.semantic_weight > profile.keyword_weight
    else:
        assert profile.keyword_weight > profile.semantic_weight


def test_multi_hop_widens_the_candidate_pool() -> None:
    policy = RuleBasedRetrievalPolicy()
    simple = policy.profile_for(analyzer.analyze("why do reefs bleach"))
    multihop = policy.profile_for(analyzer.analyze("explain bleaching and then describe recovery"))

    assert multihop.fetch_multiplier > simple.fetch_multiplier


def test_the_plan_widens_top_k_but_not_the_query() -> None:
    """Weights are applied at fusion, so the caller still gets what it asked
    for; only the pool each arm draws from grows."""
    policy = RuleBasedRetrievalPolicy()
    query = RetrievalQuery(text="why do reefs bleach", project_id=uuid4(), top_k=10)

    planned = policy.plan(analyzer.analyze(query.text), query)

    assert planned.top_k == 30
    assert planned.text == query.text
    assert planned.project_id == query.project_id


def test_profiles_are_configurable() -> None:
    """Spec section 76 requires the weights be configurable, because whether
    they help is an open question."""
    policy = RuleBasedRetrievalPolicy(
        profiles={QueryType.CONCEPTUAL: RetrievalProfile(0.1, 0.9, 1, "inverted")}
    )

    profile = policy.profile_for(analyzer.analyze("why do reefs bleach"))

    assert profile.keyword_weight == pytest.approx(0.9)
    assert profile.reason == "inverted"


def test_an_unmapped_type_falls_back_rather_than_failing() -> None:
    policy = RuleBasedRetrievalPolicy(profiles={})

    profile = policy.profile_for(analyzer.analyze("why do reefs bleach"))

    assert profile.semantic_weight == pytest.approx(0.5)
    assert profile.keyword_weight == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("semantic", "keyword", "multiplier"),
    [(-0.1, 0.5, 3), (0.0, 0.0, 3), (0.5, 0.5, 0)],
)
def test_an_invalid_profile_is_rejected(semantic: float, keyword: float, multiplier: int) -> None:
    with pytest.raises(ValueError):
        RetrievalProfile(semantic, keyword, multiplier)


def test_weights_are_ordered_semantic_first() -> None:
    """Fusion receives [dense, keyword]; a swapped order would invert the
    policy silently."""
    profile = RetrievalProfile(0.75, 0.25)

    assert profile.weights == [0.75, 0.25]


def test_escalation_triggers_only_on_thin_results() -> None:
    policy = RuleBasedRetrievalPolicy(escalate_below=3)

    assert policy.should_escalate([]) is True
    assert policy.should_escalate([object()] * 5) is False  # type: ignore[list-item]
