import json

import pytest

from backend.agents.assessment import AssessmentAgentError, assess_understanding
from backend.schemas import (
    AssessmentResult,
    GroundingSource,
    LearningUnit,
    TeachingMethod,
    Verdict,
)
from tests._fakes import FakeAnthropic, text_response


def _unit() -> LearningUnit:
    return LearningUnit(
        id="u1",
        objective="Distinguish INNER vs LEFT joins",
        recommended_pedagogy=TeachingMethod.VISUAL,
        grounding_sources=[GroundingSource(url="https://x.com", title="t")],
    )


def _transcript() -> list[dict]:
    return [
        {"role": "assistant", "content": "Here's how INNER and LEFT differ..."},
        {"role": "user", "content": "So LEFT keeps unmatched rows from the left table?"},
        {"role": "assistant", "content": "Exactly. Can you give me a case?"},
        {
            "role": "user",
            "content": "Customers without orders — LEFT keeps them, INNER drops them.",
        },
    ]


async def test_pass_verdict_parses() -> None:
    payload = {
        "verdict": "pass",
        "diagnostic_notes": "Learner produced transfer example unprompted.",
        "confidence": 0.9,
    }
    client = FakeAnthropic([text_response(json.dumps(payload))])
    result = await assess_understanding(
        _unit(), TeachingMethod.VISUAL, _transcript(), client=client,
    )
    assert isinstance(result, AssessmentResult)
    assert result.verdict is Verdict.PASS
    assert result.unit_id == "u1"
    assert result.confidence == 0.9


async def test_needs_different_approach_verdict_parses() -> None:
    payload = {
        "verdict": "needs_different_approach",
        "diagnostic_notes": "Learner confused about matching semantics.",
        "confidence": 0.75,
    }
    client = FakeAnthropic([text_response(json.dumps(payload))])
    result = await assess_understanding(
        _unit(), TeachingMethod.VISUAL, _transcript(), client=client,
    )
    assert result.verdict is Verdict.NEEDS_DIFFERENT_APPROACH


async def test_rejects_bad_confidence() -> None:
    payload = {
        "verdict": "pass",
        "diagnostic_notes": "ok",
        "confidence": 1.4,
    }
    client = FakeAnthropic([text_response(json.dumps(payload))])
    with pytest.raises(AssessmentAgentError):
        await assess_understanding(
            _unit(), TeachingMethod.VISUAL, _transcript(), client=client,
        )


async def test_rejects_unknown_verdict() -> None:
    payload = {
        "verdict": "kinda_okay",
        "diagnostic_notes": "ok",
        "confidence": 0.5,
    }
    client = FakeAnthropic([text_response(json.dumps(payload))])
    with pytest.raises(AssessmentAgentError):
        await assess_understanding(
            _unit(), TeachingMethod.VISUAL, _transcript(), client=client,
        )


async def test_rejects_bad_json() -> None:
    client = FakeAnthropic([text_response("not json at all")])
    with pytest.raises(AssessmentAgentError):
        await assess_understanding(
            _unit(), TeachingMethod.VISUAL, _transcript(), client=client,
        )


async def test_agent_stamps_unit_id_even_when_missing() -> None:
    payload = {
        "verdict": "pass",
        "diagnostic_notes": "ok",
        "confidence": 0.85,
        # unit_id absent
    }
    client = FakeAnthropic([text_response(json.dumps(payload))])
    result = await assess_understanding(
        _unit(), TeachingMethod.VISUAL, _transcript(), client=client,
    )
    assert result.unit_id == "u1"


async def test_transcript_included_in_user_message() -> None:
    payload = {"verdict": "pass", "diagnostic_notes": "ok", "confidence": 0.9}
    client = FakeAnthropic([text_response(json.dumps(payload))])
    await assess_understanding(_unit(), TeachingMethod.GAME, _transcript(), client=client)
    call = client.messages.calls[0]
    user_content = call["messages"][0]["content"]
    assert "Distinguish INNER vs LEFT joins" in user_content
    assert "Method used: game" in user_content
    assert "Customers without orders" in user_content
