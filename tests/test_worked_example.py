"""Unit tests for backend.agents.teaching.worked_example."""
from __future__ import annotations

import pytest

from backend.agents.teaching.worked_example import (
    WorkedExampleAgentError,
    generate_worked_example,
)
from backend.schemas.curriculum import GroundingSource, LearningUnit
from backend.schemas.teaching import TeachingMethod
from tests._fakes import FakeAnthropic, text_response


def _unit() -> LearningUnit:
    return LearningUnit(
        id="u1",
        objective="Use asyncio.create_task to run two coroutines concurrently",
        prerequisites=[],
        recommended_pedagogy=TeachingMethod.WORKED_EXAMPLE,
        grounding_sources=[
            GroundingSource(url="https://docs.python.org/3/library/asyncio.html",
                            title="asyncio")
        ],
    )


@pytest.mark.asyncio
async def test_returns_markdown_text():
    md = "# Two coroutines\n```python\nimport asyncio\n```\nThat's the gist."
    client = FakeAnthropic([text_response(md)])
    out = await generate_worked_example(_unit(), client=client)
    assert out == md


@pytest.mark.asyncio
async def test_records_token_usage_when_in_session():
    from backend.orchestration.cost import session_accumulator
    md = "# Hi\n```python\nx=1\n```"
    client = FakeAnthropic([text_response(md, in_tok=500, out_tok=200)])
    with session_accumulator() as acc:
        await generate_worked_example(_unit(), client=client)
    assert len(acc.entries) == 1
    assert acc.entries[0].agent == "worked_example"
    assert acc.entries[0].input_tokens == 500
    assert acc.entries[0].output_tokens == 200


@pytest.mark.asyncio
async def test_empty_response_raises():
    client = FakeAnthropic([text_response("   \n  \n")])
    with pytest.raises(WorkedExampleAgentError):
        await generate_worked_example(_unit(), client=client)
