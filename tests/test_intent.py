import json

import pytest

from backend.agents.intent import (
    IntentAgentError,
    IntentAskResponse,
    IntentCompleteResponse,
    intent_turn,
)
from backend.schemas import DesiredDepth, KnowledgeLevel, Modality
from tests._fakes import FakeAnthropic, text_response


def _complete_goal_payload() -> dict:
    return {
        "status": "complete",
        "goal": {
            "domain": "databases",
            "sub_goal": "understand SQL joins",
            "current_level": "beginner",
            "desired_depth": "working",
            "time_budget_hours": 4.0,
            "preferred_modalities": ["visual"],
        },
    }


async def test_asking_response_parses() -> None:
    client = FakeAnthropic([text_response(
        json.dumps({"status": "asking", "question": "What's your current level?"})
    )])
    result = await intent_turn(
        client,
        conversation=[{"role": "user", "content": "help me learn SQL"}],
    )
    assert isinstance(result, IntentAskResponse)
    assert "level" in result.question.lower()


async def test_complete_response_parses_into_learning_goal() -> None:
    client = FakeAnthropic([text_response(json.dumps(_complete_goal_payload()))])
    result = await intent_turn(
        client,
        conversation=[{
            "role": "user",
            "content": "I'm a JS dev, want to learn SQL joins in a weekend",
        }],
    )
    assert isinstance(result, IntentCompleteResponse)
    assert result.goal.domain == "databases"
    assert result.goal.current_level is KnowledgeLevel.BEGINNER
    assert result.goal.desired_depth is DesiredDepth.WORKING
    assert Modality.VISUAL in result.goal.preferred_modalities


async def test_markdown_fenced_json_is_parsed() -> None:
    payload = _complete_goal_payload()
    fenced = f"```json\n{json.dumps(payload)}\n```"
    client = FakeAnthropic([text_response(fenced)])
    result = await intent_turn(client, conversation=[{"role": "user", "content": "x"}])
    assert isinstance(result, IntentCompleteResponse)


async def test_non_json_output_raises() -> None:
    client = FakeAnthropic([text_response("sure, let me ask you some questions...")])
    with pytest.raises(IntentAgentError):
        await intent_turn(client, conversation=[{"role": "user", "content": "x"}])


async def test_unknown_status_rejected() -> None:
    bad = json.dumps({"status": "wat", "question": "hmm?"})
    client = FakeAnthropic([text_response(bad)])
    with pytest.raises(IntentAgentError):
        await intent_turn(client, conversation=[{"role": "user", "content": "x"}])


async def test_malformed_goal_rejected() -> None:
    bad = json.dumps({
        "status": "complete",
        "goal": {
            "domain": "databases",
            "sub_goal": "joins",
            "current_level": "expert-ish",  # not a valid enum
            "desired_depth": "working",
            "time_budget_hours": 4.0,
            "preferred_modalities": [],
        },
    })
    client = FakeAnthropic([text_response(bad)])
    with pytest.raises(IntentAgentError):
        await intent_turn(client, conversation=[{"role": "user", "content": "x"}])


async def test_system_prompt_and_messages_passed_through() -> None:
    client = FakeAnthropic([text_response(json.dumps(_complete_goal_payload()))])
    convo = [
        {"role": "user", "content": "let's learn SQL"},
        {"role": "assistant", "content": "what's your level?"},
        {"role": "user", "content": "beginner. weekend project."},
    ]
    await intent_turn(client, conversation=convo, model="claude-sonnet-4-6")
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["messages"] == convo
    assert "Intent Agent" in call["system"][0]["text"]
