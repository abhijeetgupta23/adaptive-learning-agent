from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from backend.persistence.models import LearnerStateRow
from backend.schemas.state import LearnerState


async def save_learner_state(state: LearnerState, *, session: AsyncSession) -> None:
    row = await session.get(LearnerStateRow, state.user_id)
    if row is None:
        session.add(LearnerStateRow(
            user_id=state.user_id,
            state_json=state.model_dump_json(),
        ))
    else:
        row.state_json = state.model_dump_json()
    await session.commit()


async def load_learner_state(
    user_id: str,
    *,
    session: AsyncSession,
) -> LearnerState | None:
    row = await session.get(LearnerStateRow, user_id)
    if row is None:
        return None
    return LearnerState.model_validate_json(row.state_json)
