import pytest_asyncio

from backend.persistence import (
    IN_MEMORY_DB_URL,
    create_sqlite_engine,
    init_db,
    load_learner_state,
    make_session_factory,
    save_learner_state,
)
from backend.schemas import LearnerState, TeachingMethod


@pytest_asyncio.fixture
async def session_factory():
    engine = create_sqlite_engine(IN_MEMORY_DB_URL)
    await init_db(engine)
    factory = make_session_factory(engine)
    yield factory
    await engine.dispose()


async def test_save_and_load_roundtrips(session_factory) -> None:
    original = LearnerState(
        user_id="alice",
        mastered_concepts=["joins"],
        effective_pedagogies={"joins": TeachingMethod.GAME},
    )
    async with session_factory() as session:
        await save_learner_state(original, session=session)

    async with session_factory() as session:
        loaded = await load_learner_state("alice", session=session)

    assert loaded == original
    assert loaded.effective_pedagogies["joins"] is TeachingMethod.GAME


async def test_load_missing_user_returns_none(session_factory) -> None:
    async with session_factory() as session:
        loaded = await load_learner_state("nobody", session=session)
    assert loaded is None


async def test_save_overwrites_existing_user(session_factory) -> None:
    async with session_factory() as session:
        await save_learner_state(
            LearnerState(user_id="alice", mastered_concepts=["one"]),
            session=session,
        )
        await save_learner_state(
            LearnerState(user_id="alice", mastered_concepts=["one", "two"]),
            session=session,
        )

    async with session_factory() as session:
        loaded = await load_learner_state("alice", session=session)
    assert loaded is not None
    assert loaded.mastered_concepts == ["one", "two"]


async def test_multiple_users_isolated(session_factory) -> None:
    async with session_factory() as session:
        await save_learner_state(
            LearnerState(user_id="alice", mastered_concepts=["a"]),
            session=session,
        )
        await save_learner_state(
            LearnerState(user_id="bob", mastered_concepts=["b"]),
            session=session,
        )

    async with session_factory() as session:
        alice = await load_learner_state("alice", session=session)
        bob = await load_learner_state("bob", session=session)
    assert alice.mastered_concepts == ["a"]
    assert bob.mastered_concepts == ["b"]
