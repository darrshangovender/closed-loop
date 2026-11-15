"""Shared, expensive objects for the shield tests: each game is solved once per session."""

from __future__ import annotations

import pytest

from closed_loop.shield.abstraction import ConveyorModel, CrusherJobModel, PumpPairModel
from closed_loop.shield.game import Winning, solve
from closed_loop.shield.table import ShieldTable, build_shield


@pytest.fixture(scope="session")
def crusher() -> CrusherJobModel:
    return CrusherJobModel()


@pytest.fixture(scope="session")
def crusher_win(crusher: CrusherJobModel) -> Winning:
    return solve(crusher)


@pytest.fixture(scope="session")
def crusher_tables(crusher: CrusherJobModel, crusher_win: Winning) -> dict[int, ShieldTable]:
    return {w: build_shield(crusher, crusher_win, w) for w in (1, 2, 3, 4)}


@pytest.fixture(scope="session")
def pumps() -> PumpPairModel:
    return PumpPairModel()


@pytest.fixture(scope="session")
def pumps_win(pumps: PumpPairModel) -> Winning:
    return solve(pumps)


@pytest.fixture(scope="session")
def pump_tables(pumps: PumpPairModel, pumps_win: Winning) -> dict[int, ShieldTable]:
    return {w: build_shield(pumps, pumps_win, w) for w in (1, 2, 3, 4)}


@pytest.fixture(scope="session")
def conveyor() -> ConveyorModel:
    return ConveyorModel()


@pytest.fixture(scope="session")
def conveyor_win(conveyor: ConveyorModel) -> Winning:
    return solve(conveyor)


@pytest.fixture(scope="session")
def conveyor_tables(conveyor: ConveyorModel, conveyor_win: Winning) -> dict[int, ShieldTable]:
    return {w: build_shield(conveyor, conveyor_win, w) for w in (1, 2, 3, 4)}
