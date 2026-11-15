import pytest

from closed_loop.core import timegrid
from closed_loop.core.timegrid import (
    DAY_Q,
    HORIZON_DAYS,
    HORIZON_Q,
    QUARTER_MIN,
    SHIFT_Q,
    day_of,
    is_handover_next,
    shift_of,
)


def test_constants_are_the_interface_values():
    assert (QUARTER_MIN, SHIFT_Q, DAY_Q, HORIZON_DAYS, HORIZON_Q) == (15, 32, 96, 90, 8640)


def test_constants_agree_with_each_other():
    assert SHIFT_Q * QUARTER_MIN == 8 * 60  # eight-hour shifts
    assert DAY_Q == 3 * SHIFT_Q == 24 * 60 // QUARTER_MIN
    assert HORIZON_Q == HORIZON_DAYS * DAY_Q


@pytest.mark.parametrize(
    ("q", "shift"),
    [(0, 0), (31, 0), (32, 1), (63, 1), (64, 2), (95, 2), (96, 0), (96 + 32, 1), (8639, 2)],
)
def test_shift_of_boundaries(q, shift):
    assert shift_of(q) == shift


@pytest.mark.parametrize(
    ("q", "day"), [(0, 0), (95, 0), (96, 1), (191, 1), (192, 2), (8639, 89), (8640, 90)]
)
def test_day_of_boundaries(q, day):
    assert day_of(q) == day


def test_day_starts_at_0600():
    # quarter 0 is 06:00; the night shift 22:00-06:00 is the last third of the day
    minutes_from_0600 = 16 * 60  # 22:00
    assert shift_of(minutes_from_0600 // QUARTER_MIN) == 2
    assert shift_of(minutes_from_0600 // QUARTER_MIN - 1) == 1


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0, False), (30, False), (31, True), (32, False), (63, True), (95, True), (96, False)],
)
def test_handover_next(q, expected):
    assert is_handover_next(q) is expected


def test_handover_count_over_the_horizon():
    handovers = [q for q in range(HORIZON_Q) if is_handover_next(q)]
    assert len(handovers) == 3 * HORIZON_DAYS
    # exactly the quarters that precede a change of shift
    for q in range(HORIZON_Q - 1):
        assert is_handover_next(q) == (shift_of(q + 1) != shift_of(q))


def test_every_shift_is_32_quarters_long():
    runs = []
    start = 0
    for q in range(1, HORIZON_Q + 1):
        if q == HORIZON_Q or shift_of(q) != shift_of(q - 1):
            runs.append(q - start)
            start = q
    assert set(runs) == {SHIFT_Q}
    assert len(runs) == 3 * HORIZON_DAYS


@pytest.mark.parametrize("fn", [shift_of, day_of, is_handover_next])
def test_negative_quarter_is_refused(fn):
    with pytest.raises(ValueError, match="non-negative"):
        fn(-1)


def test_public_names():
    assert {"shift_of", "day_of", "is_handover_next"} <= set(dir(timegrid))
