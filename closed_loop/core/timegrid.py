"""The 15-minute time grid. Quarter 0 is 06:00 on day 0, and a day starts at 06:00.

Interface text: docs/adr/0002-interfaces.md, section 3.2.
"""

QUARTER_MIN = 15
SHIFT_Q = 32  # eight hours
DAY_Q = 96  # three shifts
HORIZON_DAYS = 90
HORIZON_Q = 8640


def _check(q: int) -> int:
    if q < 0:
        raise ValueError(f"quarter must be non-negative, got {q}")
    return q


def shift_of(q: int) -> int:
    """0 = 06:00-14:00, 1 = 14:00-22:00, 2 = 22:00-06:00."""
    return (_check(q) % DAY_Q) // SHIFT_Q


def day_of(q: int) -> int:
    return _check(q) // DAY_Q


def is_handover_next(q: int) -> bool:
    """True when quarter ``q + 1`` starts a new shift (``q`` is the last quarter of a shift)."""
    return (_check(q) + 1) % SHIFT_Q == 0
