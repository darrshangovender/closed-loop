"""P1: the bounded-stream e-process (0003 A6 replaces the Bernoulli e-process).

The type-I tests run the real class on whole streams. They are the ones that fail when the
threshold is changed from ``1/delta`` to ``1/(2 delta)``.
"""

import math

import numpy as np
import pytest

from closed_loop.stats.eprocess import BoundedEProcess
from closed_loop.stats.intervals import wilson


def _fires(stream: np.ndarray, p0: float, delta: float) -> bool:
    process = BoundedEProcess(p0, delta)
    for x in stream:
        process.update(float(x))
        if process.fired:
            return True
    return False


class TestPower:
    def test_fires_on_a_miss_rate_of_five_percent_against_one_percent(self) -> None:
        """Every one of 200 streams at 5% misses fires, and quickly (median under 400 labels)."""
        rng = np.random.default_rng(101)
        fired_at = []
        for _ in range(200):
            process = BoundedEProcess(0.01, 0.01)
            for x in rng.random(2000) < 0.05:
                process.update(float(x))
                if process.fired:
                    fired_at.append(process.fired_at)
                    break
        assert len(fired_at) == 200
        assert float(np.median(fired_at)) < 400

    def test_a_clean_stream_never_gets_near_the_threshold(self) -> None:
        process = BoundedEProcess(0.05, 0.01)
        for _ in range(5000):
            process.update(0.0)
        assert not process.fired
        assert process.e_value < 1.0  # all-zero data is evidence for H0

    def test_bounded_non_binary_values_work(self) -> None:
        """A stream of 0.3 against p0 = 0.1 (mean above p0) fires; 0.05 against 0.1 does not."""
        high = BoundedEProcess(0.1, 0.01)
        low = BoundedEProcess(0.1, 0.01)
        for _ in range(300):
            high.update(0.3)
            low.update(0.05)
        assert high.fired and not low.fired


class TestTypeOne:
    def test_false_alarm_rate_at_the_null_rate_is_at_most_delta(self) -> None:
        """2,000 streams at miss rate exactly p0 = 0.01: at most delta = 0.01 of them fire."""
        rng = np.random.default_rng(202)
        fired = sum(_fires(rng.random(300) < 0.01, 0.01, 0.01) for _ in range(2000))
        lo, hi = wilson(fired, 2000)
        assert fired / 2000 <= 0.01, f"{fired}/2000 fired (Wilson {lo:.4f}-{hi:.4f})"

    def test_the_bound_is_close_to_tight_so_a_halved_threshold_would_be_caught(self) -> None:
        """Fair coin, p0 = 0.5, delta = 0.3, 1,000 streams of 500: the rate is at most delta.

        The process is conservative (measured: 0.157 at delta = 0.3, 0.048 at 0.1), so the
        1% test above cannot see a halved threshold. At delta = 0.3 the measured rate with the
        threshold 1/(2 delta) is 0.42, over delta, so this is the type-I test that fails for it.
        """
        rng = np.random.default_rng(303)
        fired = sum(_fires(rng.random(500) < 0.5, 0.5, 0.3) for _ in range(1000))
        assert fired / 1000 <= 0.3, f"{fired}/1000 fired"
        assert fired / 1000 > 0.05  # not vacuous: the process really does fire at this level

    def test_bounded_streams_with_mean_below_p0_stay_valid(self) -> None:
        """Beta-distributed values with mean exactly p0 (a non-binary bounded stream)."""
        rng = np.random.default_rng(404)
        p0 = 0.2
        fired = 0
        for _ in range(1000):
            fired += _fires(rng.beta(0.4, 1.6, 400), p0, 0.05)  # mean 0.4/2.0 = 0.2
        assert fired / 1000 <= 0.05

    def test_a_mean_just_above_the_null_does_fire_eventually(self) -> None:
        rng = np.random.default_rng(505)
        fired = sum(_fires(rng.beta(1.0, 3.0, 4000), 0.2, 0.05) for _ in range(40))  # mean 0.25
        assert fired >= 30


class TestDependenceInflatesTheBernoulliTest:
    """Why the sequential unit is the life (0003 A6).

    Lives have ``k`` checkpoints. Per-checkpoint labels are fed one by one to the same test that is
    valid for independent items. The marginal miss rate per checkpoint is exactly ``p0``, but the
    labels inside a life are dependent, so the conditional-mean condition fails.
    """

    P0 = 0.02
    K = 10  # checkpoints per life
    LIVES = 100
    DELTA = 0.01

    def _lives(self, rng: np.random.Generator, mode: str) -> np.ndarray:
        """Per-life miss fractions' components: array (LIVES, K) of 0/1 labels."""
        if mode == "common":  # a life is either wholly missed or wholly covered
            wholly = rng.random((self.LIVES, 1)) < self.P0
            return np.repeat(wholly, self.K, axis=1).astype(np.float64)
        # "overdispersed": each life has its own miss rate, Beta with mean p0, then k labels
        rate = rng.beta(0.1, 0.1 * (1 - self.P0) / self.P0, size=(self.LIVES, 1))
        return (rng.random((self.LIVES, self.K)) < rate).astype(np.float64)

    @pytest.mark.parametrize("mode", ["common", "overdispersed"])
    def test_per_checkpoint_labels_inflate_per_life_values_do_not(self, mode: str) -> None:
        rng = np.random.default_rng(606)
        streams = 300
        per_label = 0
        per_life = 0
        marginal = []
        for _ in range(streams):
            labels = self._lives(rng, mode)
            marginal.append(labels.mean())
            per_label += _fires(labels.reshape(-1), self.P0, self.DELTA)
            per_life += _fires(labels.mean(axis=1), self.P0, self.DELTA)
        # the marginal rate per checkpoint is the null rate in both cases
        assert float(np.mean(marginal)) == pytest.approx(self.P0, abs=0.004)
        lo, _ = wilson(per_label, streams)
        assert lo > 5 * self.DELTA, f"per-label false alarms {per_label}/{streams}"
        assert per_life / streams <= self.DELTA + 0.01, f"per-life {per_life}/{streams}"

    def test_the_inflation_is_not_an_accident_of_the_seed(self) -> None:
        rates = []
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            fired = sum(
                _fires(self._lives(rng, "common").reshape(-1), self.P0, self.DELTA)
                for _ in range(100)
            )
            rates.append(fired / 100)
        assert min(rates) > 0.2  # against delta = 0.01; one burst of misses is enough to fire


class TestMechanics:
    def test_update_returns_the_current_e_value_and_counts(self) -> None:
        process = BoundedEProcess(0.1, 0.05)
        assert process.n == 0 and process.e_value == pytest.approx(1.0) and not process.fired
        value = process.update(1.0)
        assert value == process.e_value and process.n == 1 and value > 1.0
        assert process.update(0.0) < value  # a covered label gives some of it back

    def test_e_value_is_the_mean_of_the_bet_capitals(self) -> None:
        """Recompute the average of the 20 fixed-bet capitals independently."""
        p0, xs = 0.1, [1.0, 0.0, 0.0, 0.4, 1.0, 0.0]
        process = BoundedEProcess(p0)
        for x in xs:
            process.update(x)
        bets = np.geomspace(0.005, 0.99, 20) / p0
        expected = np.mean([np.prod([1.0 + b * (x - p0) for x in xs]) for b in bets])
        assert process.e_value == pytest.approx(expected, rel=1e-12)
        assert process.log_e_value == pytest.approx(math.log(expected), rel=1e-12)

    def test_fires_exactly_when_the_e_value_reaches_one_over_delta(self) -> None:
        process = BoundedEProcess(0.1, 0.2)  # threshold 5
        previous = 1.0
        while not process.fired:
            previous = process.e_value
            process.update(1.0)
        assert previous < 5.0 <= process.e_value
        assert process.fired_at == process.n

    def test_fired_latches_and_e_value_keeps_updating(self) -> None:
        process = BoundedEProcess(0.1, 0.2)
        while not process.fired:
            process.update(1.0)
        at = process.fired_at
        for _ in range(100):
            process.update(0.0)
        assert process.fired and process.fired_at == at
        assert process.e_value < 5.0  # fell back below the threshold; the flag did not

    def test_fired_at_stays_at_the_first_crossing_while_the_evidence_keeps_growing(self) -> None:
        """Mutant found by the pass: re-assigning fired_at on every update above the threshold."""
        process = BoundedEProcess(0.1, 0.2)
        while not process.fired:
            process.update(1.0)
        first = process.fired_at
        assert first == process.n
        for _ in range(30):
            process.update(1.0)  # still far above the threshold
        assert process.e_value > 5.0 and process.fired_at == first

    def test_no_overflow_on_a_very_long_run_of_misses(self) -> None:
        process = BoundedEProcess(0.01, 0.01)
        for _ in range(3000):
            process.update(1.0)
        assert math.isfinite(process.e_value) and process.log_e_value > 700
        assert process.fired

    def test_the_bet_grid_never_makes_a_factor_non_positive(self) -> None:
        process = BoundedEProcess(0.9, 0.01)
        for x in (0.0, 0.0, 0.0, 1.0, 0.0):
            assert process.update(x) > 0.0

    def test_refusals(self) -> None:
        for p0 in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="p0"):
                BoundedEProcess(p0)
        for delta in (0.0, 1.0, 2.0):
            with pytest.raises(ValueError, match="delta"):
                BoundedEProcess(0.1, delta)
        process = BoundedEProcess(0.1)
        for bad in (-0.1, 1.1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="in \\[0, 1\\]"):
                process.update(bad)
        assert process.n == 0 and process.e_value == pytest.approx(1.0)

    def test_bool_labels_are_accepted(self) -> None:
        process = BoundedEProcess(0.1)
        process.update(True)
        process.update(False)
        assert process.n == 2
