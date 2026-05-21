"""Tests for src/sampling.py — ParameterSpec and sample_scenario_parameters."""

import math

import numpy as np
import pytest

from src.sampling import ParameterSpec, ScenarioPriors, sample_scenario_parameters


class TestParameterSpecFixed:
    def test_always_returns_mean(self, rng):
        spec = ParameterSpec(dist="fixed", mean=7.5)
        assert spec.sample(rng) == 7.5

    def test_deterministic_regardless_of_rng(self):
        spec = ParameterSpec(dist="fixed", mean=3.0)
        r1 = np.random.default_rng(0)
        r2 = np.random.default_rng(99)
        assert spec.sample(r1) == spec.sample(r2) == 3.0


class TestParameterSpecLognormal:
    def test_positive_draws(self, rng):
        spec = ParameterSpec(dist="lognormal", mean=2.0, cv=0.2)
        draws = [spec.sample(rng) for _ in range(500)]
        assert all(d > 0 for d in draws)

    def test_mean_approximately_correct(self):
        rng = np.random.default_rng(0)
        spec = ParameterSpec(dist="lognormal", mean=5.0, cv=0.1)
        draws = [spec.sample(rng) for _ in range(5000)]
        assert abs(sum(draws) / len(draws) - 5.0) < 0.3  # within 6% of target mean


class TestParameterSpecBeta:
    def test_in_unit_interval(self, rng):
        spec = ParameterSpec(dist="beta", mean=0.3, std=0.05)
        draws = [spec.sample(rng) for _ in range(500)]
        assert all(0.0 < d < 1.0 for d in draws)

    def test_std_too_large_raises(self, rng):
        # std must be < sqrt(mean * (1 - mean)) = sqrt(0.1 * 0.9) ≈ 0.3
        spec = ParameterSpec(dist="beta", mean=0.1, std=0.5)
        with pytest.raises(ValueError, match="too large"):
            spec.sample(rng)

    def test_mean_approximately_correct(self):
        rng = np.random.default_rng(0)
        spec = ParameterSpec(dist="beta", mean=0.6, std=0.08)
        draws = [spec.sample(rng) for _ in range(5000)]
        assert abs(sum(draws) / len(draws) - 0.6) < 0.05


class TestParameterSpecUniform:
    def test_within_bounds(self, rng):
        spec = ParameterSpec(dist="uniform", lo=1.0, hi=10.0)
        draws = [spec.sample(rng) for _ in range(500)]
        assert all(1.0 <= d <= 10.0 for d in draws)


class TestParameterSpecUnknown:
    def test_unknown_dist_raises(self, rng):
        spec = ParameterSpec(dist="gamma", mean=1.0)
        with pytest.raises(ValueError, match="Unknown distribution"):
            spec.sample(rng)


class TestSampleScenarioParameters:
    def test_fixed_priors_deterministic(self, fixed_priors):
        r1 = np.random.default_rng(0)
        r2 = np.random.default_rng(0)
        p1 = sample_scenario_parameters(r1, fixed_priors)
        p2 = sample_scenario_parameters(r2, fixed_priors)
        assert p1.lambda_F == p2.lambda_F
        assert p1.theta == p2.theta

    def test_integer_fields_are_int(self, fixed_priors, rng):
        params = sample_scenario_parameters(rng, fixed_priors)
        assert isinstance(params.Q_thr_rev, int)
        assert isinstance(params.Q_thr_esc, int)
        assert isinstance(params.Q_thr_block, int)
        assert isinstance(params.R_warmup, int)

    def test_integer_fields_minimum_one(self, fixed_priors):
        # Force Q_thr_rev mean to 0.4 → round(0.4)=0 → max(1,0)=1
        from src.sampling import ScenarioPriors
        specs = dict(fixed_priors.specs)
        specs["Q_thr_rev"] = ParameterSpec(dist="fixed", mean=0.4)
        priors = ScenarioPriors(specs=specs, c_token=fixed_priors.c_token)
        params = sample_scenario_parameters(np.random.default_rng(0), priors)
        assert params.Q_thr_rev == 1

    def test_D0_clamped_below(self, fixed_priors):
        from src.sampling import ScenarioPriors
        specs = dict(fixed_priors.specs)
        specs["D0"] = ParameterSpec(dist="fixed", mean=-0.5)
        priors = ScenarioPriors(specs=specs, c_token=fixed_priors.c_token)
        params = sample_scenario_parameters(np.random.default_rng(0), priors)
        assert params.D0 == 0.0

    def test_D0_clamped_above(self, fixed_priors):
        from src.sampling import ScenarioPriors
        specs = dict(fixed_priors.specs)
        specs["D0"] = ParameterSpec(dist="fixed", mean=1.5)
        priors = ScenarioPriors(specs=specs, c_token=fixed_priors.c_token)
        params = sample_scenario_parameters(np.random.default_rng(0), priors)
        assert params.D0 == 1.0

    def test_c_token_passed_through(self, fixed_priors, rng):
        params = sample_scenario_parameters(rng, fixed_priors)
        assert params.c_token == fixed_priors.c_token
        assert len(params.c_token) == 13
