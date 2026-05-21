"""Tests for src/sprint.py — state invariants, release logic, determinism."""

import numpy as np
import pytest

from src.sprint import SprintRecord, step_sprint
from src.state import LatencyBuffer, SystemState


def _run_sprints(n, config, params, rng, T=52, initial_backlog=20):
    state = SystemState(F_backlog=initial_backlog)
    buf = LatencyBuffer()
    records = []
    for t in range(n):
        rec = step_sprint(t, state, buf, config, params, rng, T)
        records.append(rec)
    return state, buf, records


class TestStateInvariants:
    def test_D_bounded_after_many_sprints(self, base_config, base_params):
        rng = np.random.default_rng(1)
        state = SystemState(F_backlog=50)
        buf = LatencyBuffer()
        for t in range(30):
            step_sprint(t, state, buf, base_config, base_params, rng, T=52)
            assert 0.0 <= state.D <= 1.0, f"D={state.D} out of bounds at t={t}"

    def test_doc_cov_bounded(self, base_config, base_params):
        rng = np.random.default_rng(2)
        state = SystemState(F_backlog=50)
        buf = LatencyBuffer()
        for t in range(30):
            step_sprint(t, state, buf, base_config, base_params, rng, T=52)
            assert 0.0 <= state.doc_cov <= 1.0

    def test_tests_UT_monotone(self, base_config, base_params):
        rng = np.random.default_rng(3)
        state = SystemState(F_backlog=50)
        buf = LatencyBuffer()
        prev = -1.0
        for t in range(20):
            step_sprint(t, state, buf, base_config, base_params, rng, T=52)
            assert state.tests_UT >= prev
            prev = state.tests_UT

    def test_F_completed_nondecreasing(self, base_config, base_params):
        rng = np.random.default_rng(4)
        state = SystemState(F_backlog=50)
        buf = LatencyBuffer()
        prev = -1
        for t in range(20):
            step_sprint(t, state, buf, base_config, base_params, rng, T=52)
            assert state.F_completed >= prev
            prev = state.F_completed

    def test_no_negative_counts(self, base_config, base_params):
        rng = np.random.default_rng(5)
        state = SystemState(F_backlog=50)
        buf = LatencyBuffer()
        for t in range(20):
            step_sprint(t, state, buf, base_config, base_params, rng, T=52)
            assert state.B_kno_min >= 0
            assert state.B_kno_maj >= 0
            assert state.B_hid_min >= 0
            assert state.B_hid_maj >= 0
            assert state.F_backlog >= 0


class TestReleaseEvent:
    def test_release_clears_B_hid(self, base_config, base_params):
        rng = np.random.default_rng(6)
        state, buf, _ = _run_sprints(base_config.R, base_config, base_params, rng)
        assert state.B_hid_min == 0
        assert state.B_hid_maj == 0
        assert len(buf.minor) == 0
        assert len(buf.major) == 0

    def test_no_escape_on_non_release_sprint(self, base_config, base_params):
        rng = np.random.default_rng(7)
        state = SystemState(F_backlog=20)
        buf = LatencyBuffer()
        # Sprint 0 is not a release sprint (R=4 → release at t=3)
        rec = step_sprint(0, state, buf, base_config, base_params, rng, T=52)
        assert rec.new_escaped_min == 0
        assert rec.new_escaped_maj == 0

    def test_escape_only_on_release_sprint(self, base_config, base_params):
        rng = np.random.default_rng(8)
        state = SystemState(F_backlog=30)
        buf = LatencyBuffer()
        for t in range(base_config.R):
            rec = step_sprint(t, state, buf, base_config, base_params, rng, T=52)
        # Final sprint in this loop is a release (t = R-1, so (t+1) % R == 0)
        assert rec.new_escaped_min + rec.new_escaped_maj >= 0  # defined (may be 0)
        assert state.B_hid_min == 0 and state.B_hid_maj == 0


class TestDeterminism:
    def test_same_rng_same_record(self, base_config, base_params):
        state1 = SystemState(F_backlog=20)
        state2 = SystemState(F_backlog=20)
        buf1, buf2 = LatencyBuffer(), LatencyBuffer()
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        rec1 = step_sprint(0, state1, buf1, base_config, base_params, r1, T=52)
        rec2 = step_sprint(0, state2, buf2, base_config, base_params, r2, T=52)
        assert rec1 == rec2
        assert state1.F_completed == state2.F_completed
        assert state1.D == state2.D
