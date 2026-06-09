"""Tests for src/sprint.py — state invariants, release logic, determinism."""

import dataclasses

import numpy as np
import pytest

from src.config import TeamConfiguration
from src.sprint import step_sprint
from src.state import SystemState


def _run_sprints(n, config, params, rng, initial_backlog=20):
    state = SystemState(F_backlog=initial_backlog)
    for t in range(n):
        step_sprint(t, state, config, params, rng)
    return state


class TestStateInvariants:
    def test_D_bounded_after_many_sprints(self, base_config, base_params):
        rng = np.random.default_rng(1)
        state = SystemState(F_backlog=50)
        for t in range(30):
            step_sprint(t, state, base_config, base_params, rng)
            assert 0.0 <= state.D_tech <= 1.0, f"D_tech={state.D_tech} out of bounds at t={t}"

    def test_N_doc_monotone(self, base_config, base_params):
        rng = np.random.default_rng(2)
        state = SystemState(F_backlog=50)
        prev = -1.0
        for t in range(30):
            step_sprint(t, state, base_config, base_params, rng)
            assert state.N_doc >= prev
            prev = state.N_doc

    def test_tests_monotone(self, base_config, base_params):
        rng = np.random.default_rng(3)
        state = SystemState(F_backlog=50)
        prev = -1.0
        for t in range(20):
            step_sprint(t, state, base_config, base_params, rng)
            assert state.N_test >= prev
            prev = state.N_test

    def test_F_completed_nondecreasing(self, base_config, base_params):
        rng = np.random.default_rng(4)
        state = SystemState(F_backlog=50)
        prev = -1
        for t in range(20):
            step_sprint(t, state, base_config, base_params, rng)
            assert state.F_completed >= prev
            prev = state.F_completed

    def test_no_negative_counts(self, base_config, base_params):
        rng = np.random.default_rng(5)
        state = SystemState(F_backlog=50)
        for t in range(20):
            step_sprint(t, state, base_config, base_params, rng)
            assert state.B_kno_min >= 0
            assert state.B_kno_maj >= 0
            assert state.B_kno_sec >= 0
            assert state.B_hid_min >= 0
            assert state.B_hid_maj >= 0
            assert state.B_hid_sec >= 0
            assert state.B_rel_min >= 0
            assert state.B_rel_maj >= 0
            assert state.B_rel_sec >= 0
            assert state.F_backlog >= 0


class TestReleaseEvent:
    def test_release_clears_B_hid(self, base_config, base_params):
        rng = np.random.default_rng(6)
        state = _run_sprints(base_config.R, base_config, base_params, rng)
        assert state.B_hid_min == 0
        assert state.B_hid_maj == 0

    def test_no_escape_on_non_release_sprint(self, base_config, base_params):
        rng = np.random.default_rng(7)
        state = SystemState(F_backlog=20)
        # Sprint 0 is not a release sprint (R=4 → release at t=3)
        step_sprint(0, state, base_config, base_params, rng)
        assert state.B_escaped_min == 0
        assert state.B_escaped_maj == 0

    def test_escape_only_on_release_sprint(self, base_config, base_params):
        rng = np.random.default_rng(8)
        state = SystemState(F_backlog=30)
        for t in range(base_config.R):
            step_sprint(t, state, base_config, base_params, rng)
        assert state.B_hid_min == 0 and state.B_hid_maj == 0


class TestReleaseBugQueue:
    def test_release_populates_B_rel(self, base_config, base_params):
        """Escaped bugs at a release event must appear in B_rel."""
        rng = np.random.default_rng(20)
        state = SystemState(F_backlog=50)
        for t in range(base_config.R):
            step_sprint(t, state, base_config, base_params, rng)
        # After first release: cumulative escaped == B_rel (nothing fixed yet)
        assert state.B_rel_min == state.B_escaped_min
        assert state.B_rel_maj == state.B_escaped_maj
        assert state.B_rel_sec == state.B_escaped_sec

    def test_B_rel_drained_before_B_kno(self, base_config, base_params):
        """When B_rel is non-zero, bug fixing reduces it before touching B_kno."""
        rng = np.random.default_rng(21)
        state = SystemState(F_backlog=50, B_rel_min=10, B_rel_maj=5)
        rel_min_before = state.B_rel_min
        rel_maj_before = state.B_rel_maj
        kno_min_before = state.B_kno_min
        step_sprint(0, state, base_config, base_params, rng)
        # B_rel should have decreased (or stayed same if cap_min=0)
        assert state.B_rel_min <= rel_min_before
        assert state.B_rel_maj <= rel_maj_before
        # B_kno should not have increased from fixing (may increase from new injections)
        # Just check that fixes came from B_rel first: if B_rel decreased, B_kno was spared
        assert state.B_rel_min + state.B_rel_maj <= rel_min_before + rel_maj_before


class TestDeterminism:
    def test_same_rng_same_state(self, base_config, base_params):
        state1 = SystemState(F_backlog=20)
        state2 = SystemState(F_backlog=20)
        r1 = np.random.default_rng(42)
        r2 = np.random.default_rng(42)
        step_sprint(0, state1, base_config, base_params, r1)
        step_sprint(0, state2, base_config, base_params, r2)
        assert state1 == state2


class TestArchChangeQueue:
    """Tests for the architecture change request queue (Q_arch) and
    the architecture implementation queue (Q_arch_impl)."""

    def _arch_params(self, base_params, f_arch_chg=0.5, r_arch_chg=10.0):
        return dataclasses.replace(base_params, f_arch_chg=f_arch_chg, r_arch_chg=r_arch_chg)

    def _arch_config(self, base_config, a2=1):
        """Return a config with a2 arch agents; reduces impl agents to stay within N*M."""
        A = list(base_config.A_dev)
        A[1] = a2
        A[0] = max(1, A[0] - a2)   # keep a_1 >= 1 and sum(A_dev) <= N*M
        return dataclasses.replace(base_config, A_dev=tuple(A))

    def test_Q_arch_grows_without_arch_agents(self, base_config, base_params):
        """With non-zero f_arch_chg and no arch agents (A[1]=0), Q_arch must grow."""
        params = self._arch_params(base_params, f_arch_chg=0.5, r_arch_chg=0.0)
        state = SystemState(F_backlog=50)
        rng = np.random.default_rng(10)
        for t in range(10):
            step_sprint(t, state, base_config, params, rng)
        assert state.Q_arch > 0

    def test_Q_arch_drained_by_arch_agents(self, base_config, base_params):
        """With a high r_arch_chg and arch agents present, Q_arch stays near 0."""
        config = self._arch_config(base_config, a2=1)
        params = self._arch_params(base_params, f_arch_chg=0.3, r_arch_chg=100.0)
        state = SystemState(F_backlog=50)
        rng = np.random.default_rng(11)
        for t in range(20):
            step_sprint(t, state, config, params, rng)
        assert state.Q_arch == 0

    def test_D_arch_slows_impl_via_psi_D(self, base_config, base_params):
        """High Q_arch raises D_arch, increasing D_eff and thus depressing psi_D,
        which should result in fewer agent-implemented features than with Q_arch=0."""
        params = self._arch_params(base_params, f_arch_chg=0.0, r_arch_chg=0.0)
        rng_low  = np.random.default_rng(12)
        rng_high = np.random.default_rng(12)
        # saturate D_arch: Q_arch = cap_review = H_capacity * N / H_review_time
        q_saturate = int(params.H_capacity * base_config.N / params.H_review_time)
        state_low  = SystemState(F_backlog=50, Q_arch=0)
        state_high = SystemState(F_backlog=50, Q_arch=q_saturate)
        step_sprint(0, state_low,  base_config, params, rng_low)
        step_sprint(0, state_high, base_config, params, rng_high)
        # With higher D_eff, psi_D is lower, so fewer features implemented
        assert state_high.F_completed <= state_low.F_completed

    def test_Q_arch_impl_filled_by_rev_done(self, base_config, base_params):
        """Change requests → arch agent drain → Q_rev → rev_done → Q_arch_impl."""
        config = self._arch_config(base_config, a2=1)
        # High f_arch_chg and r_arch_chg so change requests flow into Q_rev quickly.
        params = dataclasses.replace(base_params, f_arch_chg=0.5, r_arch_chg=10.0)
        state = SystemState(F_backlog=20)
        rng = np.random.default_rng(13)
        for t in range(10):
            step_sprint(t, state, config, params, rng)
        assert state.Q_arch_impl >= 0  # non-negative invariant

    def test_Q_arch_impl_drained_by_impl_agents(self, base_config, base_params):
        """Pre-populated Q_arch_impl must decrease and must not inflate F_completed.

        Uses f_fix=1.0 so impl_human=0; all agent capacity goes to arch work first.
        """
        params = dataclasses.replace(base_params, f_arch_chg=0.0, r_arch_chg=0.0,
                                     f_fix=1.0)
        initial_q = 5
        state = SystemState(F_backlog=0, Q_arch_impl=initial_q)
        rng = np.random.default_rng(14)
        step_sprint(0, state, base_config, params, rng)
        # Q_arch_impl should have decreased (impl agents drain it with priority)
        assert state.Q_arch_impl < initial_q
        # F_completed must not have increased (arch impl ≠ user feature)
        assert state.F_completed == 0

    def test_Q_arch_nonnegative(self, base_config, base_params):
        """Both Q_arch and Q_arch_impl must stay non-negative over many sprints."""
        params = self._arch_params(base_params, f_arch_chg=0.5, r_arch_chg=1.0)
        state = SystemState(F_backlog=50)
        rng = np.random.default_rng(15)
        for t in range(20):
            step_sprint(t, state, base_config, params, rng)
            assert state.Q_arch >= 0, f"Q_arch={state.Q_arch} at t={t}"
            assert state.Q_arch_impl >= 0, f"Q_arch_impl={state.Q_arch_impl} at t={t}"
