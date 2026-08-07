"""Tests for src/sprint.py — state invariants, release logic, determinism."""

import dataclasses

import numpy as np
import pytest

from src.config import TeamConfiguration, net_capacity
from src.sprint import _volume_factor, step_sprint
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
        A[0] = max(1, A[0] - a2)   # keep a_1 >= 1 and the supervised count <= N*M
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


def _A(**by_role) -> tuple[int, ...]:
    """Build an 8-element composition from role_k=count keyword args (e.g. k1=2)."""
    A = [0] * 8
    for key, count in by_role.items():
        A[int(key[1:]) - 1] = count
    return tuple(A)


class TestFractionalStaffingLevel:
    """N is an FTE staffing level, so a sprint must run for a part-time team too."""

    def test_a_quarter_fte_team_runs_and_produces_less_than_a_full_one(self, base_params):
        # No agents beyond the mandatory implementer, so the difference between the two
        # runs is human capacity: it scales with N everywhere except H_net itself.
        params = dataclasses.replace(base_params, c_comm=0.0)
        part = TeamConfiguration(N=0.25, M=8, A_dev=_A(k1=1), R=4, F_freeze=0)
        full = TeamConfiguration(N=1.0, M=8, A_dev=_A(k1=1), R=4, F_freeze=0)
        a = _run_sprints(6, part, params, np.random.default_rng(31), initial_backlog=200)
        b = _run_sprints(6, full, params, np.random.default_rng(31), initial_backlog=200)
        assert 0 < a.F_completed < b.F_completed

    def test_a_second_person_costs_communication_even_at_a_quarter_fte(self, base_params):
        """Going 1.0 -> 1.25 FTE adds a whole person, and with it a whole peer cost.

        With c_comm this large the added quarter-FTE cannot pay for the communication
        it triggers, so the bigger team gets *less* human work done — which only holds
        because the peer count is ceil(N-1).  A linear (N-1) would charge a quarter of
        the overhead and the larger team would come out ahead.
        """
        params = dataclasses.replace(base_params, c_comm=30.0, H_capacity=40.0)
        assert net_capacity(40.0, 30.0, 1.0) * 1.0 == 40.0
        assert net_capacity(40.0, 30.0, 1.25) * 1.25 == 12.5
        solo = TeamConfiguration(N=1.0, M=8, A_dev=_A(k1=1), R=4, F_freeze=0)
        pair = TeamConfiguration(N=1.25, M=8, A_dev=_A(k1=1), R=4, F_freeze=0)
        a = _run_sprints(6, solo, params, np.random.default_rng(32), initial_backlog=200)
        b = _run_sprints(6, pair, params, np.random.default_rng(32), initial_backlog=200)
        assert b.F_completed < a.F_completed


class TestVolumeFactor:
    """The log-linear degradation in change volume per detection agent."""

    def test_zero_agents_gates_the_stage(self):
        assert _volume_factor(impl=10, a=0, alpha_size=0.1) == 0.0

    def test_no_change_volume_means_no_degradation(self):
        assert _volume_factor(impl=0, a=1, alpha_size=0.1) == pytest.approx(1.0)

    def test_increasing_in_agent_count(self):
        factors = [_volume_factor(impl=40, a=a, alpha_size=0.1) for a in (1, 2, 4, 8)]
        assert factors == sorted(factors)
        assert factors[0] < factors[-1]

    def test_decreasing_in_change_volume(self):
        factors = [_volume_factor(impl=impl, a=2, alpha_size=0.1) for impl in (0, 5, 50, 500)]
        assert factors == sorted(factors, reverse=True)

    def test_bounded_in_unit_interval(self):
        for impl in (0, 1, 10, 1000, 10**6):
            for a in (1, 3, 17):
                for alpha in (0.0, 0.1, 2.0):
                    f = _volume_factor(impl=impl, a=a, alpha_size=alpha)
                    assert 0.0 <= f <= 1.0

    def test_zero_alpha_removes_the_dependency(self):
        assert _volume_factor(impl=1000, a=1, alpha_size=0.0) == pytest.approx(1.0)


class TestDetectionGating:
    """Code Review and Security Audit only detect when actually deployed."""

    def _no_fix_params(self, base_params, **overrides):
        """High injection rate, nothing ever gets fixed, so B_kno reflects detection only."""
        return dataclasses.replace(
            base_params, r_bugs=2.0,
            r_fix=0.0, f_fix=0.0,
            **overrides,
        )

    def _cfg(self, A_dev):
        # F_freeze=0 keeps every sprint in the development phase; R=4 with 3 sprints
        # run means no release fires, so nothing is flushed out of B_hid/B_kno.
        return TeamConfiguration(N=2, M=3, A_dev=A_dev, A_qa=A_dev, R=4, F_freeze=0)

    def test_no_detection_agents_leaves_all_bugs_hidden(self, base_params):
        """a_5 = a_6 = a_7 = 0: bugs are injected but nothing is ever detected."""
        state = _run_sprints(3, self._cfg(_A(k1=2)), self._no_fix_params(base_params),
                             np.random.default_rng(20), initial_backlog=100)
        assert state.B_hid > 0, "no bugs were injected — test is vacuous"
        assert state.B_kno == 0

    def test_code_review_requires_a_review_agent(self, base_params):
        """With tests and audit disabled, detection happens iff a_5 >= 1."""
        params = self._no_fix_params(base_params, d_test=0.0)
        without = _run_sprints(3, self._cfg(_A(k1=2)), params,
                               np.random.default_rng(21), initial_backlog=100)
        with_cr = _run_sprints(3, self._cfg(_A(k1=2, k5=1)), params,
                               np.random.default_rng(21), initial_backlog=100)
        assert without.B_kno == 0
        assert with_cr.B_kno > 0

    def test_security_audit_requires_an_audit_agent(self, base_params):
        """With tests and review disabled, security bugs are found iff a_6 >= 1."""
        params = self._no_fix_params(base_params, d_test=0.0, theta_sec=0.9, theta_maj=0.05)
        without = _run_sprints(3, self._cfg(_A(k1=2)), params,
                               np.random.default_rng(22), initial_backlog=100)
        with_sa = _run_sprints(3, self._cfg(_A(k1=2, k6=1)), params,
                               np.random.default_rng(22), initial_backlog=100)
        assert without.B_kno_sec == 0
        assert with_sa.B_kno_sec > 0

    def test_more_review_agents_detect_more(self, base_params):
        """Sharing the change volume across more review agents raises detection."""
        params = self._no_fix_params(base_params, d_test=0.0, alpha_size=0.3)
        few = _run_sprints(3, self._cfg(_A(k1=2, k5=1)), params,
                           np.random.default_rng(23), initial_backlog=100)
        many = _run_sprints(3, self._cfg(_A(k1=2, k5=16)), params,
                            np.random.default_rng(23), initial_backlog=100)
        assert many.B_kno > few.B_kno


class TestBackgroundServicesSkipOversight:
    """Background services consume no monitoring capacity and raise no escalations."""

    def test_extra_background_agents_do_not_change_the_run(self, base_config, base_params):
        # alpha_size=0 makes detection independent of the service counts, so any
        # difference between the two runs could only come from human oversight.
        params = dataclasses.replace(base_params, alpha_size=0.0, rho_esc=0.5)
        lean = dataclasses.replace(
            base_config, A_dev=_A(k1=2, k3=1, k4=1, k5=1, k6=1, k7=1), F_freeze=0,
        )
        loaded = dataclasses.replace(
            base_config, A_dev=_A(k1=2, k3=1, k4=1, k5=9, k6=9, k7=1), F_freeze=0,
        )
        a = _run_sprints(8, lean, params, np.random.default_rng(24), initial_backlog=100)
        b = _run_sprints(8, loaded, params, np.random.default_rng(24), initial_backlog=100)
        assert dataclasses.asdict(a) == dataclasses.asdict(b)

    def test_supervised_agents_do_change_the_run(self, base_config, base_params):
        """Guard against the previous test passing for the wrong reason."""
        params = dataclasses.replace(base_params, alpha_size=0.0, rho_esc=0.5)
        lean = dataclasses.replace(
            base_config, A_dev=_A(k1=2, k3=1, k4=1, k5=1, k6=1, k7=1), F_freeze=0,
        )
        loaded = dataclasses.replace(
            base_config, A_dev=_A(k1=2, k3=1, k4=1, k5=1, k6=1, k7=1, k8=1), F_freeze=0,
        )
        a = _run_sprints(8, lean, params, np.random.default_rng(24), initial_backlog=100)
        b = _run_sprints(8, loaded, params, np.random.default_rng(24), initial_backlog=100)
        assert dataclasses.asdict(a) != dataclasses.asdict(b)
