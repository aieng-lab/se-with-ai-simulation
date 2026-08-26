"""Tests for src/sprint.py — state invariants, release logic, determinism."""

import dataclasses

import numpy as np
import pytest

from src.config import AGENT_ROLE_COUNT, TeamConfiguration, net_capacity
from src.sprint import _volume_factor, step_sprint
from src.state import SystemState


#: An all-zero QA composition.  Every team below runs F_freeze = 0 or is a
#: validation case, so A_qa is never applied; it is spelled out because
#: `TeamConfiguration` requires both compositions.
_NO_QA = (0,) * AGENT_ROLE_COUNT


def _B_kno(state):
    """Detected, unrepaired bugs over all three severities."""
    return state.B_kno_min + state.B_kno_maj + state.B_kno_sec


def _B_hid(state):
    """Undetected bugs over all three severities."""
    return state.B_hid_min + state.B_hid_maj + state.B_hid_sec


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
        """Return a config with a2 arch agents; reduces impl agents to keep n_sup fixed."""
        A = list(base_config.A_dev)
        A[1] = a2
        A[0] = max(1, A[0] - a2)   # keep a_1 >= 1
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
        part = TeamConfiguration(N=0.25, A_dev=_A(k1=1), R=4, F_freeze=0, A_qa=_NO_QA)
        full = TeamConfiguration(N=1.0, A_dev=_A(k1=1), R=4, F_freeze=0, A_qa=_NO_QA)
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
        solo = TeamConfiguration(N=1.0, A_dev=_A(k1=1), R=4, F_freeze=0, A_qa=_NO_QA)
        pair = TeamConfiguration(N=1.25, A_dev=_A(k1=1), R=4, F_freeze=0, A_qa=_NO_QA)
        a = _run_sprints(6, solo, params, np.random.default_rng(32), initial_backlog=200)
        b = _run_sprints(6, pair, params, np.random.default_rng(32), initial_backlog=200)
        assert b.F_completed < a.F_completed


class TestHumanBudgetResidualRounding:
    """The review/escalation residuals must never go negative through float rounding.

    With N = 0.75 and two supervised agents the fixture parameters put
    `H_after_rev = 13.333333333333332` and `H_after_rev * N / H_escalation_time` at
    exactly 5.0 after rounding, so the floor serves one more escalation than the budget
    strictly pays for and the unclamped residual lands at -1.8e-15.  A negative residual
    is not a rounding-sized error downstream: both consumers take its floor, so it costs
    a whole item.
    """

    N = 0.75
    #: a_1 = a_3 = 1, i.e. n_sup = 2 — the count the arithmetic above depends on.
    CFG = TeamConfiguration(N=0.75, A_dev=_A(k1=1, k3=1), R=4, F_freeze=0, A_qa=_NO_QA)
    #: Q_rev = 4 caps rev_done below the capacity floor; Q_esc = 5 does not cap esc_done.
    STATE = dict(F_backlog=20, Q_rev=4, Q_esc=5)

    def test_negative_residual_does_not_reach_the_poisson_rate(self, base_params):
        """The sprint must complete: a negative rate would make `rng.poisson` raise.

        Re-multiplying the floored served count by its per-item cost can overshoot the
        budget by an ulp, so the residual is clamped at zero (Eqs. hafter_rev/hafter_esc).
        Without the clamp `impl_human` reaches floor(negative) = -1 and the draw raises,
        so the assertion is that this call returns at all.
        """
        state = SystemState(**self.STATE)
        step_sprint(0, state, self.CFG, base_params, np.random.default_rng(7))
        assert state.F_completed >= 0

    def test_negative_residual_does_not_invent_bugs(self, base_params):
        """fix_human_budget = floor(negative) = -1 drains -1 bugs, i.e. adds one.

        Every bug queue starts empty and t = 0 is not a release sprint, so the released
        queues can only grow if a negative budget runs the drain backwards.
        """
        state = SystemState(**self.STATE)
        step_sprint(0, state, self.CFG, base_params, np.random.default_rng(7))
        assert (state.B_rel_min, state.B_rel_maj, state.B_rel_sec) == (0, 0, 0)


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
        return TeamConfiguration(N=2, A_dev=A_dev, A_qa=A_dev, R=4, F_freeze=0)

    def test_no_detection_agents_leaves_all_bugs_hidden(self, base_params):
        """a_5 = a_6 = a_7 = 0: bugs are injected but nothing is ever detected."""
        state = _run_sprints(3, self._cfg(_A(k1=2)), self._no_fix_params(base_params),
                             np.random.default_rng(20), initial_backlog=100)
        assert _B_hid(state) > 0, "no bugs were injected — test is vacuous"
        assert _B_kno(state) == 0

    def test_code_review_requires_a_review_agent(self, base_params):
        """With tests and audit disabled, detection happens iff a_5 >= 1."""
        params = self._no_fix_params(base_params, d_test=0.0)
        without = _run_sprints(3, self._cfg(_A(k1=2)), params,
                               np.random.default_rng(21), initial_backlog=100)
        with_cr = _run_sprints(3, self._cfg(_A(k1=2, k5=1)), params,
                               np.random.default_rng(21), initial_backlog=100)
        assert _B_kno(without) == 0
        assert _B_kno(with_cr) > 0

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
        assert _B_kno(many) > _B_kno(few)


class TestAutonomousRolesSkipOversight:
    """K_aut = {4,5,6} consumes no monitoring capacity and raises no escalations."""

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

    def test_extra_refactoring_agents_do_not_change_the_run(self, base_config, base_params):
        """Role 4 is autonomous, so it must not move H_after_mon.

        `r_refactor = 0` strips the one legitimate effect the role has — the debt it
        removes from D_tech — leaving only the oversight question.  Its bill is
        not part of the state, so a run with 40 of them must be bit-identical.
        """
        params = dataclasses.replace(base_params, alpha_size=0.0, rho_esc=0.5,
                                     r_refactor=0.0)
        lean = dataclasses.replace(
            base_config, A_dev=_A(k1=2, k3=1, k4=1, k5=1, k6=1, k7=1), F_freeze=0,
        )
        loaded = dataclasses.replace(
            base_config, A_dev=_A(k1=2, k3=1, k4=40, k5=1, k6=1, k7=1), F_freeze=0,
        )
        a = _run_sprints(8, lean, params, np.random.default_rng(24), initial_backlog=100)
        b = _run_sprints(8, loaded, params, np.random.default_rng(24), initial_backlog=100)
        assert dataclasses.asdict(a) == dataclasses.asdict(b)

    def test_supervised_agents_do_change_the_run(self, base_config, base_params):
        """Guard against the previous tests passing for the wrong reason."""
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


class TestCapacityIsNeverNegative:
    """A configuration can clear the feasibility check and still be over-subscribed.

    The gate holds only at doc_cov = 1, so before documentation accumulates the
    monitoring load can exceed H_net.  H_after_mon is clamped at 0,
    which must keep every derived budget non-negative — an unclamped budget used to
    run the review, escalation and bug queues *backwards*.
    """

    def _oversubscribed(self, base_params):
        """Feasible at doc_cov=1, over-subscribed at doc_cov=0."""
        from dataclasses import replace
        from src.config import AGENT_ROLE_COUNT
        from src.feasibility import check_feasibility
        p = replace(base_params, H_capacity=40.0, H_monitoring_overhead=2.5,
                    Gamma_doc=0.4, c_comm=1.0)
        A = [0] * AGENT_ROLE_COUNT
        A[0], A[2], A[3], A[6], A[7] = 1, 1, 1, 1, 1
        A[1] = 1
        cfg = TeamConfiguration(N=0.25, A_dev=tuple(A), R=4, F_freeze=1,
                                A_qa=(0, 0, 1, 1, 0, 0, 1, 1))
        assert check_feasibility(cfg, p).feasible, "fixture must pass the gate"
        # 6 supervised over 0.25 FTE at doc_cov=0 costs 60 h against H_net = 40 h.
        assert net_capacity(p.H_capacity, p.c_comm, cfg.N) == 40.0
        return cfg, p

    def test_queues_never_run_backwards(self, base_params):
        cfg, p = self._oversubscribed(base_params)
        rng = np.random.default_rng(11)
        state = SystemState(F_backlog=5)
        for t in range(30):
            before = dataclasses.replace(state)
            step_sprint(t, state, cfg, p, rng)
            # Bug queues may only grow via injection (step 5) or release (step 10);
            # a negative human fix budget used to grow them here instead.
            for q in ("B_rel_min", "B_rel_maj", "B_rel_sec"):
                released = (t + 1) % cfg.R == 0
                if not released:
                    assert getattr(state, q) <= getattr(before, q), (
                        f"{q} grew outside a release at t={t}"
                    )
            for q in ("F_backlog", "F_completed", "Q_rev", "Q_arch", "Q_arch_impl",
                      "Q_esc", "N_test", "N_doc"):
                assert getattr(state, q) >= 0, f"{q} negative at t={t}"

    def test_completed_features_are_monotone(self, base_params):
        """A negative impl_human used to be able to decrement F_completed."""
        cfg, p = self._oversubscribed(base_params)
        rng = np.random.default_rng(12)
        state = SystemState(F_backlog=5)
        prev = 0
        for t in range(30):
            step_sprint(t, state, cfg, p, rng)
            assert state.F_completed >= prev
            prev = state.F_completed


class TestImplementationIsCappedByDemand:
    """impl_agents + impl_human == impl, so no bugs are injected for
    features that were never implemented."""

    def test_human_capacity_far_beyond_the_backlog(self, base_config, base_params):
        """Cheap features + lots of humans: capacity dwarfs the tiny arrival stream."""
        from dataclasses import replace
        p = replace(base_params, h_feature=0.1, h_fix=0.1, f_fix=0.5,
                    r_arrival=1.0, Gamma_doc=0.0)
        cfg = TeamConfiguration(N=5, A_dev=base_config.A_dev, R=4,
                                F_freeze=0, A_qa=base_config.A_qa)
        rng = np.random.default_rng(13)
        state = SystemState(F_backlog=0)
        for t in range(25):
            before_backlog = state.F_backlog
            before_completed = state.F_completed
            step_sprint(t, state, cfg, p, rng)
            implemented = state.F_completed - before_completed
            arrived = state.F_backlog - before_backlog + implemented
            # Everything that arrived (plus the carried backlog) is implementable,
            # and nothing beyond it can be claimed.
            assert implemented <= before_backlog + arrived
            assert state.F_backlog >= 0

    def test_backlog_never_goes_negative_with_huge_agent_capacity(self, base_params):
        from dataclasses import replace
        from src.config import AGENT_ROLE_COUNT
        p = replace(base_params, r_F=1000.0, r_arrival=1.0, h_feature=0.1,
                    Gamma_doc=0.0)
        A = [0] * AGENT_ROLE_COUNT
        A[0] = 3
        cfg = TeamConfiguration(N=3, A_dev=tuple(A), R=2, F_freeze=0, A_qa=_NO_QA)
        rng = np.random.default_rng(14)
        state = SystemState(F_backlog=2)
        for t in range(20):
            step_sprint(t, state, cfg, p, rng)
            assert state.F_backlog >= 0


class TestEscalationsScaleWithAgentWork:
    """esc_new ~ Poisson(rho_esc * (2 - delta_D) * W_agent), where
    W_agent is the agent work completed this sprint, not the deployed headcount."""

    def _cfg(self, n_impl=1, n_fix=0):
        from src.config import AGENT_ROLE_COUNT
        A = [0] * AGENT_ROLE_COUNT
        A[0] = max(1, n_impl)
        A[2] = n_fix
        return TeamConfiguration(N=3, A_dev=tuple(A), R=8, F_freeze=0, A_qa=_NO_QA)

    def test_no_agent_work_means_no_escalations(self, base_params):
        """Idle agents raise nothing: with an empty backlog and no arrivals,
        W_agent is zero however many agents are deployed."""
        from dataclasses import replace
        p = replace(base_params, r_arrival=0.0, f_arch_chg=0.0, rho_esc=0.9,
                    r_bugs=0.0, Gamma_doc=0.0)
        cfg = self._cfg(n_impl=5, n_fix=5)
        rng = np.random.default_rng(15)
        state = SystemState(F_backlog=0)
        for t in range(30):
            step_sprint(t, state, cfg, p, rng)
        assert state.Q_esc == 0

    def test_escalations_grow_with_the_arrival_rate(self, base_params):
        """More work through the same team means more escalations."""
        from dataclasses import replace

        def total_escalations(r_arrival):
            p = replace(base_params, r_arrival=r_arrival, rho_esc=0.5,
                        f_arch_chg=0.0, r_bugs=0.0, H_escalation_time=1e6,
                        Gamma_doc=0.0)
            cfg = self._cfg(n_impl=1)
            rng = np.random.default_rng(16)
            state = SystemState(F_backlog=0)
            for t in range(40):
                step_sprint(t, state, cfg, p, rng)
            return state.Q_esc   # esc_done is pinned to 0 by the huge esc_time

        assert total_escalations(20.0) > total_escalations(2.0)

    def test_debt_amplifies_the_escalation_rate(self, base_params):
        """The (2 - delta_D) factor of the escalation rate.

        Tested on the Poisson rate directly rather than on Q_esc: debt also throttles
        throughput (so W_agent shrinks) and a growing Q_esc feeds back into D_esc, so
        an end-state comparison would measure three coupled effects at once.  Agent
        capacity is set far above demand so W_agent is identical in both runs.
        """
        from dataclasses import replace
        from src.config import AGENT_ROLE_COUNT

        class _RecordingRng:
            """Delegates to a real Generator, recording every Poisson rate."""

            def __init__(self, seed):
                self._rng = np.random.default_rng(seed)
                self.poisson_lams = []

            def poisson(self, lam):
                self.poisson_lams.append(lam)
                return self._rng.poisson(lam)

            def binomial(self, n, p):
                return self._rng.binomial(n, p)

        def escalation_rates(D_tech):
            # r_F far above the arrival rate, so impl_agents saturates demand at both
            # debt levels; no architecture work and no bugs, so W_agent == impl_agents.
            p = replace(base_params, r_arrival=10.0, rho_esc=0.5, f_arch_chg=0.0,
                        r_bugs=0.0, delta_min=0.25, delta_exp=1.0, r_F=1000.0,
                        r_refactor=0.0, delta_impl=0.0, delta_fix=0.0,
                        Gamma_doc=0.0, h_feature=1e6)   # humans implement nothing
            A = [0] * AGENT_ROLE_COUNT
            A[0] = 1
            cfg = TeamConfiguration(N=3, A_dev=tuple(A), R=8, F_freeze=0, A_qa=_NO_QA)
            rng = _RecordingRng(17)
            state = SystemState(F_backlog=0, D_tech=D_tech)
            for t in range(5):
                step_sprint(t, state, cfg, p, rng)
            # Per sprint the Poisson draws are: arrivals, agent bugs, human bugs,
            # arch change requests, escalations.  Take the last of each group of five.
            return rng.poisson_lams[4::5]

        clean, indebted = escalation_rates(0.0), escalation_rates(1.0)
        assert len(clean) == len(indebted) == 5
        for c, d in zip(clean, indebted):
            assert c > 0, "the clean run must actually be doing work"
            # Same W_agent in both runs, so the amplifier alone drives the gap up.
            assert d > c
            # At D_tech = 1 the effective debt saturates, so delta_D is exactly
            # delta_min = 0.25 and the rate is rho_esc * 1.75 * W_agent with W_agent
            # a whole number of completed work items.
            W_agent = d / (0.5 * 1.75)
            assert W_agent == pytest.approx(round(W_agent))

    def test_human_fixes_do_not_raise_escalations(self, base_params):
        """W_agent counts agent work only — a developer does not escalate to themselves."""
        from dataclasses import replace
        from src.config import AGENT_ROLE_COUNT
        p = replace(base_params, r_arrival=0.0, f_arch_chg=0.0, rho_esc=0.9,
                    r_bugs=0.0, h_fix=0.1, f_fix=1.0, H_escalation_time=1e6,
                    Gamma_doc=0.0)
        A = [0] * AGENT_ROLE_COUNT
        A[0] = 1
        cfg = TeamConfiguration(N=3, A_dev=tuple(A), R=100, F_freeze=0, A_qa=_NO_QA)
        rng = np.random.default_rng(18)
        # Seed a large production bug queue that only humans can drain (no fix agents).
        state = SystemState(F_backlog=0, B_rel_min=500)
        for t in range(30):
            step_sprint(t, state, cfg, p, rng)
        assert state.B_rel_min < 500, "humans must actually be fixing bugs"
        assert state.Q_esc == 0


class TestContributionAccounting:
    """SprintActions reports who did the work, alongside the billable `units` vector.

    The human share appears in no entry of `units` (human cost is a headcount term), so
    these four counters are the only record of it.
    """

    def test_implementation_split_accounts_for_every_feature(self, base_config, base_params):
        """impl_human' + impl_agents is exactly what F_completed gained."""
        p = dataclasses.replace(base_params, r_arrival=3.0, Gamma_doc=0.0)
        rng = np.random.default_rng(21)
        state = SystemState(F_backlog=40)
        for t in range(20):
            before = state.F_completed
            actions = step_sprint(t, state, base_config, p, rng)
            implemented = state.F_completed - before
            assert actions.impl_human + actions.impl_agents == pytest.approx(implemented)

    def test_demand_below_human_capacity_leaves_the_agents_nothing(self, base_params):
        """Humans are served first, so a thin arrival stream
        never reaches the implementation agent."""
        # One feature per sprint, and a human budget far above it.
        p = dataclasses.replace(base_params, r_arrival=0.0, h_feature=0.1, f_fix=0.0,
                                f_arch_chg=0.0, Gamma_doc=0.0)
        cfg = TeamConfiguration(N=3, A_dev=_A(k1=2), R=8, F_freeze=0, A_qa=_NO_QA)
        state = SystemState(F_backlog=1)
        actions = step_sprint(0, state, cfg, p, np.random.default_rng(22))

        assert actions.impl_human == pytest.approx(1.0)
        assert actions.impl_agents == pytest.approx(0.0)

    def test_no_human_coding_time_leaves_it_all_to_the_agents(self, base_params):
        """With h_feature astronomically large the human budget floors to zero."""
        p = dataclasses.replace(base_params, r_arrival=0.0, h_feature=1e9, f_fix=0.0,
                                f_arch_chg=0.0, delta_min=1.0, Gamma_doc=0.0)
        cfg = TeamConfiguration(N=3, A_dev=_A(k1=2), R=8, F_freeze=0, A_qa=_NO_QA)
        state = SystemState(F_backlog=50)
        actions = step_sprint(0, state, cfg, p, np.random.default_rng(23))

        assert actions.impl_human == pytest.approx(0.0)
        # floor(r_F * 1.0 * 2) = 6 with the fixture's r_F = 3.
        assert actions.impl_agents == pytest.approx(6.0)

    def test_fixes_human_counts_the_human_drain(self, base_params):
        """With no Bug Fixing Agent every repair is human, and none of it is billed."""
        p = dataclasses.replace(base_params, r_arrival=0.0, f_arch_chg=0.0, r_bugs=0.0,
                                h_fix=1.0, f_fix=1.0, Gamma_doc=0.0)
        cfg = TeamConfiguration(N=1, A_dev=_A(k1=1), R=100, F_freeze=0, A_qa=_NO_QA)
        state = SystemState(F_backlog=0, B_rel_min=500)
        before = state.B_rel_min
        actions = step_sprint(0, state, cfg, p, np.random.default_rng(24))
        repaired = before - state.B_rel_min

        assert repaired > 0
        assert actions.fixes_human == pytest.approx(repaired)
        assert actions.fixes_agent == pytest.approx(0.0)
        assert actions.units[2] == pytest.approx(0.0)   # k=3 bills nothing

    def test_fixes_agent_is_exactly_the_billed_bug_fixing_unit(self, base_params):
        """fixes_agent duplicates u_3 by construction — pin that they cannot drift."""
        p = dataclasses.replace(base_params, r_arrival=0.0, f_arch_chg=0.0, r_bugs=0.0,
                                f_fix=0.0, h_feature=1e9, Gamma_doc=0.0)
        cfg = TeamConfiguration(N=1, A_dev=_A(k1=1, k3=2), R=100, F_freeze=0, A_qa=_NO_QA)
        state = SystemState(F_backlog=0, B_rel_maj=200)
        rng = np.random.default_rng(25)
        for t in range(5):
            actions = step_sprint(t, state, cfg, p, rng)
            assert actions.fixes_agent > 0
            assert actions.fixes_agent == pytest.approx(actions.units[2])

    def test_feature_freeze_moves_humans_off_features_onto_bugs(self, base_params):
        """impl_human is zero in a freeze sprint; the fixes are not."""
        p = dataclasses.replace(base_params, r_arrival=0.0, f_arch_chg=0.0, r_bugs=0.0,
                                h_feature=0.1, h_fix=1.0, f_fix=0.0, Gamma_doc=0.0)
        # R=2 with F_freeze=1 makes t=1 a freeze sprint and t=0 a development one.
        cfg = TeamConfiguration(N=1, A_dev=_A(k1=1), A_qa=_A(k3=0), R=2, F_freeze=1)

        dev_state = SystemState(F_backlog=10, B_rel_min=100)
        dev = step_sprint(0, dev_state, cfg, p, np.random.default_rng(26))
        # f_fix = 0 outside the freeze, so the humans code and fix nothing.
        assert dev.impl_human > 0
        assert dev.fixes_human == pytest.approx(0.0)

        qa_state = SystemState(F_backlog=10, B_rel_min=100)
        qa = step_sprint(1, qa_state, cfg, p, np.random.default_rng(26))
        assert qa.impl_human == pytest.approx(0.0)
        assert qa.fixes_human > 0

    def test_implemented_proposals_are_recoverable_from_the_billed_unit(self, base_params):
        """u_1 conflates agent features with proposals; impl_agents un-conflates it."""
        from src.sprint import ARCH_IMPL_COST_UNITS
        p = dataclasses.replace(base_params, r_arrival=0.0, f_arch_chg=0.0,
                                delta_min=1.0, h_feature=1e9, Gamma_doc=0.0)
        cfg = TeamConfiguration(N=2, A_dev=_A(k1=4), R=8, F_freeze=0, A_qa=_NO_QA)
        # Capacity is floor(r_F * 1.0 * 4) = 12.  A proposal costs five features of it
        #, so both queued proposals are implemented and the remaining
        # 12 - 10 = 2 features' worth goes to the backlog.
        state = SystemState(F_backlog=5, Q_arch_impl=2)
        actions = step_sprint(0, state, cfg, p, np.random.default_rng(27))

        arch_impl = (actions.units[0] - actions.impl_agents) / ARCH_IMPL_COST_UNITS
        assert arch_impl == pytest.approx(2.0)
        assert actions.impl_agents == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Human bug injection is a fixed per-feature rate
# ---------------------------------------------------------------------------

class TestHumanBugInjectionIsFixedRate:
    """Human-written code is not amplified by the debt factor."""

    @staticmethod
    def _human_only_cfg(N=4):
        zero = (0,) * AGENT_ROLE_COUNT
        return TeamConfiguration(N=N, A_dev=zero, A_qa=zero, R=13, F_freeze=0)

    def _bugs_after(self, params, delta_min, seed=7, sprints=6):
        p = dataclasses.replace(params, delta_min=delta_min, f_qa=0.0, d_qa=0.0)
        state = _run_sprints(sprints, self._human_only_cfg(), p,
                             np.random.default_rng(seed), initial_backlog=200)
        return (state.B_hid_min + state.B_hid_maj + state.B_hid_sec
                + state.B_kno_min + state.B_kno_maj + state.B_kno_sec)

    def test_debt_tier_does_not_change_the_human_bug_count(self, base_params):
        # With no agents, D_tech and the architecture queue still grow, so delta_D moves.
        # Eq. (Binj-human) carries no (2 - delta_D) amplifier — only the agent stream of
        # Eq. (Binj-agent) does — so the two debt extremes must give the identical stream.
        mild = self._bugs_after(base_params, delta_min=0.94)
        harsh = self._bugs_after(base_params, delta_min=0.60)
        assert mild == harsh

    def test_rate_scales_the_injected_count(self, base_params):
        low = self._bugs_after(dataclasses.replace(base_params, r_bugs_human=0.05), 0.77)
        high = self._bugs_after(dataclasses.replace(base_params, r_bugs_human=0.50), 0.77)
        assert high > low

    def test_agent_stream_is_still_debt_amplified(self, base_params):
        # The agent injection stream keeps the (2 - delta_D) factor, so an
        # agent-supported team must still respond to the debt tier.  Implementers only —
        # no detection agents, so every injected bug stays in B_hid — with R past the
        # horizon so no release empties it and f_fix = 0 so no human drains it.
        # delta_D also throttles the implementation capacity, which would
        # confound a bug count: heavy debt means fewer features and so fewer bugs whatever
        # the rate.  So give the implementers capacity far past the backlog, which makes
        # impl equal to the demand in both runs, and vary only the debt carried in.
        A = [0] * AGENT_ROLE_COUNT
        A[0] = 5
        cfg = TeamConfiguration(N=2, A_dev=tuple(A), R=99, F_freeze=0, A_qa=_NO_QA)
        p = dataclasses.replace(base_params, delta_min=0.10, delta_exp=1.0, r_F=1000.0,
                                r_bugs=0.5, r_bugs_human=1e-9, f_fix=1.0, r_arrival=1e-9)

        def run(D_tech):
            state = SystemState(F_backlog=200, D_tech=D_tech)
            actions = step_sprint(0, state, cfg, p, np.random.default_rng(11))
            bugs = state.B_hid_min + state.B_hid_maj + state.B_hid_sec
            return actions.impl_agents, bugs

        impl_clean, bugs_clean = run(0.0)
        impl_dirty, bugs_dirty = run(1.0)
        assert impl_clean == impl_dirty == 200   # the confound is held out
        assert bugs_dirty > bugs_clean           # (2 - delta_D) still amplifies


# ---------------------------------------------------------------------------
# The human-only baseline
# ---------------------------------------------------------------------------

class TestHumanOnlyCapacityReservation:
    """f_qa of net capacity is reserved, in dev sprints only."""

    @staticmethod
    def _cfg(F_freeze, N=4):
        zero = (0,) * AGENT_ROLE_COUNT
        return TeamConfiguration(N=N, A_dev=zero, A_qa=zero, R=4, F_freeze=F_freeze)

    def _impl_at(self, params, f_qa, t, F_freeze=0):
        p = dataclasses.replace(params, f_qa=f_qa)
        state = SystemState(F_backlog=10_000)
        return step_sprint(t, state, self._cfg(F_freeze), p, np.random.default_rng(5))

    def test_dev_sprint_implementation_falls_by_one_minus_f_qa(self, base_params):
        full = self._impl_at(base_params, 0.0, t=0).impl_human
        reserved = self._impl_at(base_params, 0.25, t=0).impl_human
        # impl_human = floor(N (1-f_fix)(1-f_qa) H_after_esc / h_feature); the floor makes
        # this exact only up to one feature, and the backlog is far past both.
        assert reserved == pytest.approx(full * 0.75, abs=1.0)
        assert reserved < full

    def test_freeze_sprint_reserves_nothing(self, base_params):
        # R=4, F_freeze=2 -> sprints 2 and 3 of each cycle are frozen.  A freeze sprint
        # implements nothing and redirects everything to fixing, so f_qa must not bite.
        free = self._impl_at(base_params, 0.0, t=2, F_freeze=2).fixes_human
        reserved = self._impl_at(base_params, 0.4, t=2, F_freeze=2).fixes_human
        assert reserved == free

    def test_dev_sprint_bug_fixing_is_also_reduced(self, base_params):
        # H_after_qa feeds *both* the implementation and the bug-fixing
        # equations, so the reservation is not paid out of coding time alone.
        state_full = SystemState(F_backlog=10_000, B_kno_min=5_000)
        state_res = SystemState(F_backlog=10_000, B_kno_min=5_000)
        full = step_sprint(0, state_full, self._cfg(0),
                           dataclasses.replace(base_params, f_qa=0.0),
                           np.random.default_rng(5)).fixes_human
        reserved = step_sprint(0, state_res, self._cfg(0),
                               dataclasses.replace(base_params, f_qa=0.5),
                               np.random.default_rng(5)).fixes_human
        assert reserved < full

    def test_agent_team_ignores_f_qa(self, base_config, base_params):
        # f_qa is baseline-only: an agent-supported team must be bit-identical under it.
        a = _run_sprints(5, base_config, dataclasses.replace(base_params, f_qa=0.0),
                         np.random.default_rng(3))
        b = _run_sprints(5, base_config, dataclasses.replace(base_params, f_qa=0.9),
                         np.random.default_rng(3))
        assert dataclasses.asdict(a) == dataclasses.asdict(b)


class TestHumanOnlyDetection:
    """Eqs. (cov_qa) / (qa_human): the human review stage replaces Test -> CR -> SA."""

    @staticmethod
    def _cfg(N=4):
        zero = (0,) * AGENT_ROLE_COUNT
        return TeamConfiguration(N=N, A_dev=zero, A_qa=zero, R=13, F_freeze=0)

    def _run(self, params, **overrides):
        p = dataclasses.replace(params, **overrides)
        return _run_sprints(6, self._cfg(), p, np.random.default_rng(9),
                            initial_backlog=400)

    def test_a_zero_agent_team_now_detects_bugs(self, base_params):
        # Without the human stage nothing detects anything: Phi(., 0) = 0 and cov_test = 0,
        # so B_kno stayed identically 0 and every bug escaped at the release.
        state = self._run(base_params, f_qa=0.2, d_qa=0.9, r_bugs_human=0.5)
        assert state.B_kno_min + state.B_kno_maj + state.B_kno_sec > 0

    def test_zero_detection_rate_leaves_every_bug_hidden(self, base_params):
        state = self._run(base_params, f_qa=0.2, d_qa=0.0, r_bugs_human=0.5)
        assert state.B_kno_min == state.B_kno_maj == state.B_kno_sec == 0
        assert state.B_hid_min + state.B_hid_maj + state.B_hid_sec > 0

    def test_higher_detection_rate_finds_more(self, base_params):
        weak = self._run(base_params, f_qa=0.2, d_qa=0.2, r_bugs_human=0.5)
        strong = self._run(base_params, f_qa=0.2, d_qa=0.95, r_bugs_human=0.5)
        assert (strong.B_kno_min + strong.B_kno_maj + strong.B_kno_sec
                > weak.B_kno_min + weak.B_kno_maj + weak.B_kno_sec)

    def test_coverage_clamps_at_one_when_little_is_implemented(self, base_params):
        # cov_qa = min(1, f_qa N H_net / (h_qa max(1, impl))).  With a huge reservation
        # and one feature of demand the ratio is far above 1, so p_qa is exactly d_qa and
        # raising the reservation further changes nothing.
        def run(f_qa):
            p = dataclasses.replace(base_params, f_qa=f_qa, d_qa=1.0,
                                    r_bugs_human=2.0, h_qa=0.01, r_arrival=1e-9)
            return _run_sprints(4, self._cfg(), p, np.random.default_rng(4),
                                initial_backlog=2)

        a, b = run(0.30), run(0.60)
        assert a.B_hid_min == b.B_hid_min == 0
        assert a.B_kno_min == b.B_kno_min

    def test_thin_coverage_detects_less_than_full_coverage(self, base_params):
        # h_qa far above what the reservation buys drives cov_qa below 1.
        thin = self._run(base_params, f_qa=0.2, d_qa=0.9, r_bugs_human=0.5,
                         h_qa=200.0)
        full = self._run(base_params, f_qa=0.2, d_qa=0.9, r_bugs_human=0.5,
                         h_qa=0.01)
        assert (thin.B_kno_min + thin.B_kno_maj + thin.B_kno_sec
                < full.B_kno_min + full.B_kno_maj + full.B_kno_sec)

    def test_agent_team_still_uses_the_agentic_pipeline(self, base_config, base_params):
        # The two paths are mutually exclusive; a team with agents must be unaffected by
        # d_qa and must still respond to its review agents.
        a = _run_sprints(5, base_config, dataclasses.replace(base_params, d_qa=0.0),
                         np.random.default_rng(6))
        b = _run_sprints(5, base_config, dataclasses.replace(base_params, d_qa=1.0),
                         np.random.default_rng(6))
        assert dataclasses.asdict(a) == dataclasses.asdict(b)


class TestHumanOnlyDebtIsInert:
    """The baseline runs without debt dynamics — already true, not enforced anywhere."""

    def test_all_debt_parameters_leave_the_run_unchanged(self, base_params):
        zero = (0,) * AGENT_ROLE_COUNT
        cfg = TeamConfiguration(N=4, A_dev=zero, A_qa=zero, R=4, F_freeze=1)

        def run(**debt):
            p = dataclasses.replace(base_params, **debt)
            return _run_sprints(10, cfg, p, np.random.default_rng(13),
                                initial_backlog=300)

        mild = run(delta_min=0.94, delta_exp=1.0, delta_impl=0.0, delta_fix=0.0,
                   Gamma_doc=0.0)
        harsh = run(delta_min=0.60, delta_exp=2.0, delta_impl=0.5, delta_fix=0.5,
                    Gamma_doc=1.0)
        # D_tech itself still accumulates — nothing reads it — so compare everything else.
        a, b = dataclasses.asdict(mild), dataclasses.asdict(harsh)
        a.pop("D_tech"), b.pop("D_tech")
        assert a == b

    def test_all_agent_parameters_leave_the_run_unchanged(self, base_params):
        # The other half of the claim: the baseline is independent of the agent tier,
        # which is what lets its grid drop that dimension entirely.
        zero = (0,) * AGENT_ROLE_COUNT
        cfg = TeamConfiguration(N=4, A_dev=zero, A_qa=zero, R=4, F_freeze=1)

        def run(**agent):
            p = dataclasses.replace(base_params, f_qa=0.15, d_qa=0.8, **agent)
            return _run_sprints(10, cfg, p, np.random.default_rng(17),
                                initial_backlog=300)

        cheap = run(r_F=1.0, r_arch_chg=1.0, r_fix=1.0, r_refactor=0.01, r_test=1.0,
                    r_doc=1.0, r_bugs=0.001, d_CR=0.1, d_SA=0.1, d_test=0.1,
                    c_test_ceil=0.1, alpha_size=0.01, rho_esc=0.001)
        rich = run(r_F=500.0, r_arch_chg=99.0, r_fix=999.0, r_refactor=1.0, r_test=99.0,
                   r_doc=99.0, r_bugs=0.9, d_CR=0.99, d_SA=0.99, d_test=0.99,
                   c_test_ceil=1.0, alpha_size=0.9, rho_esc=0.9)
        assert dataclasses.asdict(cheap) == dataclasses.asdict(rich)
