"""Tests for src/feasibility.py — single human-capacity check + feasible path."""

import pytest

from src.config import AGENT_ROLE_COUNT, TeamConfiguration, net_capacity
from src.feasibility import check_feasibility


def _cfg(comp_dict, N=2, R=4, F_freeze=0):
    A = [0] * AGENT_ROLE_COUNT
    for k, v in comp_dict.items():
        A[k - 1] = v
    return TeamConfiguration(N=N, A_dev=tuple(A), R=R, F_freeze=F_freeze)


class TestFeasiblePath:
    def test_feasible_returns_true(self, base_config, base_params):
        report = check_feasibility(base_config, base_params)
        assert report.feasible
        assert report.failed_step is None


class TestStep1MonitoringCapacity:
    def test_too_many_supervised_agents_infeasible(self, base_params):
        # Gamma_doc=0 so doc_cov assumption is irrelevant; 12 supervised agents over
        # N=2 developers cost 6 * 2h = 12h of monitoring each > H_net ≈ 9h
        cfg = _cfg({1: 4, 3: 3, 4: 3, 7: 2}, N=2)
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0, Gamma_doc=0.0)
        report = check_feasibility(cfg, p)
        assert not report.feasible
        assert report.failed_step == 1

    def test_autonomous_roles_never_cause_infeasibility(self, base_params):
        """K_aut = {4,5,6} consumes no monitoring capacity at all (paper §3.3)."""
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0, Gamma_doc=0.0)
        # 4 supervised agents over N=2 developers = 2 * 2h = 4h < H_net ≈ 9h
        lean = _cfg({1: 2, 3: 1, 7: 1}, N=2)
        assert check_feasibility(lean, p).feasible
        # Same supervised load, but 60 autonomous agents on top
        loaded = _cfg({1: 2, 3: 1, 7: 1, 4: 20, 5: 20, 6: 20}, N=2)
        assert check_feasibility(loaded, p).feasible

    def test_refactoring_agents_do_not_tip_a_team_over_the_gate(self, base_params):
        """Since commit "make refactoring unsupervised", role 4 is outside n_sup.

        The team below sits just under the bound on its supervised roles; adding
        Refactoring Agents used to push it over and must now change nothing.
        """
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0, Gamma_doc=0.0)
        # 4 supervised agents over N=1: 4 * 2h = 8h < H_net = 10h, but 5 would be 10h.
        borderline = _cfg({1: 2, 3: 1, 7: 1}, N=1)
        assert check_feasibility(borderline, p).feasible
        assert not check_feasibility(_cfg({1: 3, 3: 1, 7: 1}, N=1), p).feasible
        assert check_feasibility(_cfg({1: 2, 3: 1, 7: 1, 4: 8}, N=1), p).feasible


class TestCommunicationOverheadUsesHeadcount:
    """H_net = H_capacity - c_comm * ceil(N-1) (paper §3.6, Eq. hnet).

    Communication is a per-person cost that does not shrink when someone invests only
    part of their time, so the peer count follows the headcount ceil(N), not N itself.
    A half-time developer is still one person and still talks to nobody.
    """

    @pytest.mark.parametrize("N, peers", [
        (0.25, 0), (0.75, 0), (1.0, 0), (1.25, 1), (2.0, 1), (2.25, 2), (2.5, 2), (3.0, 2),
    ])
    def test_peer_count_is_the_ceiling(self, N, peers):
        assert net_capacity(40.0, 1.0, N) == 40.0 - peers

    def test_the_feasibility_gate_uses_the_same_formula(self, base_params):
        """N = 2.5 pays for 3 people, so it must be tighter than the linear N-1 = 1.5."""
        from dataclasses import replace
        p = replace(base_params, H_capacity=10.0, H_monitoring_overhead=1.0,
                    Gamma_doc=0.0, c_comm=1.0)
        # H_net = 10 - 2 = 8 under the ceiling, 8.5 under a linear (N-1).  With 20
        # supervised agents over 2.5 FTE, monitoring costs exactly 8 h per FTE, which
        # the ">=" of the gate rejects — but a linear rule would still accept.
        cfg = _cfg({1: 10, 3: 10}, N=2.5)
        report = check_feasibility(cfg, p)
        assert not report.feasible and report.failed_step == 1
        # One agent fewer leaves 7.6 h < 8 h and passes.
        assert check_feasibility(_cfg({1: 10, 3: 9}, N=2.5), p).feasible


class TestQaCompositionIsChecked:
    """Eq. (feas_monitoring) is applied to A_qa too, but only when it is ever used.

    With F_freeze = 0 the QA vector is never applied, and the search leaves an
    arbitrary unexercised composition in those rows, so checking it would reject
    configurations on the strength of agents that are never deployed.
    """

    def _params(self, base_params):
        from dataclasses import replace
        return replace(base_params, H_capacity=10.0, H_monitoring_overhead=2.0,
                       Gamma_doc=0.0)

    def _pair(self, F_freeze):
        """Lean A_dev (feasible) paired with a bloated A_qa (over the bound)."""
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0] = 2                       # 2 supervised over N=2 -> 2h < H_net
        A_qa = [0] * AGENT_ROLE_COUNT
        A_qa[2] = 12                       # 12 supervised over N=2 -> 12h > H_net
        return TeamConfiguration(N=2, A_dev=tuple(A_dev), R=4,
                                 F_freeze=F_freeze, A_qa=tuple(A_qa))

    def test_bloated_qa_composition_is_rejected_when_a_freeze_exists(self, base_params):
        report = check_feasibility(self._pair(F_freeze=1), self._params(base_params))
        assert not report.feasible
        assert report.failed_step == 1
        assert "A_qa" in report.reason

    def test_bloated_qa_composition_is_ignored_without_a_freeze(self, base_params):
        assert check_feasibility(self._pair(F_freeze=0), self._params(base_params)).feasible

    def test_dev_composition_is_still_checked(self, base_params):
        """A_dev is checked regardless of F_freeze, and is named in the reason."""
        A_dev = [0] * AGENT_ROLE_COUNT
        A_dev[0] = 12
        cfg = TeamConfiguration(N=2, A_dev=tuple(A_dev), R=4, F_freeze=0)
        report = check_feasibility(cfg, self._params(base_params))
        assert not report.feasible
        assert "A_dev" in report.reason


class TestHumanOnlyBaseline:
    """A zero-agent team (paper §4.1) has no monitoring load, so it is always feasible."""

    _N_VALUES = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.00, 10.00)

    def test_feasible_at_every_staffing_level_in_the_search_range(self, base_params):
        zero = (0,) * AGENT_ROLE_COUNT
        for N in self._N_VALUES:
            cfg = TeamConfiguration(N=N, A_dev=zero, A_qa=zero, R=4, F_freeze=1)
            report = check_feasibility(cfg, base_params)
            assert report.feasible, f"N={N}: {report.reason}"
            assert report.failed_step is None

    def test_feasible_even_when_monitoring_would_exhaust_any_real_team(self, base_params):
        # The condition is n_sup/N * H_mon * (1 - Gamma_doc) < H_net; with n_sup = 0 the
        # left side is 0 whatever the overhead, so only H_net >= 0 can still bite.
        from dataclasses import replace
        p = replace(base_params, H_monitoring_overhead=1000.0, Gamma_doc=0.0)
        zero = (0,) * AGENT_ROLE_COUNT
        assert check_feasibility(
            TeamConfiguration(N=2, A_dev=zero, A_qa=zero, R=4), p
        ).feasible
