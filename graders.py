"""
graders.py — Energy Grid Balancing Task Graders
Real-world validated against NERC, EIA, and historical events.
Scores 0.0 – 1.0 with full component breakdown.
"""
from __future__ import annotations
from typing import Any


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ── Task Easy — Morning Ramp ──────────────────────────────────────────────────

def _grade_easy(state: dict) -> dict:
    """
    Task: Normal day with afternoon wind lull.
    Pass = 0.60.  NERC standard uptime ≥ 99.97% (we use ≥ 95% for AI agent).

    Components:
        uptime_score      (40%) → % intervals without blackout
        frequency_score   (30%) → avg Hz deviation from 50.0
        cost_score        (20%) → generation cost efficiency
        renewable_score   (10%) → % renewable in mix
    """
    kpis   = state["kpis"]
    impact = state["human_impact"]

    uptime   = _clamp((kpis["uptime_percent"] - 80.0) / 20.0)
    freq_dev = kpis["avg_frequency_dev_hz"]
    freq     = _clamp(1.0 - freq_dev / 0.5)
    cost     = _clamp(1.0 - kpis["total_cost_usd"] / 500_000)
    renew    = _clamp(kpis["renewable_percent"] / 60.0)

    final = uptime*0.40 + freq*0.30 + cost*0.20 + renew*0.10

    return {
        "task_id": "task_easy", "final_score": round(final, 4),
        "components": {
            "uptime_score":    round(uptime, 4),
            "frequency_score": round(freq,   4),
            "cost_score":      round(cost,   4),
            "renewable_score": round(renew,  4),
        },
        "passed": final >= 0.60,
        "rationale": (
            f"Uptime: {kpis['uptime_percent']:.1f}% | "
            f"Freq dev: {freq_dev:.3f} Hz | "
            f"Cost: ${kpis['total_cost_usd']:,.0f} | "
            f"Renewable: {kpis['renewable_percent']:.1f}% | "
            f"Homes protected: {int(1_248_000 - impact['homes_at_risk']):,}"
        ),
        "real_world_validation": state.get("real_world_validation", {}),
    }


# ── Task Medium — Heat Wave + Solar Cliff ─────────────────────────────────────

def _grade_medium(state: dict) -> dict:
    """
    Task: California Sep 2022 style — heatwave + solar cliff.
    Real outcome: California avoided blackouts by <200 MW margin.
    Pass = 0.60.

    Components:
        uptime_score        (35%) → must beat California 2022 (91%)
        disruption_response (25%) → did agent react fast enough?
        cost_score          (20%) → avoid expensive emergency imports
        human_impact_score  (20%) → homes/hospitals protected
    """
    kpis   = state["kpis"]
    impact = state["human_impact"]

    # Uptime — benchmark: California 2022 actual = 91%
    uptime = _clamp((kpis["uptime_percent"] - 70.0) / 30.0)

    # Disruption response — inferred from critical/warning intervals
    bad_intervals = kpis["warning_intervals"] + kpis["critical_intervals"] * 2
    disruption = _clamp(1.0 - bad_intervals / 40.0)

    # Cost — penalise heavy emergency imports
    import_penalty = kpis["total_import_mwh"] / 1000.0
    cost = _clamp(1.0 - import_penalty / 5.0 - kpis["total_cost_usd"] / 600_000)

    # Human impact
    homes_protected_pct = 1.0 - impact["homes_at_risk"] / max(1_248_000, 1)
    human = _clamp(homes_protected_pct)

    final = uptime*0.35 + disruption*0.25 + cost*0.20 + human*0.20

    return {
        "task_id": "task_medium", "final_score": round(final, 4),
        "components": {
            "uptime_score":        round(uptime,     4),
            "disruption_response": round(disruption, 4),
            "cost_score":          round(cost,        4),
            "human_impact_score":  round(human,       4),
        },
        "passed": final >= 0.60,
        "rationale": (
            f"Uptime: {kpis['uptime_percent']:.1f}% (California 2022 actual: 91%) | "
            f"Warning+Critical: {bad_intervals} intervals | "
            f"Imports: {kpis['total_import_mwh']:.0f} MWh | "
            f"Homes at risk: {impact['homes_at_risk']:,} | "
            f"Blackout minutes: {impact['blackout_minutes']:.0f}"
        ),
        "real_world_validation": state.get("real_world_validation", {}),
    }


# ── Task Hard — Texas 2021 Cascade ────────────────────────────────────────────

def _grade_hard(state: dict) -> dict:
    """
    Task: Texas Feb 2021 style — cascading failure.
    Real outcome: 4.5M homes lost power, 246 deaths, $195B damage.
    Pass = 0.60.

    Components:
        survival_score    (30%) → did agent avoid full blackout?
        uptime_score      (30%) → must beat Texas 2021 (67%)
        resilience_score  (25%) → managed cascading failures
        co2_score         (15%) → clean generation under stress
    """
    kpis   = state["kpis"]
    impact = state["human_impact"]

    # Survival — did agent avoid complete blackout?
    survival = _clamp(1.0 - kpis["blackout_intervals"] / 30.0)

    # Uptime — benchmark: Texas 2021 actual = 67%
    uptime = _clamp((kpis["uptime_percent"] - 50.0) / 50.0)

    # Resilience — handled cascade without total shed
    shed_ratio = kpis["total_shed_mw"] / (96 * 300)  # vs max possible shed
    resilience = _clamp(1.0 - shed_ratio)

    # CO2 — maintained some renewable mix under pressure
    co2 = _clamp(kpis["renewable_percent"] / 40.0)

    final = survival*0.30 + uptime*0.30 + resilience*0.25 + co2*0.15

    return {
        "task_id": "task_hard", "final_score": round(final, 4),
        "components": {
            "survival_score":   round(survival,   4),
            "uptime_score":     round(uptime,     4),
            "resilience_score": round(resilience, 4),
            "co2_score":        round(co2,         4),
        },
        "passed": final >= 0.60,
        "rationale": (
            f"Uptime: {kpis['uptime_percent']:.1f}% (Texas 2021 actual: 67%) | "
            f"Blackout intervals: {kpis['blackout_intervals']} | "
            f"Total shed: {kpis['total_shed_mw']:.0f} MW | "
            f"Renewable: {kpis['renewable_percent']:.1f}% | "
            f"CO2: {impact['co2_saved_kg']:,.0f} kg saved | "
            f"Blackout minutes: {impact['blackout_minutes']:.0f}"
        ),
        "real_world_validation": state.get("real_world_validation", {}),
    }


_GRADERS = {
    "task_easy":   _grade_easy,
    "task_medium": _grade_medium,
    "task_hard":   _grade_hard,
}


def grade(task_id: str, state: dict) -> dict:
    fn = _GRADERS.get(task_id)
    if not fn:
        raise ValueError(f"Unknown task_id: {task_id}")
    return fn(state)
