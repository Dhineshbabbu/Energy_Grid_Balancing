"""
scripts/baseline_inference.py — Baseline agents for Energy Grid OpenEnv

Agents:
    RandomAgent      — random valid actions (lower bound)
    HeuristicAgent   — rule-based dispatch policy (mid baseline)
    ReactiveAgent    — forecast-aware disruption-reactive (upper bound)

Run:
    python scripts/baseline_inference.py
"""
from __future__ import annotations
import sys, json, random, statistics
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energy_grid_balancing_environment import EnergyGridEnv
from graders import grade

TASKS = ["task_easy", "task_medium", "task_hard"]
SEEDS = [42, 123, 777]

ACTIONS = [
    "wait", "increase_gas_small", "increase_gas_medium", "increase_gas_large",
    "decrease_gas_small", "decrease_gas_medium", "activate_peaker",
    "deactivate_peaker", "charge_battery", "discharge_battery_half",
    "discharge_battery_full", "buy_from_neighbor", "sell_to_neighbor",
    "shed_load_small", "shed_load_large",
]


# ── RandomAgent ───────────────────────────────────────────────────────────────

class RandomAgent:
    name = "RandomAgent"
    _WEIGHTS = {
        "wait":                  0.40,
        "increase_gas_small":    0.15,
        "increase_gas_medium":   0.08,
        "increase_gas_large":    0.04,
        "decrease_gas_small":    0.08,
        "charge_battery":        0.08,
        "discharge_battery_half":0.06,
        "activate_peaker":       0.04,
        "buy_from_neighbor":     0.04,
        "shed_load_small":       0.02,
        "decrease_gas_medium":   0.01,
    }

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs: dict) -> dict:
        choices = list(self._WEIGHTS.keys())
        weights = list(self._WEIGHTS.values())
        action  = self.rng.choices(choices, weights=weights, k=1)[0]
        return {"action_type": action, "params": {}}


# ── HeuristicAgent ────────────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Rule-based economic dispatch policy:
      - If grid short (balance < -100 MW) → increase generation
      - If grid long  (balance > +200 MW) → decrease or charge battery
      - If blackout/critical → emergency response
      - Otherwise → optimise cost (reduce expensive generation)
    """
    name = "HeuristicAgent"

    def act(self, obs: dict) -> dict:
        balance = obs["balance_mw"]
        freq    = obs["frequency_hz"]
        status  = obs["grid_status"]
        soc     = obs["battery"]["soc"]
        hour    = obs["hour"]

        # CRITICAL / BLACKOUT — emergency response
        if status in ("blackout", "critical"):
            if soc > 0.20:
                return {"action_type": "discharge_battery_full", "params": {}}
            return {"action_type": "activate_peaker", "params": {}}

        # WARNING — act quickly
        if status == "warning":
            if balance < 0:
                if soc > 0.30:
                    return {"action_type": "discharge_battery_half", "params": {}}
                return {"action_type": "increase_gas_medium", "params": {}}
            else:
                if soc < 0.80:
                    return {"action_type": "charge_battery", "params": {}}

        # NORMAL — economic dispatch
        if balance < -150:
            return {"action_type": "increase_gas_medium", "params": {}}
        elif balance < -50:
            return {"action_type": "increase_gas_small", "params": {}}
        elif balance > 300 and soc < 0.90:
            return {"action_type": "charge_battery", "params": {}}
        elif balance > 400:
            return {"action_type": "sell_to_neighbor", "params": {}}
        elif balance > 200:
            return {"action_type": "decrease_gas_small", "params": {}}

        return {"action_type": "wait", "params": {}}


# ── ReactiveAgent ─────────────────────────────────────────────────────────────

class ReactiveAgent:
    """
    Extends HeuristicAgent with:
      - Disruption detection → pre-emptive generation increase
      - Forecast awareness  → prepare for evening peak / solar cliff
      - Battery management  → preserve charge for peak periods
      - Cost optimisation   → avoid peaker unless necessary
    """
    name = "ReactiveAgent"

    def act(self, obs: dict) -> dict:
        balance      = obs["balance_mw"]
        freq         = obs["frequency_hz"]
        status       = obs["grid_status"]
        soc          = obs["battery"]["soc"]
        hour         = obs["hour"]
        disruptions  = obs.get("active_disruptions", [])
        forecast     = obs.get("forecast", {})
        sources      = obs.get("sources", {})
        has_disruption = len(disruptions) > 0

        # 1. BLACKOUT / CRITICAL — maximum response
        if status in ("blackout", "critical"):
            if soc > 0.15:
                return {"action_type": "discharge_battery_full", "params": {}}
            if not sources.get("peaker", {}).get("is_online"):
                return {"action_type": "activate_peaker", "params": {}}
            return {"action_type": "increase_gas_large", "params": {}}

        # 2. WARNING
        if status == "warning":
            if balance < -100 and soc > 0.25:
                return {"action_type": "discharge_battery_half", "params": {}}
            if balance < -100:
                return {"action_type": "increase_gas_medium", "params": {}}
            if balance > 0 and soc < 0.85:
                return {"action_type": "charge_battery", "params": {}}

        # 3. Disruption detected — pre-emptive action
        if has_disruption and balance < 50:
            if soc > 0.40:
                return {"action_type": "discharge_battery_half", "params": {}}
            return {"action_type": "increase_gas_medium", "params": {}}

        # 4. Forecast-aware: evening peak approaching (5–8pm)
        if 16 <= hour <= 19:
            next_demand = forecast.get("next_interval_demand_mw", 0)
            if next_demand > obs["demand_mw"] + 100:
                if balance < 200:
                    return {"action_type": "increase_gas_small", "params": {}}

        # 5. Solar cliff approaching (after 5pm, solar dropping)
        if hour >= 17 and sources.get("solar", {}).get("current_mw", 0) < 100:
            if balance < 100:
                return {"action_type": "increase_gas_medium", "params": {}}

        # 6. Charge battery during solar peak (10am–2pm)
        if 10 <= hour <= 14 and soc < 0.80 and balance > 150:
            return {"action_type": "charge_battery", "params": {}}

        # 7. Normal economic dispatch
        if balance < -100:
            return {"action_type": "increase_gas_medium", "params": {}}
        elif balance < -30:
            return {"action_type": "increase_gas_small", "params": {}}
        elif balance > 350 and soc < 0.90:
            return {"action_type": "charge_battery", "params": {}}
        elif balance > 250:
            return {"action_type": "decrease_gas_small", "params": {}}

        # 8. Deactivate peaker if no longer needed
        if sources.get("peaker", {}).get("is_online") and balance > 300:
            return {"action_type": "deactivate_peaker", "params": {}}

        return {"action_type": "wait", "params": {}}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(agent, task_id: str, seed: int) -> dict:
    env = EnergyGridEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    total_reward  = 0.0
    steps         = 0
    reward_detail = []

    while True:
        action = agent.act(obs)
        result = env.step(action)
        obs    = result["observation"]
        total_reward += result["reward"]
        steps        += 1
        reward_detail.append({
            "step":      steps,
            "reward":    result["reward"],
            "breakdown": result.get("reward_breakdown", {}),
        })
        if result["done"]: break

    final_state   = env.state()
    grader_result = grade(task_id, final_state)

    return {
        "agent":             agent.name,
        "task_id":           task_id,
        "seed":              seed,
        "steps":             steps,
        "cumulative_reward": round(total_reward, 4),
        "grader_score":      grader_result["final_score"],
        "passed":            grader_result["passed"],
        "components":        grader_result["components"],
        "kpis":              final_state["kpis"],
        "human_impact":      final_state["human_impact"],
        "real_world":        final_state["real_world_validation"]["agent_vs_real_world"],
        "reward_detail":     reward_detail,
    }


# ── Main benchmark ────────────────────────────────────────────────────────────

def main():
    agents  = [RandomAgent(seed=0), HeuristicAgent(), ReactiveAgent()]
    results = []

    print(f"\n{'═'*72}")
    print(f"  Energy Grid Balancing OpenEnv — Baseline Benchmark")
    print(f"  Real-world validated vs Texas 2021, California 2022, NERC standards")
    print(f"{'═'*72}")

    for task_id in TASKS:
        print(f"\n▶  Task: {task_id}")
        print(f"{'─'*60}")

        for agent in agents:
            scores = []
            for seed in SEEDS:
                r = run_episode(agent, task_id, seed)
                scores.append(r["grader_score"])
                results.append(r)

                bd_list = r["reward_detail"]
                avg_bal  = sum(x["breakdown"].get("balance_signal",0)    for x in bd_list)/len(bd_list)
                avg_freq = sum(x["breakdown"].get("frequency_penalty",0)  for x in bd_list)/len(bd_list)
                avg_renew= sum(x["breakdown"].get("renewable_bonus",0)    for x in bd_list)/len(bd_list)

                print(
                    f"  {agent.name:<18} seed={seed} "
                    f"score={r['grader_score']:.4f} ({'✅' if r['passed'] else '❌'}) "
                    f"| uptime={r['kpis']['uptime_percent']:.1f}% "
                    f"| renew={r['kpis']['renewable_percent']:.1f}% "
                    f"| blackout_min={r['human_impact']['blackout_minutes']:.0f}"
                )

            mean = statistics.mean(scores)
            std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"  {'':18}       avg={mean:.4f} ±{std:.4f}\n")

    print(f"{'═'*72}")
    print(f"  Summary — Real World Validation")
    print(f"{'═'*72}")
    print(f"  {'Agent':<18} {'Task':<14} {'Score':>7}  {'Uptime':>8}  "
          f"{'Renewable':>10}  {'Blackout(min)':>14}")
    print(f"  {'─'*18} {'─'*14} {'─'*7}  {'─'*8}  {'─'*10}  {'─'*14}")

    for r in results:
        print(
            f"  {r['agent']:<18} {r['task_id']:<14} "
            f"{r['grader_score']:>7.4f}  "
            f"{r['kpis']['uptime_percent']:>7.1f}%  "
            f"{r['kpis']['renewable_percent']:>9.1f}%  "
            f"{r['human_impact']['blackout_minutes']:>14.0f}"
        )

    print()
    # Print real-world comparison for last hard task result
    hard_results = [r for r in results if r["task_id"] == "task_hard"]
    if hard_results:
        best = max(hard_results, key=lambda x: x["grader_score"])
        print(f"  Best agent on task_hard: {best['agent']} (score={best['grader_score']:.4f})")
        print(f"  Real-world comparison:")
        for k, v in best["real_world"].items():
            print(f"    {v}")

    out = Path(__file__).parent / "baseline_results.json"
    slim = [{k: v for k, v in r.items() if k != "reward_detail"} for r in results]
    out.write_text(json.dumps(slim, indent=2))
    print(f"\n📁 Results saved → {out}\n")


if __name__ == "__main__":
    main()
