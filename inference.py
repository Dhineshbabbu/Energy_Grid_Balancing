"""
inference.py — LLM Agent for Energy Grid Balancing OpenEnv
===========================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier.
    HF_TOKEN       Your Hugging Face / API key.

Run:
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \\
    HF_TOKEN=hf_xxxx \\
    python inference.py --task task_medium --seed 42
"""
from __future__ import annotations
import os, json, argparse, textwrap
from typing import Any
from openai import OpenAI
from energy_grid_balancing_environment import EnergyGridEnv
from graders import grade

# ── Mandatory config ──────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS   = 96
TEMPERATURE = 0.05        # ✅ very low — decisive, not creative
MAX_TOKENS  = 80          # ✅ short — we only need one JSON line
FALLBACK    = {"action_type": "wait", "params": {}}
DEBUG       = os.getenv("DEBUG", "0") == "1"

VALID_ACTIONS = {
    "wait", "increase_gas_small", "increase_gas_medium", "increase_gas_large",
    "decrease_gas_small", "decrease_gas_medium", "activate_peaker",
    "deactivate_peaker", "charge_battery", "discharge_battery_half",
    "discharge_battery_full", "buy_from_neighbor", "sell_to_neighbor",
    "shed_load_small", "shed_load_large",
}

# ── System prompt — strict rule-based, no ambiguity ──────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an AI electricity grid operator.
Output ONLY a single JSON object — no explanation, no text, just JSON.

═══ STRICT DECISION RULES — follow IN ORDER, stop at first match ═══

RULE 1 — EMERGENCY (status=BLACKOUT or status=CRITICAL or freq < 49.5):
  if peaker_online=false  → {"action_type": "activate_peaker", "params": {}}
  if battery_soc > 0.20   → {"action_type": "discharge_battery_full", "params": {}}
  else                    → {"action_type": "buy_from_neighbor", "params": {}}

RULE 2 — WARNING (status=WARNING or freq < 49.8 or balance < -100):
  if balance < -200       → {"action_type": "increase_gas_large", "params": {}}
  if balance < -100       → {"action_type": "increase_gas_medium", "params": {}}
  if battery_soc > 0.30   → {"action_type": "discharge_battery_half", "params": {}}
  else                    → {"action_type": "increase_gas_small", "params": {}}

RULE 3 — DISRUPTION ACTIVE (disruptions list is not empty):
  if balance < -200       → {"action_type": "increase_gas_large", "params": {}}
  if balance < -50        → {"action_type": "increase_gas_medium", "params": {}}
  if balance < 0          → {"action_type": "increase_gas_small", "params": {}}
  if balance > 100        → {"action_type": "charge_battery", "params": {}}
  else                    → {"action_type": "wait", "params": {}}

RULE 4 — SURPLUS (balance > 200):
  if battery_soc < 0.85   → {"action_type": "charge_battery", "params": {}}
  if battery_soc >= 0.85  → {"action_type": "sell_to_neighbor", "params": {}}

RULE 5 — SMALL SURPLUS (balance 50 to 200):
  if battery_soc < 0.70 and hour between 10-14  → {"action_type": "charge_battery", "params": {}}
  if gas_current > 300   → {"action_type": "decrease_gas_small", "params": {}}
  else                   → {"action_type": "wait", "params": {}}

RULE 6 — SMALL DEFICIT (balance -100 to -10):
  if hour between 17-21 and battery_soc > 0.25 → {"action_type": "discharge_battery_half", "params": {}}
  else → {"action_type": "increase_gas_small", "params": {}}

RULE 7 — BALANCED (balance -10 to +50):
  → {"action_type": "wait", "params": {}}

═══ OUTPUT FORMAT ═══
{"action_type": "<one of the valid actions>", "params": {}}
""").strip()


# ── Prompt builder — minimal, focused on decision variables ──────────────────

def build_user_prompt(step: int, obs: dict,
                      last_reward: float | None,
                      last_breakdown: dict | None,
                      last_action: str) -> str:
    """
    Tightly structured prompt that maps directly to the decision rules.
    Gives LLM exactly the variables it needs — nothing more.
    """
    battery  = obs.get("battery", {})
    sources  = obs.get("sources", {})
    dis      = obs.get("active_disruptions", [])
    kpis     = obs.get("kpis", {})
    forecast = obs.get("forecast", {})

    status        = obs.get("grid_status", "normal").upper()
    freq          = obs.get("frequency_hz", 50.0)
    balance       = obs.get("balance_mw", 0.0)
    hour          = obs.get("hour", 0.0)
    soc           = battery.get("soc", 0.0)
    batt_mw       = battery.get("available_discharge_mw", 0)
    peaker_online = sources.get("peaker", {}).get("is_online", False)
    gas_current   = sources.get("gas", {}).get("current_mw", 0)
    disruptions   = "YES — " + ", ".join(
        f"{d['disruption_type']}(sev={d['severity']:.0%})" for d in dis
    ) if dis else "NO"

    # Urgency line — maps directly to rules
    if status in ("BLACKOUT", "CRITICAL") or freq < 49.5:
        rule_hint = "→ APPLY RULE 1 (EMERGENCY)"
    elif status == "WARNING" or freq < 49.8 or balance < -100:
        rule_hint = "→ APPLY RULE 2 (WARNING)"
    elif dis:
        rule_hint = "→ APPLY RULE 3 (DISRUPTION)"
    elif balance > 200:
        rule_hint = "→ APPLY RULE 4 (SURPLUS)"
    elif 50 <= balance <= 200:
        rule_hint = "→ APPLY RULE 5 (SMALL SURPLUS)"
    elif -100 <= balance < -10:
        rule_hint = "→ APPLY RULE 6 (SMALL DEFICIT)"
    else:
        rule_hint = "→ APPLY RULE 7 (BALANCED) → wait"

    last_reward_txt = (
        f"{last_reward:+.4f} "
        f"(bal={last_breakdown.get('balance_signal',0):+.3f} "
        f"freq={last_breakdown.get('frequency_penalty',0):+.3f} "
        f"cost={last_breakdown.get('cost_efficiency',0):+.3f} "
        f"renew={last_breakdown.get('renewable_bonus',0):+.3f})"
        if last_reward is not None and last_breakdown else "n/a"
    )

    return f"""STEP {step}/96  TIME {hour:.2f}h  {rule_hint}

DECISION VARIABLES:
  status        = {status}
  freq          = {freq:.3f} Hz
  balance       = {balance:+.1f} MW
  battery_soc   = {soc:.2f} ({soc:.0%})
  battery_mw    = {batt_mw:.0f} MW available
  peaker_online = {peaker_online}
  gas_current   = {gas_current:.0f} MW
  hour          = {hour:.2f}
  disruptions   = {disruptions}

GRID STATE:
  supply = {obs.get('supply_mw',0):.0f} MW
  demand = {obs.get('demand_mw',0):.0f} MW
  uptime = {kpis.get('uptime_percent',100):.1f}%
  renewable = {kpis.get('renewable_percent',0):.1f}%
  blackout_intervals = {kpis.get('blackout_intervals',0)}

LAST REWARD: {last_reward_txt}
LAST ACTION: {last_action}
NEXT DEMAND FORECAST: {forecast.get('next_interval_demand_mw',0):.0f} MW

{rule_hint}
Apply the matching rule and output ONLY the JSON:"""


# ── JSON parser ───────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def parse_action(response_text: str) -> dict:
    if not response_text:
        return FALLBACK
    js = _extract_json(response_text)
    if not js:
        if DEBUG:
            print(f"  [parse] No JSON in: {response_text!r}")
        return FALLBACK
    try:
        action = json.loads(js)
    except json.JSONDecodeError:
        return FALLBACK
    at = action.get("action_type", "")
    if at not in VALID_ACTIONS:
        if DEBUG:
            print(f"  [parse] Invalid: {at!r}")
        return FALLBACK
    return {"action_type": at, "params": {}}


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default="task_easy",
                        choices=["task_easy", "task_medium", "task_hard"])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global DEBUG
    if args.debug:
        DEBUG = True

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = EnergyGridEnv()
    obs = env.reset(task_id=args.task, seed=args.seed)

    print(f"\n{'═'*72}")
    print(f"  Energy Grid Balancing OpenEnv — LLM Inference")
    print(f"  Task : {args.task}  |  Seed : {args.seed}")
    print(f"  Model: {MODEL_NAME}  |  Temp: {TEMPERATURE}")
    print(f"{'═'*72}\n")

    total_reward:   float        = 0.0
    last_reward:    float | None = None
    last_breakdown: dict | None  = None
    last_action:    str          = "none"

    for step in range(1, MAX_STEPS + 1):

        user_prompt = build_user_prompt(
            step, obs, last_reward, last_breakdown, last_action)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM] Failed ({exc}). Using fallback.")
            response_text = ""

        if DEBUG:
            print(f"  [LLM raw] {response_text!r}")

        action      = parse_action(response_text)
        last_action = action["action_type"]

        # ── Print step summary ──
        status = obs.get("grid_status","?")
        freq   = obs.get("frequency_hz", 50)
        bal    = obs.get("balance_mw", 0)
        soc    = obs.get("battery", {}).get("soc", 0)
        print(f"t={obs.get('hour',0):.2f}h | "
              f"step={step:>2} | "
              f"{status:<8} | "
              f"freq={freq:.2f}Hz | "
              f"bal={bal:+.0f}MW | "
              f"SoC={soc:.0%} | "
              f"→ {last_action}")

        try:
            result = env.step(action)
        except RuntimeError as e:
            print(f"  [env] Error: {e}")
            break

        last_reward    = result["reward"]
        last_breakdown = result.get("reward_breakdown", {})
        obs            = result["observation"]
        total_reward  += last_reward

        bd = last_breakdown or {}
        print(f"         reward={last_reward:+.4f} "
              f"[bal={bd.get('balance_signal',0):+.3f} "
              f"freq={bd.get('frequency_penalty',0):+.3f} "
              f"cost={bd.get('cost_efficiency',0):+.3f} "
              f"renew={bd.get('renewable_bonus',0):+.3f} "
              f"blackout={bd.get('blackout_penalty',0):+.3f}] "
              f"uptime={obs['kpis'].get('uptime_percent',100):.1f}%")

        if result.get("done"):
            print(f"\n  Episode complete at step {step}.")
            break

    # ── Grade + real-world validation ──────────────────────────────────────
    final_state = env.state()
    grader      = grade(args.task, final_state)
    rwv         = final_state["real_world_validation"]["agent_vs_real_world"]

    print(f"\n{'═'*72}")
    print(f"  FINAL RESULTS")
    print(f"{'─'*72}")
    print(f"  Score         : {grader['final_score']:.4f}  "
          f"{'✅ PASSED' if grader['passed'] else '❌ FAILED'}")
    print(f"  Total reward  : {total_reward:.4f}")
    print(f"  Uptime        : {final_state['kpis']['uptime_percent']:.2f}%")
    print(f"  Renewable     : {final_state['kpis']['renewable_percent']:.1f}%")
    print(f"  Blackout mins : {final_state['human_impact']['blackout_minutes']:.0f}")
    print(f"  CO2 saved     : {final_state['human_impact']['co2_saved_kg']:,.0f} kg")
    print(f"\n  Component scores:")
    for k, v in grader["components"].items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"    {k:<22} {v:.4f}  [{bar}]")
    print(f"\n  Real-World Validation:")
    for v in rwv.values():
        print(f"    {v}")
    print(f"\n  Rationale: {grader['rationale']}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()