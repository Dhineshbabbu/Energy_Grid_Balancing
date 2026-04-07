"""
server/app.py — Gradio HumanAgent Interface + State Observer
Energy Grid Balancing OpenEnv — Gradio 6.x compatible
"""
from __future__ import annotations
import sys, json, threading, time, uuid, textwrap, re
from pathlib import Path
from datetime import datetime
from typing import Any
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import uvicorn
from server.my_environment_api import app as fastapi_app
from energy_grid_balancing_environment import EnergyGridEnv
from graders import grade


def _start_api():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7861, log_level="error")


def new_session() -> dict:
    return {
        "env": EnergyGridEnv(),
        "obs": {},
        "history": [],
        "step_count": 0,
        "episode_id": "—",
        "running": False,
        "task_id": "task_easy",
    }


STATUS_ICONS = {"normal": "🟢", "warning": "🟡", "critical": "🔴", "blackout": "💀"}
DIS_ICONS = {
    "plant_trip": "🏭",
    "wind_drop": "💨",
    "solar_cloud": "☁️",
    "demand_spike": "📈",
    "transmission_fault": "⚡",
    "fuel_shortage": "⛽",
}


def fmt(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def build_state_md(s: dict) -> str:
    if not s["running"]:
        return "**Status:** ⬜ Not started\n\n*Click Reset Environment to begin.*"

    obs = s["obs"]
    kpi = obs.get("kpis", {})
    impact = obs.get("human_impact", {})
    dis = obs.get("active_disruptions", [])
    sources = obs.get("sources", {})
    battery = obs.get("battery", {})

    status_icon = STATUS_ICONS.get(obs.get("grid_status", "normal"), "🟢")

    dis_line = "**Active Disruption:** None ✅"
    if dis:
        d = dis[0]
        icon = DIS_ICONS.get(d.get("disruption_type", ""), "⚠️")
        dis_line = (
            f"**Active Disruption:** {icon} "
            f"{d.get('disruption_type', '').replace('_', ' ').title()} "
            f"· severity {d.get('severity', 0):.0%} "
            f"· ref: {d.get('real_world_ref', '')}"
        )

    src_lines = []
    for sid, src in sources.items():
        pct = src.get("current_mw", 0) / max(src.get("capacity_mw", 1), 1)
        bar = "█" * int(pct * 8) + "░" * (8 - int(pct * 8))
        renew = "🌿" if src.get("is_renewable") else "🏭"
        src_lines.append(
            f"{renew} `{sid}` [{bar}] **{src.get('current_mw', 0):.0f}**/"
            f"{src.get('capacity_mw', 0):.0f} MW "
            f"{'✅' if src.get('is_online', True) else '🔴 OFFLINE'}"
        )

    soc_bar = "█" * int(battery.get("soc", 0) * 10) + "░" * (10 - int(battery.get("soc", 0) * 10))

    return f"""**Status:** {status_icon} {obs.get('grid_status', '?').upper()}
**Episode:** `{s['episode_id']}`  **Task:** `{s['task_id']}`
**Interval:** {s['step_count']} / 20  **Time:** {obs.get('hour', 0)*0.6:.1f}min

---
### ⚡ Grid
| Metric | Value |
|---|---|
| Frequency | **{obs.get('frequency_hz', 50):.3f} Hz** (target 50.0) |
| Supply | {obs.get('supply_mw', 0):.0f} MW |
| Demand | {obs.get('demand_mw', 0):.0f} MW |
| Balance | **{obs.get('balance_mw', 0):+.0f} MW** |

### 🔋 Battery
[{soc_bar}] **{battery.get('soc', 0):.0%}** · {battery.get('available_discharge_mw', 0):.0f} MW available

### 🏭 Sources
{chr(10).join(src_lines)}

### 📊 KPIs
| | |
|---|---|
| Uptime | **{kpi.get('uptime_percent', 100):.2f}%** |
| Renewable | {kpi.get('renewable_percent', 0):.1f}% |
| Blackout intervals | {kpi.get('blackout_intervals', 0)} |
| Avg freq dev | {kpi.get('avg_frequency_dev_hz', 0):.4f} Hz |

### 🏠 Human Impact
| | |
|---|---|
| Homes at risk | **{impact.get('homes_at_risk', 0):,}** |
| Hospitals backup | {impact.get('hospitals_on_backup', 0)} |
| CO2 saved | {impact.get('co2_saved_kg', 0):,.0f} kg |
| Blackout minutes | {impact.get('blackout_minutes', 0):.0f} |

---
{dis_line}
"""


def build_obs_json(s: dict) -> str:
    if not s["obs"]:
        return "{}"
    obs = s["obs"]
    last_reward = s["history"][-1]["reward"] if s["history"] else 0
    last_breakdown = s["history"][-1]["breakdown"] if s["history"] else {}
    return fmt({
        "done": obs.get("done", False),
        "interval": obs.get("interval", 0),
        "hour": obs.get("hour", 0),
        "grid_status": obs.get("grid_status", "normal"),
        "frequency_hz": obs.get("frequency_hz", 50.0),
        "balance_mw": obs.get("balance_mw", 0),
        "reward": last_reward,
        "reward_breakdown": last_breakdown,
        "situation_summary": obs.get("situation_summary", ""),
        "kpis": obs.get("kpis", {}),
        "human_impact": obs.get("human_impact", {}),
        "active_disruptions": obs.get("active_disruptions", []),
        "battery": obs.get("battery", {}),
        "forecast": obs.get("forecast", {}),
        "recent_events": obs.get("recent_events", []),
        "financials": obs.get("financials", {}),
    })


def build_history_md(s: dict) -> str:
    if not s["history"]:
        return "*No actions taken yet.*"

    lines = []
    for h in reversed(s["history"][-12:]):
        bd = h.get("breakdown", {})
        lines.append(
            f"**{h['timestamp']} (Step {h['step']})**\n\n"
            f"```json\n{h['action_json']}\n```\n\n"
            f"**Reward:** `{h['reward']:+.4f}`  "
            f"bal `{bd.get('balance_signal', 0):+.3f}` "
            f"| freq `{bd.get('frequency_penalty', 0):+.3f}` "
            f"| renew `{bd.get('renewable_bonus', 0):+.3f}` "
            f"| shed `{bd.get('shed_penalty', 0):+.3f}` "
            f"| blackout `{bd.get('blackout_penalty', 0):+.3f}`\n\n"
            f"*Uptime: {h['uptime']:.1f}% | Freq: {h['freq']:.3f} Hz*"
        )
    return "\n\n---\n\n".join(lines)


def do_reset(task_id, seed, session):
    session["env"] = EnergyGridEnv()
    session["task_id"] = task_id
    session["obs"] = session["env"].reset(task_id=task_id, seed=int(seed))
    session["history"] = []
    session["step_count"] = 0
    session["episode_id"] = str(uuid.uuid4())[:8].upper()
    session["running"] = True
    return (
        build_obs_json(session),
        build_state_md(session),
        build_history_md(session),
        f"✅ Reset · Task: **{task_id}** · Seed: {seed} · Episode: `{session['episode_id']}`",
        session,
    )


def do_get_state(session):
    if not session["running"]:
        return (
            "{}",
            build_state_md(session),
            build_history_md(session),
            "⚠️ No active episode.",
            session,
        )
    return (
        fmt(session["env"].state()),
        build_state_md(session),
        build_history_md(session),
        "📋 Full state loaded.",
        session,
    )


def do_step(action_type, session):
    if not session["running"]:
        return (
            build_obs_json(session),
            build_state_md(session),
            build_history_md(session),
            "⚠️ Reset first.",
            session,
        )

    if session["obs"].get("done"):
        g = grade(session["task_id"], session["env"].state())
        return (
            build_obs_json(session),
            build_state_md(session),
            build_history_md(session),
            f"🏁 Done! Score: **{g['final_score']:.4f}** · {'✅ PASSED' if g['passed'] else '❌ FAILED'}",
            session,
        )

    action = {"action_type": action_type, "params": {}}
    try:
        res = session["env"].step(action)
    except RuntimeError as e:
        return (
            build_obs_json(session),
            build_state_md(session),
            build_history_md(session),
            f"❌ {e}",
            session,
        )

    session["obs"] = res["observation"]
    session["step_count"] += 1
    kpis = res["observation"].get("kpis", {})
    session["history"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "step": session["step_count"],
        "action_json": fmt(action),
        "reward": res["reward"],
        "breakdown": res.get("reward_breakdown", {}),
        "uptime": kpis.get("uptime_percent", 100),
        "freq": res["observation"].get("frequency_hz", 50),
    })

    events = res["observation"].get("recent_events", [])
    ev_str = " · ".join(events[:2]) if events else "—"
    status = (
        f"Step {session['step_count']} · Reward: **{res['reward']:+.4f}** · "
        f"Uptime: {kpis.get('uptime_percent', 100):.1f}% · "
        f"Freq: {res['observation'].get('frequency_hz', 50):.3f} Hz · {ev_str}"
    )
    if res.get("done"):
        status += " 🏁 **Episode finished!**"

    return (
        build_obs_json(session),
        build_state_md(session),
        build_history_md(session),
        status,
        session,
    )


def do_run_agent(agent_name, task_id, seed, session):
    from scripts.baseline_inference import HeuristicAgent, ReactiveAgent, RandomAgent

    agents = {
        "HeuristicAgent": HeuristicAgent(),
        "ReactiveAgent": ReactiveAgent(),
        "RandomAgent": RandomAgent(seed=int(seed)),
    }
    agent = agents.get(agent_name, HeuristicAgent())

    session["env"] = EnergyGridEnv()
    session["task_id"] = task_id
    obs = session["env"].reset(task_id=task_id, seed=int(seed))
    session["obs"] = obs
    session["history"] = []
    session["step_count"] = 0
    session["episode_id"] = str(uuid.uuid4())[:8].upper()
    session["running"] = True

    while True:
        action = agent.act(obs)
        res = session["env"].step(action)
        obs = res["observation"]
        session["obs"] = obs
        session["step_count"] += 1
        kpis = obs.get("kpis", {})
        session["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "step": session["step_count"],
            "action_json": json.dumps(action),
            "reward": res["reward"],
            "breakdown": res.get("reward_breakdown", {}),
            "uptime": kpis.get("uptime_percent", 100),
            "freq": obs.get("frequency_hz", 50),
        })
        if res["done"]:
            break

    g = grade(task_id, session["env"].state())
    st = session["env"].state()
    rwv = st["real_world_validation"]["agent_vs_real_world"]
    status = (
        f"🤖 **{agent_name}** · Task:`{task_id}` · Seed:{seed}\n\n"
        f"**Score:{g['final_score']:.4f}** · {'✅ PASSED' if g['passed'] else '❌ FAILED'}\n\n"
        f"Uptime:{st['kpis']['uptime_percent']:.1f}% · "
        f"Renewable:{st['kpis']['renewable_percent']:.1f}% · "
        f"Blackout mins:{st['human_impact']['blackout_minutes']:.0f}\n\n"
        f"**vs Real World:** {rwv.get('uptime_vs_texas_2021', '')}"
    )
    return (
        build_obs_json(session),
        build_state_md(session),
        build_history_md(session),
        status,
        session,
    )


_VALID_ACTIONS = {
    "wait", "increase_gas_small", "increase_gas_medium", "increase_gas_large",
    "decrease_gas_small", "decrease_gas_medium", "activate_peaker",
    "deactivate_peaker", "charge_battery", "discharge_battery_half",
    "discharge_battery_full", "buy_from_neighbor", "sell_to_neighbor",
    "shed_load_small", "shed_load_large",
}


_LLM_SYSTEM_PROMPT = textwrap.dedent("""
You are an AI electricity grid operator. Your ONLY output must be a single JSON object.
Do NOT write any explanation, reasoning, or text — just the JSON.

=== DECISION RULES (apply IN ORDER, stop at first match) ===

RULE 1 — BLACKOUT / CRITICAL (freq < 49.5 Hz OR balance < -300 MW):
  -> {"action_type": "activate_peaker", "params": {}}        if peaker offline
  -> {"action_type": "discharge_battery_full", "params": {}} if battery SoC > 20%
  -> {"action_type": "buy_from_neighbor", "params": {}}      otherwise

RULE 2 — WARNING (freq 49.5-49.8 Hz OR balance -300 to -100 MW):
  -> {"action_type": "increase_gas_large", "params": {}}     if balance < -200 MW
  -> {"action_type": "increase_gas_medium", "params": {}}    if balance < -100 MW
  -> {"action_type": "discharge_battery_half", "params": {}} if battery SoC > 30%

RULE 3 — DISRUPTION ACTIVE (any disruption in list):
  -> {"action_type": "increase_gas_large", "params": {}}     if balance < -100 MW
  -> {"action_type": "increase_gas_medium", "params": {}}    if balance < 0 MW
  -> {"action_type": "activate_peaker", "params": {}}        if balance < -200 MW and peaker offline

RULE 4 — OVERSUPPLY (balance > +200 MW):
  -> {"action_type": "charge_battery", "params": {}}         if battery SoC < 80%
  -> {"action_type": "sell_to_neighbor", "params": {}}       if battery SoC >= 80%
  -> {"action_type": "decrease_gas_medium", "params": {}}    otherwise

RULE 5 — NORMAL, time-of-day management:
  Morning 6-9h (demand rising):   {"action_type": "increase_gas_small", "params": {}}
  Midday 10-14h (solar peak):     {"action_type": "charge_battery", "params": {}}      if SoC < 90%
  Midday 10-14h (solar peak):     {"action_type": "decrease_gas_small", "params": {}}  if SoC >= 90%
  Evening 17-21h (solar falling): {"action_type": "discharge_battery_half", "params": {}} if SoC > 20%
  Evening 17-21h (solar falling): {"action_type": "increase_gas_medium", "params": {}} if SoC <= 20%
  Night / stable:                 {"action_type": "wait", "params": {}}

=== OUTPUT FORMAT ===
Respond with EXACTLY this structure and nothing else:
{"action_type": "<action>", "params": {}}
""").strip()


def _llm_build_prompt(step, obs, history, last_reward, last_breakdown):
    battery = obs.get("battery", {})
    dis = obs.get("active_disruptions", [])
    kpis = obs.get("kpis", {})
    sources = obs.get("sources", {})
    forecast = obs.get("forecast", {})

    status = obs.get("grid_status", "normal").upper()
    freq = obs.get("frequency_hz", 50.0)
    balance = obs.get("balance_mw", 0.0)
    hour = obs.get("hour", 0.0)
    soc = battery.get("soc", 0.0)
    batt_mw = battery.get("available_discharge_mw", 0)
    peaker = sources.get("peaker", {})
    peaker_online = peaker.get("is_online", False)

    if status in ("BLACKOUT", "CRITICAL") or freq < 49.5:
        urgency = "!!! EMERGENCY — GRID FAILING — ACT NOW !!!"
    elif status == "WARNING" or freq < 49.8:
        urgency = "!! WARNING — grid unstable, act this step !!"
    elif dis:
        urgency = "! DISRUPTION ACTIVE — do not wait !"
    else:
        urgency = "Grid stable."

    dis_txt = ", ".join(
        f"{d['disruption_type']}(sev={d['severity']:.0%},ends={d.get('end_interval', '?')})"
        for d in dis
    ) or "none"

    src_txt = "  " + " | ".join(
        f"{sid}:{s.get('current_mw', 0):.0f}/{s.get('capacity_mw', 0):.0f}MW"
        + ("(OFF)" if not s.get("is_online", True) else "")
        for sid, s in sources.items()
    )

    last_reward_txt = (
        f"{last_reward:+.4f} (bal={last_breakdown.get('balance_signal', 0):+.3f} "
        f"freq={last_breakdown.get('frequency_penalty', 0):+.3f} "
        f"blackout={last_breakdown.get('blackout_penalty', 0):+.3f})"
        if last_reward is not None and last_breakdown else "n/a (step 1)"
    )

    return f"""=== GRID STATE (Step {step}/96, Time {hour:.2f}h) ===
{urgency}

STATUS  : {status}
FREQ    : {freq:.3f} Hz  [NORMAL=49.8-50.2 | WARNING=49.5-49.8 | CRITICAL<49.5 | BLACKOUT<49.0]
BALANCE : {balance:+.1f} MW  (supply={obs.get("supply_mw", 0):.0f} MW, demand={obs.get("demand_mw", 0):.0f} MW)
BATTERY : SoC={soc:.0%}  discharge_available={batt_mw:.0f} MW
PEAKER  : {"ONLINE" if peaker_online else "OFFLINE (can activate for +200 MW instantly)"}
DISRUPTIONS: {dis_txt}

SOURCES:
{src_txt}

FORECAST (next interval):
  demand={forecast.get("next_interval_demand_mw", 0):.0f} MW  solar={forecast.get("next_interval_solar_mw", 0):.0f} MW
  upcoming={forecast.get("upcoming_disruptions", [])}

KPIs so far:
  uptime={kpis.get("uptime_percent", 100):.1f}%  blackout_intervals={kpis.get("blackout_intervals", 0)}  renewable={kpis.get("renewable_percent", 0):.1f}%

LAST REWARD: {last_reward_txt}
LAST 3 ACTIONS: {"; ".join(history[-3:]) if history else "none"}

=== YOUR TASK ===
Apply the DECISION RULES from the system prompt IN ORDER and output the matching JSON action.
Freq={freq:.3f}Hz  Balance={balance:+.1f}MW  Status={status}  SoC={soc:.0%}  Disruptions={"YES" if dis else "NO"}
Output ONLY the JSON:"""


def _llm_parse_action(text):
    start = text.find("{")
    print(f"Parsing LLM response for action: {text}")
    if start == -1:
        return {"action_type": "wait", "params": {}}

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    action = json.loads(text[start:i + 1])
                    at = action.get("action_type", "")
                    if at in _VALID_ACTIONS:
                        return {"action_type": at, "params": action.get("params", {})}
                except json.JSONDecodeError:
                    pass
                break
    return {"action_type": "wait", "params": {}}


def extract_real_url(url_or_label: str) -> str:
    if not url_or_label:
        return ""
    match = re.search(r"\((https?://[^)]+)\)", url_or_label)
    if match:
        return match.group(1)
    return url_or_label.strip()


def do_run_llm_agent(api_base_url, api_key, model_name, task_id, seed, session):
    if not api_key.strip():
        return (
            build_obs_json(session),
            build_state_md(session),
            build_history_md(session),
            "Error: API key is required.",
            session,
        )

    real_api_base_url = extract_real_url(api_base_url)
    client = OpenAI(base_url=real_api_base_url.strip(), api_key=api_key.strip())

    session["env"] = EnergyGridEnv()
    session["task_id"] = task_id
    obs = session["env"].reset(task_id=task_id, seed=int(seed))
    session["obs"] = obs
    session["history"] = []
    session["step_count"] = 0
    session["episode_id"] = str(uuid.uuid4())[:8].upper()
    session["running"] = True

    llm_history = []
    last_reward = None
    last_breakdown = None
    total_reward = 0.0

    for step in range(1, 21):
        prompt = _llm_build_prompt(step, obs, llm_history, last_reward, last_breakdown)
        try:
            completion = client.chat.completions.create(
                model=model_name.strip(),
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.05,
                max_tokens=200,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"LLM error at step {step}: {exc}")
            response_text = ""
            llm_history.append(f"Step {step}: LLM error ({exc}), used wait")

        print(f"LLM response (Step {step}): {response_text}")
        action = _llm_parse_action(response_text)

        try:
            res = session["env"].step(action)
        except RuntimeError:
            break

        obs = res["observation"]
        last_reward = res["reward"]
        last_breakdown = res.get("reward_breakdown", {})
        total_reward += last_reward

        session["obs"] = obs
        session["step_count"] += 1
        kpis = obs.get("kpis", {})
        session["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "step": session["step_count"],
            "action_json": json.dumps(action),
            "reward": last_reward,
            "breakdown": last_breakdown or {},
            "uptime": kpis.get("uptime_percent", 100),
            "freq": obs.get("frequency_hz", 50),
        })

        llm_history.append(
            f"Step {step} action={action['action_type']} "
            f"reward={last_reward:+.4f} status={obs.get('grid_status')} "
            f"freq={obs.get('frequency_hz', 50):.2f}Hz"
        )

        if res.get("done"):
            break

    g = grade(task_id, session["env"].state())
    st = session["env"].state()
    rwv = st["real_world_validation"]["agent_vs_real_world"]
    status = (
        f"LLM Agent ({model_name}) | Task:{task_id} | Seed:{seed}\n\n"
        f"Score: {g['final_score']:.4f} | {'PASSED' if g['passed'] else 'FAILED'} (threshold 0.60)\n\n"
        f"Total reward:{total_reward:.4f} | "
        f"Uptime:{st['kpis']['uptime_percent']:.1f}% | "
        f"Renewable:{st['kpis']['renewable_percent']:.1f}% | "
        f"Blackout mins:{st['human_impact']['blackout_minutes']:.0f}\n\n"
        + "\n\n".join(f"- {v}" for v in rwv.values())
    )

    return (
        build_obs_json(session),
        build_state_md(session),
        build_history_md(session),
        status,
        session,
    )


API_CONFIGS = {
    "Hugging Face": {
        "url": "https://router.huggingface.co/v1",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct",
            "openai/gpt-oss-20b",
        ],
    },
    "Groq": {
        "url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ],
    },
}


def update_provider(provider):
    cfg = API_CONFIGS[provider]
    url_label = f"{provider} - [{cfg['url']}]({cfg['url']})"
    return (
        gr.update(value=url_label),
        gr.update(choices=cfg["models"], value=cfg["models"][0]),
    )


def build_demo():
    with gr.Blocks(title="Energy Grid OpenEnv") as demo:
        session = gr.State(new_session())

        gr.HTML("""
        <div style="background:#0f172a;padding:16px 24px;border-radius:10px;
                    margin-bottom:14px;border:1px solid #1e3a5f;display:flex;
                    align-items:center;gap:14px;">
          <span style="font-size:2rem;">⚡</span>
          <div>
            <div style="color:#f1f5f9;font-size:1.2rem;font-weight:800;
                        font-family:'JetBrains Mono',monospace;">
              energy_grid_balancing_openenv
            </div>
            <div style="color:#475569;font-size:.8rem;font-family:monospace;margin-top:3px;">
              OpenEnv · step() / reset() / state() · 3 tasks · validated vs Texas 2021 & California 2022
            </div>
          </div>
          <div style="margin-left:auto;display:flex;gap:8px;">
            <span style="background:#14532d;color:#86efac;padding:3px 12px;
                         border-radius:20px;font-size:.75rem;font-weight:700;">● RUNNING</span>
            <span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;
                         border-radius:20px;font-size:.75rem;">NERC BAL-001</span>
          </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                gr.HTML(
                    '<div style="font-weight:800;font-size:1.05rem;'
                    'letter-spacing:.06em;border-bottom:2px solid #3b82f6;'
                    'padding-bottom:5px;margin-bottom:10px;">🧑 HumanAgent Interface</div>'
                )

                with gr.Accordion("Show Instructions", open=False):
                    gr.Markdown("""
**Tasks:**
- 🟢 `task_easy` — Morning ramp + afternoon wind lull
- 🟡 `task_medium` — California 2022: heatwave + solar cliff
- 🔴 `task_hard` — Texas 2021: gas freeze + wind ice + cascade

**Reward breakdown per step:**
`balance` + `frequency` + `cost` + `renewable` + `shed` + `blackout` = `total`

**Real-world benchmarks:**
- Texas 2021 actual uptime: **67%** (catastrophic)
- California 2022 actual: **91%** (near miss)
- NERC standard target: **99.97%**
                    """)

                gr.Markdown("### `energy_grid_env`")
                task_dd = gr.Dropdown(
                    ["task_easy", "task_medium", "task_hard"],
                    value="task_easy",
                    label="Task ID *",
                    info="Difficulty / real-world scenario",
                )
                seed_num = gr.Number(42, label="Seed", precision=0)

                gr.HTML('<div style="font-weight:700;margin:12px 0 6px;">Take Action</div>')
                action_dd = gr.Dropdown(
                    [
                        "wait",
                        "increase_gas_small", "increase_gas_medium", "increase_gas_large",
                        "decrease_gas_small", "decrease_gas_medium",
                        "activate_peaker", "deactivate_peaker",
                        "charge_battery", "discharge_battery_half", "discharge_battery_full",
                        "buy_from_neighbor", "sell_to_neighbor",
                        "shed_load_small", "shed_load_large",
                    ],
                    value="wait",
                    label="Action Type *",
                    info="The grid action to execute",
                )

                step_btn = gr.Button("⚡  Step", variant="primary", size="lg")
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")
                    state_btn = gr.Button("📋 Get State", variant="secondary")

                gr.HTML('<div style="font-weight:700;margin:12px 0 6px;">🤖 Run Baseline Agent</div>')
                with gr.Row():
                    agent_dd = gr.Dropdown(
                        ["HeuristicAgent", "ReactiveAgent", "RandomAgent"],
                        value="HeuristicAgent",
                        label="Agent",
                        scale=2,
                    )
                    agent_btn = gr.Button("▶ Run Full Episode", scale=1)

                gr.HTML('<div style="font-weight:700;margin:14px 0 6px;border-top:1px solid #334155;padding-top:10px;">🧠 LLM Agent</div>')
                with gr.Accordion("Configure LLM Agent", open=True):
                    llm_api_provider = gr.Dropdown(
                        choices=list(API_CONFIGS.keys()),
                        value="Hugging Face",
                        label="API Provider",
                    )

                    llm_api_url = gr.Dropdown(
                        choices=[
                            f"{name} - [{config['url']}]({config['url']})"
                            for name, config in API_CONFIGS.items()
                        ],
                        value=f"Hugging Face - [{API_CONFIGS['Hugging Face']['url']}]({API_CONFIGS['Hugging Face']['url']})",
                        label="API Base URL",
                        interactive=False,
                    )

                    llm_api_key = gr.Textbox(
                        value="",
                        label="API Key",
                        placeholder="hf_xxxx or sk-xxxx",
                        type="password",
                    )

                    llm_model = gr.Dropdown(
                        choices=API_CONFIGS["Hugging Face"]["models"],
                        value=API_CONFIGS["Hugging Face"]["models"][0],
                        label="Model Name",
                    )

                    llm_task = gr.Dropdown(
                        ["task_easy", "task_medium", "task_hard"],
                        value="task_easy",
                        label="Task",
                    )

                    llm_seed = gr.Number(42, label="Seed", precision=0)
                    llm_btn = gr.Button("▶ Run Full LLM Episode", variant="primary")

                llm_api_provider.change(
                    fn=update_provider,
                    inputs=[llm_api_provider],
                    outputs=[llm_api_url, llm_model],
                )

                status_out = gr.Markdown("*Reset the environment to begin.*")
                gr.HTML('<div style="font-weight:700;margin:12px 0 4px;">Current State</div>')
                state_md_out = gr.Markdown(
                    "**Status:** ⬜ Not started\n\n*Click Reset Environment to begin.*"
                )

            with gr.Column(scale=1, min_width=380):
                gr.HTML(
                    '<div style="font-weight:800;font-size:1.05rem;'
                    'letter-spacing:.06em;border-bottom:2px solid #3b82f6;'
                    'padding-bottom:5px;margin-bottom:10px;">🔭 State Observer</div>'
                )
                gr.HTML(
                    '<div style="font-size:.82rem;color:#64748b;font-weight:600;'
                    'margin-bottom:4px;">CURRENT OBSERVATION</div>'
                )
                obs_json_out = gr.Code("{}", language="json", label="", lines=24)
                gr.HTML(
                    '<div style="font-size:.82rem;color:#64748b;font-weight:600;'
                    'margin:12px 0 4px;">ACTION HISTORY</div>'
                )
                history_md_out = gr.Markdown("*No actions taken yet.*")

        outs = [obs_json_out, state_md_out, history_md_out, status_out, session]

        reset_btn.click(do_reset, [task_dd, seed_num, session], outs)
        state_btn.click(do_get_state, [session], outs)
        step_btn.click(do_step, [action_dd, session], outs)
        agent_btn.click(do_run_agent, [agent_dd, task_dd, seed_num, session], outs)
        llm_btn.click(
            do_run_llm_agent,
            [llm_api_url, llm_api_key, llm_model, llm_task, llm_seed, session],
            outs,
        )

    return demo


def main():
    threading.Thread(target=_start_api, daemon=True).start()
    time.sleep(1.0)
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    )


if __name__ == "__main__":
    main()


# """
# server/app.py — Gradio HumanAgent Interface + State Observer
# Energy Grid Balancing OpenEnv — Gradio 6.x compatible
# """
# from __future__ import annotations
# import sys, json, threading, time, uuid, textwrap, os, re
# from pathlib import Path
# from datetime import datetime
# from typing import Any
# from openai import OpenAI

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import gradio as gr
# import uvicorn
# from server.my_environment_api import app as fastapi_app
# from energy_grid_balancing_environment import EnergyGridEnv
# from graders import grade

# def _start_api():
#     uvicorn.run(fastapi_app, host="0.0.0.0", port=7861, log_level="error")

# def new_session() -> dict:
#     return {"env": EnergyGridEnv(), "obs": {}, "history": [],
#             "step_count": 0, "episode_id": "—",
#             "running": False, "task_id": "task_easy"}

# STATUS_ICONS = {"normal":"🟢","warning":"🟡","critical":"🔴","blackout":"💀"}
# DIS_ICONS    = {"plant_trip":"🏭","wind_drop":"💨","solar_cloud":"☁️",
#                 "demand_spike":"📈","transmission_fault":"⚡","fuel_shortage":"⛽"}

# def fmt(obj: Any) -> str:
#     return json.dumps(obj, indent=2, default=str)

# def build_state_md(s: dict) -> str:
#     if not s["running"]:
#         return "**Status:** ⬜ Not started\n\n*Click Reset Environment to begin.*"
#     obs    = s["obs"]
#     kpi    = obs.get("kpis", {})
#     impact = obs.get("human_impact", {})
#     dis    = obs.get("active_disruptions", [])
#     sources= obs.get("sources", {})
#     battery= obs.get("battery", {})

#     status_icon = STATUS_ICONS.get(obs.get("grid_status","normal"), "🟢")

#     dis_line = "**Active Disruption:** None ✅"
#     if dis:
#         d = dis[0]
#         icon = DIS_ICONS.get(d.get("disruption_type",""), "⚠️")
#         dis_line = (f"**Active Disruption:** {icon} "
#                     f"{d.get('disruption_type','').replace('_',' ').title()} "
#                     f"· severity {d.get('severity',0):.0%} "
#                     f"· ref: {d.get('real_world_ref','')}")

#     src_lines = []
#     for sid, src in sources.items():
#         pct   = src.get("current_mw",0) / max(src.get("capacity_mw",1), 1)
#         bar   = "█"*int(pct*8) + "░"*(8-int(pct*8))
#         renew = "🌿" if src.get("is_renewable") else "🏭"
#         src_lines.append(
#             f"{renew} `{sid}` [{bar}] **{src.get('current_mw',0):.0f}**/"
#             f"{src.get('capacity_mw',0):.0f} MW "
#             f"{'✅' if src.get('is_online',True) else '🔴 OFFLINE'}"
#         )

#     soc_bar = "█"*int(battery.get("soc",0)*10) + "░"*(10-int(battery.get("soc",0)*10))

#     return f"""**Status:** {status_icon} {obs.get('grid_status','?').upper()}
# **Episode:** `{s['episode_id']}`  **Task:** `{s['task_id']}`
# **Interval:** {s['step_count']} / 96  **Time:** {obs.get('hour',0):.2f}h

# ---
# ### ⚡ Grid
# | Metric | Value |
# |---|---|
# | Frequency | **{obs.get('frequency_hz',50):.3f} Hz** (target 50.0) |
# | Supply | {obs.get('supply_mw',0):.0f} MW |
# | Demand | {obs.get('demand_mw',0):.0f} MW |
# | Balance | **{obs.get('balance_mw',0):+.0f} MW** |

# ### 🔋 Battery
# [{soc_bar}] **{battery.get('soc',0):.0%}** · {battery.get('available_discharge_mw',0):.0f} MW available

# ### 🏭 Sources
# {chr(10).join(src_lines)}

# ### 📊 KPIs
# | | |
# |---|---|
# | Uptime | **{kpi.get('uptime_percent',100):.2f}%** |
# | Renewable | {kpi.get('renewable_percent',0):.1f}% |
# | Blackout intervals | {kpi.get('blackout_intervals',0)} |
# | Avg freq dev | {kpi.get('avg_frequency_dev_hz',0):.4f} Hz |

# ### 🏠 Human Impact
# | | |
# |---|---|
# | Homes at risk | **{impact.get('homes_at_risk',0):,}** |
# | Hospitals backup | {impact.get('hospitals_on_backup',0)} |
# | CO2 saved | {impact.get('co2_saved_kg',0):,.0f} kg |
# | Blackout minutes | {impact.get('blackout_minutes',0):.0f} |

# ---
# {dis_line}
# """

# def build_obs_json(s: dict) -> str:
#     if not s["obs"]: return "{}"
#     obs            = s["obs"]
#     last_reward    = s["history"][-1]["reward"]    if s["history"] else 0
#     last_breakdown = s["history"][-1]["breakdown"] if s["history"] else {}
#     return fmt({
#         "done":               obs.get("done", False),
#         "interval":           obs.get("interval", 0),
#         "hour":               obs.get("hour", 0),
#         "grid_status":        obs.get("grid_status", "normal"),
#         "frequency_hz":       obs.get("frequency_hz", 50.0),
#         "balance_mw":         obs.get("balance_mw", 0),
#         "reward":             last_reward,
#         "reward_breakdown":   last_breakdown,
#         "situation_summary":  obs.get("situation_summary", ""),
#         "kpis":               obs.get("kpis", {}),
#         "human_impact":       obs.get("human_impact", {}),
#         "active_disruptions": obs.get("active_disruptions", []),
#         "battery":            obs.get("battery", {}),
#         "forecast":           obs.get("forecast", {}),
#         "recent_events":      obs.get("recent_events", []),
#         "financials":         obs.get("financials", {}),
#     })

# def build_history_md(s: dict) -> str:
#     if not s["history"]: return "*No actions taken yet.*"
#     lines = []
#     for h in reversed(s["history"][-12:]):
#         bd = h.get("breakdown", {})
#         lines.append(
#             f"**{h['timestamp']} (Step {h['step']})**\n\n"
#             f"```json\n{h['action_json']}\n```\n\n"
#             f"**Reward:** `{h['reward']:+.4f}`  "
#             f"bal `{bd.get('balance_signal',0):+.3f}` "
#             f"| freq `{bd.get('frequency_penalty',0):+.3f}` "
#             f"| renew `{bd.get('renewable_bonus',0):+.3f}` "
#             f"| shed `{bd.get('shed_penalty',0):+.3f}` "
#             f"| blackout `{bd.get('blackout_penalty',0):+.3f}`\n\n"
#             f"*Uptime: {h['uptime']:.1f}% | Freq: {h['freq']:.3f} Hz*"
#         )
#     return "\n\n---\n\n".join(lines)

# # ── Handlers ──────────────────────────────────────────────────────────────────
# def do_reset(task_id, seed, session):
#     session["env"]        = EnergyGridEnv()
#     session["task_id"]    = task_id
#     session["obs"]        = session["env"].reset(task_id=task_id, seed=int(seed))
#     session["history"]    = []
#     session["step_count"] = 0
#     session["episode_id"] = str(uuid.uuid4())[:8].upper()
#     session["running"]    = True
#     return (build_obs_json(session), build_state_md(session), build_history_md(session),
#             f"✅ Reset · Task: **{task_id}** · Seed: {seed} · Episode: `{session['episode_id']}`",
#             session)

# def do_get_state(session):
#     if not session["running"]:
#         return ("{}", build_state_md(session), build_history_md(session),
#                 "⚠️ No active episode.", session)
#     return (fmt(session["env"].state()), build_state_md(session),
#             build_history_md(session), "📋 Full state loaded.", session)

# def do_step(action_type, session):
#     if not session["running"]:
#         return (build_obs_json(session), build_state_md(session),
#                 build_history_md(session), "⚠️ Reset first.", session)
#     if session["obs"].get("done"):
#         g = grade(session["task_id"], session["env"].state())
#         return (build_obs_json(session), build_state_md(session),
#                 build_history_md(session),
#                 f"🏁 Done! Score: **{g['final_score']:.4f}** · {'✅ PASSED' if g['passed'] else '❌ FAILED'}",
#                 session)
#     action = {"action_type": action_type, "params": {}}
#     try:
#         res = session["env"].step(action)
#     except RuntimeError as e:
#         return (build_obs_json(session), build_state_md(session),
#                 build_history_md(session), f"❌ {e}", session)
#     session["obs"]         = res["observation"]
#     session["step_count"] += 1
#     kpis = res["observation"].get("kpis", {})
#     session["history"].append({
#         "timestamp":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
#         "step":        session["step_count"],
#         "action_json": fmt(action),
#         "reward":      res["reward"],
#         "breakdown":   res.get("reward_breakdown", {}),
#         "uptime":      kpis.get("uptime_percent", 100),
#         "freq":        res["observation"].get("frequency_hz", 50),
#     })
#     events = res["observation"].get("recent_events", [])
#     ev_str = " · ".join(events[:2]) if events else "—"
#     status = (f"Step {session['step_count']} · Reward: **{res['reward']:+.4f}** · "
#               f"Uptime: {kpis.get('uptime_percent',100):.1f}% · "
#               f"Freq: {res['observation'].get('frequency_hz',50):.3f} Hz · {ev_str}")
#     if res.get("done"): status += " 🏁 **Episode finished!**"
#     return (build_obs_json(session), build_state_md(session),
#             build_history_md(session), status, session)

# def do_run_agent(agent_name, task_id, seed, session):
#     from scripts.baseline_inference import HeuristicAgent, ReactiveAgent, RandomAgent
#     agents = {"HeuristicAgent": HeuristicAgent(),
#               "ReactiveAgent":  ReactiveAgent(),
#               "RandomAgent":    RandomAgent(seed=int(seed))}
#     agent = agents.get(agent_name, HeuristicAgent())
#     session["env"]        = EnergyGridEnv()
#     session["task_id"]    = task_id
#     obs = session["env"].reset(task_id=task_id, seed=int(seed))
#     session["obs"]        = obs
#     session["history"]    = []
#     session["step_count"] = 0
#     session["episode_id"] = str(uuid.uuid4())[:8].upper()
#     session["running"]    = True
#     while True:
#         action = agent.act(obs)
#         res    = session["env"].step(action)
#         obs    = res["observation"]
#         session["obs"]         = obs
#         session["step_count"] += 1
#         kpis = obs.get("kpis", {})
#         session["history"].append({
#             "timestamp":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
#             "step":        session["step_count"],
#             "action_json": json.dumps(action),
#             "reward":      res["reward"],
#             "breakdown":   res.get("reward_breakdown", {}),
#             "uptime":      kpis.get("uptime_percent", 100),
#             "freq":        obs.get("frequency_hz", 50),
#         })
#         if res["done"]: break
#     g  = grade(task_id, session["env"].state())
#     st = session["env"].state()
#     rwv = st["real_world_validation"]["agent_vs_real_world"]
#     status = (
#         f"🤖 **{agent_name}** · Task:`{task_id}` · Seed:{seed}\n\n"
#         f"**Score:{g['final_score']:.4f}** · {'✅ PASSED' if g['passed'] else '❌ FAILED'}\n\n"
#         f"Uptime:{st['kpis']['uptime_percent']:.1f}% · "
#         f"Renewable:{st['kpis']['renewable_percent']:.1f}% · "
#         f"Blackout mins:{st['human_impact']['blackout_minutes']:.0f}\n\n"
#         f"**vs Real World:** {rwv.get('uptime_vs_texas_2021','')}"
#     )
#     return (build_obs_json(session), build_state_md(session),
#             build_history_md(session), status, session)


# # ── LLM Agent ─────────────────────────────────────────────────────────────────

# _VALID_ACTIONS = {
#     "wait", "increase_gas_small", "increase_gas_medium", "increase_gas_large",
#     "decrease_gas_small", "decrease_gas_medium", "activate_peaker",
#     "deactivate_peaker", "charge_battery", "discharge_battery_half",
#     "discharge_battery_full", "buy_from_neighbor", "sell_to_neighbor",
#     "shed_load_small", "shed_load_large",
# }

# _LLM_SYSTEM_PROMPT = textwrap.dedent("""
# You are an AI electricity grid operator. Your ONLY output must be a single JSON object.
# Do NOT write any explanation, reasoning, or text — just the JSON.

# === DECISION RULES (apply IN ORDER, stop at first match) ===

# RULE 1 — BLACKOUT / CRITICAL (freq < 49.5 Hz OR balance < -300 MW):
#   -> {"action_type": "activate_peaker", "params": {}}       if peaker offline
#   -> {"action_type": "discharge_battery_full", "params": {}} if battery SoC > 20%
#   -> {"action_type": "buy_from_neighbor", "params": {}}     otherwise

# RULE 2 — WARNING (freq 49.5-49.8 Hz OR balance -300 to -100 MW):
#   -> {"action_type": "increase_gas_large", "params": {}}    if balance < -200 MW
#   -> {"action_type": "increase_gas_medium", "params": {}}   if balance < -100 MW
#   -> {"action_type": "discharge_battery_half", "params": {}} if battery SoC > 30%

# RULE 3 — DISRUPTION ACTIVE (any disruption in list):
#   -> {"action_type": "increase_gas_large", "params": {}}    if balance < -100 MW
#   -> {"action_type": "increase_gas_medium", "params": {}}   if balance < 0 MW
#   -> {"action_type": "activate_peaker", "params": {}}       if balance < -200 MW and peaker offline

# RULE 4 — OVERSUPPLY (balance > +200 MW):
#   -> {"action_type": "charge_battery", "params": {}}        if battery SoC < 80%
#   -> {"action_type": "sell_to_neighbor", "params": {}}      if battery SoC >= 80%
#   -> {"action_type": "decrease_gas_medium", "params": {}}   otherwise

# RULE 5 — NORMAL, time-of-day management:
#   Morning 6-9h (demand rising):   {"action_type": "increase_gas_small", "params": {}}
#   Midday 10-14h (solar peak):     {"action_type": "charge_battery", "params": {}}     if SoC < 90%
#   Midday 10-14h (solar peak):     {"action_type": "decrease_gas_small", "params": {}} if SoC >= 90%
#   Evening 17-21h (solar falling): {"action_type": "discharge_battery_half", "params": {}} if SoC > 20%
#   Evening 17-21h (solar falling): {"action_type": "increase_gas_medium", "params": {}} if SoC <= 20%
#   Night / stable:                 {"action_type": "wait", "params": {}}

# === OUTPUT FORMAT ===
# Respond with EXACTLY this structure and nothing else:
# {"action_type": "<action>", "params": {}}
# """).strip()


# def _llm_build_prompt(step, obs, history, last_reward, last_breakdown):
#     """Build a tightly structured prompt that maps directly to the decision rules."""
#     battery  = obs.get("battery", {})
#     dis      = obs.get("active_disruptions", [])
#     kpis     = obs.get("kpis", {})
#     sources  = obs.get("sources", {})
#     forecast = obs.get("forecast", {})

#     status   = obs.get("grid_status", "normal").upper()
#     freq     = obs.get("frequency_hz", 50.0)
#     balance  = obs.get("balance_mw", 0.0)
#     hour     = obs.get("hour", 0.0)
#     soc      = battery.get("soc", 0.0)
#     batt_mw  = battery.get("available_discharge_mw", 0)
#     peaker   = sources.get("peaker", {})
#     peaker_online = peaker.get("is_online", False)

#     # Urgency banner
#     if status in ("BLACKOUT", "CRITICAL") or freq < 49.5:
#         urgency = "!!! EMERGENCY — GRID FAILING — ACT NOW !!!"
#     elif status == "WARNING" or freq < 49.8:
#         urgency = "!! WARNING — grid unstable, act this step !!"
#     elif dis:
#         urgency = "! DISRUPTION ACTIVE — do not wait !"
#     else:
#         urgency = "Grid stable."

#     # Disruption summary
#     dis_txt = ", ".join(
#         f"{d['disruption_type']}(sev={d['severity']:.0%},ends={d.get('end_interval','?')})"
#         for d in dis
#     ) or "none"

#     # Source summary (only online + non-zero or important ones)
#     src_txt = "  " + " | ".join(
#         f"{sid}:{s.get('current_mw',0):.0f}/{s.get('capacity_mw',0):.0f}MW"
#         + ("(OFF)" if not s.get("is_online", True) else "")
#         for sid, s in sources.items()
#     )

#     last_reward_txt = (
#         f"{last_reward:+.4f} (bal={last_breakdown.get('balance_signal',0):+.3f} "
#         f"freq={last_breakdown.get('frequency_penalty',0):+.3f} "
#         f"blackout={last_breakdown.get('blackout_penalty',0):+.3f})"
#         if last_reward is not None and last_breakdown else "n/a (step 1)"
#     )

#     return f"""=== GRID STATE (Step {step}/30, Time {hour:.2f}h) ===
# {urgency}

# STATUS  : {status}
# FREQ    : {freq:.3f} Hz  [NORMAL=49.8-50.2 | WARNING=49.5-49.8 | CRITICAL<49.5 | BLACKOUT<49.0]
# BALANCE : {balance:+.1f} MW  (supply={obs.get("supply_mw",0):.0f} MW, demand={obs.get("demand_mw",0):.0f} MW)
# BATTERY : SoC={soc:.0%}  discharge_available={batt_mw:.0f} MW
# PEAKER  : {"ONLINE" if peaker_online else "OFFLINE (can activate for +200 MW instantly)"}
# DISRUPTIONS: {dis_txt}

# SOURCES:
# {src_txt}

# FORECAST (next interval):
#   demand={forecast.get("next_interval_demand_mw",0):.0f} MW  solar={forecast.get("next_interval_solar_mw",0):.0f} MW
#   upcoming={forecast.get("upcoming_disruptions",[])}

# KPIs so far:
#   uptime={kpis.get("uptime_percent",100):.1f}%  blackout_intervals={kpis.get("blackout_intervals",0)}  renewable={kpis.get("renewable_percent",0):.1f}%

# LAST REWARD: {last_reward_txt}
# LAST 3 ACTIONS: {"; ".join(history[-3:]) if history else "none"}

# === YOUR TASK ===
# Apply the DECISION RULES from the system prompt IN ORDER and output the matching JSON action.
# Freq={freq:.3f}Hz  Balance={balance:+.1f}MW  Status={status}  SoC={soc:.0%}  Disruptions={"YES" if dis else "NO"}
# Output ONLY the JSON:"""


# def _llm_parse_action(text):
#     start = text.find("{")
#     print(f"Parsing LLM response for action: {text}")
#     if start == -1:
#         return {"action_type": "wait", "params": {}}
#     depth = 0
#     for i, ch in enumerate(text[start:], start):
#         if ch == "{": depth += 1
#         elif ch == "}":
#             depth -= 1
#             if depth == 0:
#                 try:
#                     action = json.loads(text[start:i+1])
#                     at = action.get("action_type", "")
#                     if at in _VALID_ACTIONS:
#                         return {"action_type": at, "params": action.get("params", {})}
#                 except json.JSONDecodeError:
#                     pass
#                 break
#     return {"action_type": "wait", "params": {}}


# def do_run_llm_agent(api_base_url, api_key, model_name, task_id, seed, session):
#     if not api_key.strip():
#         return (build_obs_json(session), build_state_md(session),
#                 build_history_md(session), "Error: API key is required.", session)
#     client = OpenAI(base_url=api_base_url.strip(), api_key=api_key.strip())
#     session["env"]        = EnergyGridEnv()
#     session["task_id"]    = task_id
#     obs                   = session["env"].reset(task_id=task_id, seed=int(seed))
#     session["obs"]        = obs
#     session["history"]    = []
#     session["step_count"] = 0
#     session["episode_id"] = str(uuid.uuid4())[:8].upper()
#     session["running"]    = True
#     llm_history = []
#     last_reward = last_breakdown = None
#     total_reward = 0.0
#     for step in range(1, 41):
#         prompt = _llm_build_prompt(step, obs, llm_history, last_reward, last_breakdown)
#         try:
#             completion = client.chat.completions.create(
#                 model=model_name.strip(),
#                 messages=[{"role": "system", "content": _LLM_SYSTEM_PROMPT},
#                           {"role": "user",   "content": prompt}],
#                 temperature=0.05, max_tokens=200, stream=False,
#             )
#             response_text = completion.choices[0].message.content or ""
#         except Exception as exc:
#             print(f"LLM error at step {step}: {exc}")
#             response_text = ""
#             llm_history.append(f"Step {step}: LLM error ({exc}), used wait")
#         print(f"LLM response (Step {step}): {response_text}")
#         action = _llm_parse_action(response_text)
#         try:
#             res = session["env"].step(action)
#         except RuntimeError:
#             break
#         obs            = res["observation"]
#         last_reward    = res["reward"]
#         last_breakdown = res.get("reward_breakdown", {})
#         total_reward  += last_reward
#         session["obs"]         = obs
#         session["step_count"] += 1
#         kpis = obs.get("kpis", {})
#         session["history"].append({
#             "timestamp":   datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
#             "step":        session["step_count"],
#             "action_json": json.dumps(action),
#             "reward":      last_reward,
#             "breakdown":   last_breakdown or {},
#             "uptime":      kpis.get("uptime_percent", 100),
#             "freq":        obs.get("frequency_hz", 50),
#         })
#         llm_history.append(
#             f"Step {step} action={action['action_type']} "
#             f"reward={last_reward:+.4f} status={obs.get('grid_status')} "
#             f"freq={obs.get('frequency_hz',50):.2f}Hz"
#         )
#         if res.get("done"):
#             break
#     g   = grade(task_id, session["env"].state())
#     st  = session["env"].state()
#     rwv = st["real_world_validation"]["agent_vs_real_world"]
#     status = (
#         f"LLM Agent ({model_name}) | Task:{task_id} | Seed:{seed}\n\n"
#         f"Score: {g['final_score']:.4f} | {'PASSED' if g['passed'] else 'FAILED'} (threshold 0.60)\n\n"
#         f"Total reward:{total_reward:.4f} | "
#         f"Uptime:{st['kpis']['uptime_percent']:.1f}% | "
#         f"Renewable:{st['kpis']['renewable_percent']:.1f}% | "
#         f"Blackout mins:{st['human_impact']['blackout_minutes']:.0f}\n\n"
#         + "\n\n".join(f"- {v}" for v in rwv.values())
#     )
#     return (build_obs_json(session), build_state_md(session),
#             build_history_md(session), status, session)

# # ── UI ────────────────────────────────────────────────────────────────────────
# def build_demo():
#     with gr.Blocks(title="Energy Grid OpenEnv") as demo:   # ✅ no theme/css here

#         session = gr.State(new_session())

#         gr.HTML("""
#         <div style="background:#0f172a;padding:16px 24px;border-radius:10px;
#                     margin-bottom:14px;border:1px solid #1e3a5f;display:flex;
#                     align-items:center;gap:14px;">
#           <span style="font-size:2rem;">⚡</span>
#           <div>
#             <div style="color:#f1f5f9;font-size:1.2rem;font-weight:800;
#                         font-family:'JetBrains Mono',monospace;">
#               energy_grid_balancing_openenv
#             </div>
#             <div style="color:#475569;font-size:.8rem;font-family:monospace;margin-top:3px;">
#               OpenEnv · step() / reset() / state() · 3 tasks · validated vs Texas 2021 & California 2022
#             </div>
#           </div>
#           <div style="margin-left:auto;display:flex;gap:8px;">
#             <span style="background:#14532d;color:#86efac;padding:3px 12px;
#                          border-radius:20px;font-size:.75rem;font-weight:700;">● RUNNING</span>
#             <span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;
#                          border-radius:20px;font-size:.75rem;">NERC BAL-001</span>
#           </div>
#         </div>
#         """)

#         with gr.Row(equal_height=False):

#             # LEFT — HumanAgent Interface
#             with gr.Column(scale=1, min_width=340):
#                 gr.HTML('<div style="font-weight:800;font-size:1.05rem;'
#                         'letter-spacing:.06em;border-bottom:2px solid #3b82f6;'
#                         'padding-bottom:5px;margin-bottom:10px;">🧑 HumanAgent Interface</div>')

#                 with gr.Accordion("Show Instructions", open=False):
#                     gr.Markdown("""
# **Tasks:**
# - 🟢 `task_easy` — Morning ramp + afternoon wind lull
# - 🟡 `task_medium` — California 2022: heatwave + solar cliff
# - 🔴 `task_hard` — Texas 2021: gas freeze + wind ice + cascade

# **Reward breakdown per step:**
# `balance` + `frequency` + `cost` + `renewable` + `shed` + `blackout` = `total`

# **Real-world benchmarks:**
# - Texas 2021 actual uptime: **67%** (catastrophic)
# - California 2022 actual: **91%** (near miss)
# - NERC standard target: **99.97%**
#                     """)

#                 gr.Markdown("### `energy_grid_env`")
#                 task_dd  = gr.Dropdown(
#                     ["task_easy", "task_medium", "task_hard"],
#                     value="task_easy", label="Task ID *",
#                     info="Difficulty / real-world scenario")
#                 seed_num = gr.Number(42, label="Seed", precision=0)

#                 gr.HTML('<div style="font-weight:700;margin:12px 0 6px;">Take Action</div>')
#                 action_dd = gr.Dropdown(
#                     ["wait",
#                      "increase_gas_small", "increase_gas_medium", "increase_gas_large",
#                      "decrease_gas_small", "decrease_gas_medium",
#                      "activate_peaker", "deactivate_peaker",
#                      "charge_battery", "discharge_battery_half", "discharge_battery_full",
#                      "buy_from_neighbor", "sell_to_neighbor",
#                      "shed_load_small", "shed_load_large"],
#                     value="wait", label="Action Type *",
#                     info="The grid action to execute")

#                 step_btn = gr.Button("⚡  Step", variant="primary", size="lg")
#                 with gr.Row():
#                     reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")
#                     state_btn = gr.Button("📋 Get State",          variant="secondary")

#                 gr.HTML('<div style="font-weight:700;margin:12px 0 6px;">🤖 Run Baseline Agent</div>')
#                 with gr.Row():
#                     agent_dd  = gr.Dropdown(
#                         ["HeuristicAgent", "ReactiveAgent", "RandomAgent"],
#                         value="HeuristicAgent", label="Agent", scale=2)
#                     agent_btn = gr.Button("▶ Run Full Episode", scale=1)


#                 # gr.HTML('<div style="font-weight:700;margin:14px 0 6px;border-top:1px solid #334155;padding-top:10px;">🧠 LLM Agent</div>')
#                 # with gr.Accordion("Configure LLM Agent", open=True):
#                 #     llm_api_url = gr.Textbox(
#                 #         value="https://router.huggingface.co/v1",
#                 #         label="API Base URL",
#                 #         info="OpenAI-compatible endpoint")
#                 #     llm_api_key = gr.Textbox(
#                 #         value="", label="API Key",
#                 #         placeholder="hf_xxxx or sk-xxxx",
#                 #         type="password")
#                 #     llm_model = gr.Textbox(
#                 #         value="meta-llama/Llama-3.3-70B-Instruct",
#                 #         label="Model Name",
#                 #         info="Model identifier for the API")
#                 #     llm_task = gr.Dropdown(
#                 #         ["task_easy", "task_medium", "task_hard"],
#                 #         value="task_easy", label="Task")
#                 #     llm_seed = gr.Number(42, label="Seed", precision=0)
#                 #     llm_btn  = gr.Button("▶ Run Full LLM Episode", variant="primary")

#                 # status_out   = gr.Markdown("*Reset the environment to begin.*")
#                 # gr.HTML('<div style="font-weight:700;margin:12px 0 4px;">Current State</div>')
#                 # state_md_out = gr.Markdown(
#                 #     "**Status:** ⬜ Not started\n\n*Click Reset Environment to begin.*")

#                 # API configurations
#                 API_CONFIGS = {
#                     "Hugging Face": {
#                         "url": "https://router.huggingface.co/v1",
#                         "models": ["meta-llama/Llama-3.3-70B-Instruct"]
#                     },
#                     "Groq": {
#                         "url": "https://api.groq.com/openai/v1", 
#                         "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
#                     }
#                 }

#                 gr.HTML('<div style="font-weight:700;margin:14px 0 6px;border-top:1px solid #334155;padding-top:10px;">🧠 LLM Agent</div>')
#                 with gr.Accordion("Configure LLM Agent", open=True):
#                     # API Base URL Dropdown
#                     llm_api_provider = gr.Dropdown(
#                         choices=["Hugging Face", "Groq"],
#                         value="Hugging Face",
#                         label="API Provider"
#                     )
                    
#                     llm_api_url = gr.Dropdown(
#                         choices=[f"{name} - [{config['url']}]({config['url']})" for name, config in API_CONFIGS.items()],
#                         value=f"Hugging Face - [{API_CONFIGS['Hugging Face']['url']}]({API_CONFIGS['Hugging Face']['url']})",
#                         label="API Base URL",
#                         interactive=False
#                     )
                    
#                     llm_api_key = gr.Textbox(
#                         value="", 
#                         label="API Key",
#                         placeholder="hf_xxxx or sk-xxxx",
#                         type="password"
#                     )
                    
#                     llm_model = gr.Dropdown(
#                         choices=[],
#                         value="meta-llama/Llama-3.3-70B-Instruct",
#                         label="Model Name",
#                         interactive=False
#                     )
                    
#                     llm_task = gr.Dropdown(
#                         ["task_easy", "task_medium", "task_hard"],
#                         value="task_easy", 
#                         label="Task"
#                     )
                    
#                     llm_seed = gr.Number(42, label="Seed", precision=0)
#                     llm_btn = gr.Button("▶ Run Full LLM Episode", variant="primary")

#                 # Add JavaScript to sync provider → URL → models
#                 llm_api_provider.change(
#                     fn=lambda provider: [
#                         f"{provider} - [{API_CONFIGS[provider]['url']}]({API_CONFIGS[provider]['url']})",
#                         API_CONFIGS[provider]['models'][0]
#                     ],
#                     inputs=[llm_api_provider],
#                     outputs=[llm_api_url, llm_model]
#                 ).then(
#                     fn=lambda url: next(config['models'][0] for name, config in API_CONFIGS.items() if config['url'] in url),
#                     inputs=[llm_api_url],
#                     outputs=[llm_model]
#                 )

#             # RIGHT — State Observer
#             with gr.Column(scale=1, min_width=380):
#                 gr.HTML('<div style="font-weight:800;font-size:1.05rem;'
#                         'letter-spacing:.06em;border-bottom:2px solid #3b82f6;'
#                         'padding-bottom:5px;margin-bottom:10px;">🔭 State Observer</div>')
#                 gr.HTML('<div style="font-size:.82rem;color:#64748b;font-weight:600;'
#                         'margin-bottom:4px;">CURRENT OBSERVATION</div>')
#                 obs_json_out   = gr.Code("{}", language="json", label="", lines=24)
#                 gr.HTML('<div style="font-size:.82rem;color:#64748b;font-weight:600;'
#                         'margin:12px 0 4px;">ACTION HISTORY</div>')
#                 history_md_out = gr.Markdown("*No actions taken yet.*")

#         outs = [obs_json_out, state_md_out, history_md_out, status_out, session]
#         reset_btn.click(do_reset,    [task_dd, seed_num, session],          outs)
#         state_btn.click(do_get_state,[session],                              outs)
#         step_btn.click( do_step,     [action_dd, session],                  outs)
#         agent_btn.click(do_run_agent,[agent_dd, task_dd, seed_num, session], outs)
#         llm_btn.click(do_run_llm_agent,
#                       [llm_api_url, llm_api_key, llm_model, llm_task, llm_seed, session],
#                       outs)

#     return demo


# def main():
#     threading.Thread(target=_start_api, daemon=True).start()
#     time.sleep(1.0)
#     demo = build_demo()             # ✅ returns single demo object
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         show_error=True,
#         theme=gr.themes.Base(       # ✅ theme in launch() for Gradio 6.x
#             primary_hue="blue",
#             neutral_hue="slate",
#         ),
#     )

# if __name__ == "__main__":
#     main()