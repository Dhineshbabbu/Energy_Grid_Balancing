---
title: Energy Grid Balancing OpenEnv
emoji: ⚡
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Energy grid AI benchmark vs Texas 2021 & CA 2022
---

# ⚡ Energy Grid Balancing — OpenEnv

AI agent benchmark simulating 24 hours of electricity grid operations under
real-world disruptions. Validated against NERC standards and historical crises.

## 🌍 Real-World Validation

| Benchmark | Real Outcome | Your Agent |
|---|---|---|
| Texas Feb 2021 | 67% uptime, 246 deaths | Must beat 67% |
| California Sep 2022 | 91% uptime, $1.4B cost | Must beat 91% |
| NERC Standard | 99.97% uptime | Target |

## 🎮 Tasks

| Task | Scenario | Real-World Reference |
|---|---|---|
| `task_easy` | Morning ramp + wind lull | Standard grid day |
| `task_medium` | Heatwave + solar cliff | California Sep 2022 |
| `task_hard` | Gas freeze + wind ice + cascade | Texas Feb 2021 |

## ⚡ Actions (15 total)

`wait` · `increase_gas_small` · `increase_gas_medium` · `increase_gas_large`  
`decrease_gas_small` · `decrease_gas_medium` · `activate_peaker` · `deactivate_peaker`  
`charge_battery` · `discharge_battery_half` · `discharge_battery_full`  
`buy_from_neighbor` · `sell_to_neighbor` · `shed_load_small` · `shed_load_large`

## 📊 Reward Breakdown

`balance_signal` + `frequency_penalty` + `cost_efficiency` + `renewable_bonus` + `shed_penalty` + `blackout_penalty` = `total`

## 🌿 Human Impact Metrics (per step)

- Homes at risk
- Hospitals on backup
- CO2 saved vs coal baseline
- Blackout minutes accumulated

## 🔌 API

| Endpoint | Description |
|---|---|
| `POST /reset` | Start episode |
| `POST /step` | Take action and return reward breakdown |
| `GET /state` | Full state and real-world validation |
| `POST /grade` | Score from 0.0 to 1.0 |
| `GET /real_world_benchmarks` | Texas, California, and NERC data |
| `GET /docs` | Swagger UI |

## 🤖 LLM API Configuration

Set provider and model values through environment variables instead of hardcoding secrets in source files. The current code already reads API base URL, model name, and token values from environment variables in the CLI flow, which is the right pattern to follow. [file:1]

### Hugging Face

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-20b"
export HF_TOKEN="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"
```

### Groq

```bash
export GROQ_API_BASE_URL="https://api.groq.com/openai/v1"
export GROQ_API_KEY="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"
```
