"""
server/my_environment_api.py — FastAPI REST API for Energy Grid OpenEnv
Endpoints: /reset /step /state /grade /tasks /health /docs
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yaml

from energy_grid_balancing_environment import EnergyGridEnv
from graders import grade as run_grade

app = FastAPI(
    title="Energy Grid Balancing — OpenEnv API",
    description="Real-world energy grid simulation validated against NERC, EIA, and historical events.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_env = EnergyGridEnv()


class ResetReq(BaseModel):
    task_id: str = "task_easy"
    seed:    int = 42

class StepReq(BaseModel):
    action_type: str
    params:      dict[str, Any] = {}

class GradeReq(BaseModel):
    task_id: str


@app.get("/health")
def health():
    return {"status": "ok", "env": "energy_grid_balancing_v1"}

@app.post("/reset")
def reset(req: ResetReq):
    return JSONResponse(_env.reset(task_id=req.task_id, seed=req.seed))

@app.post("/step")
def step(req: StepReq):
    try:
        return JSONResponse(_env.step({
            "action_type": req.action_type, "params": req.params}))
    except RuntimeError as e:
        raise HTTPException(400, str(e))

@app.get("/state")
def state():
    return JSONResponse(_env.state())

@app.post("/grade")
def grade(req: GradeReq):
    s = _env.state()
    if not s["done"]:
        raise HTTPException(400, "Episode not finished yet.")
    return JSONResponse(run_grade(req.task_id, s))

@app.get("/tasks")
def tasks():
    return JSONResponse({"tasks": [
        {
            "task_id":     "task_easy",
            "difficulty":  "easy",
            "description": "Morning ramp + afternoon wind lull. Typical grid day.",
            "real_world":  "Standard grid operations",
            "pass_threshold": 0.60,
        },
        {
            "task_id":     "task_medium",
            "difficulty":  "medium",
            "description": "Heatwave demand spike + solar cliff at 6pm.",
            "real_world":  "California September 2022",
            "pass_threshold": 0.60,
        },
        {
            "task_id":     "task_hard",
            "difficulty":  "hard",
            "description": "Polar vortex + gas freeze + wind ice + coal shortage cascade.",
            "real_world":  "Texas February 2021 — 4.5M homes, 246 deaths",
            "pass_threshold": 0.60,
        },
    ]})

@app.get("/spec")
def spec():
    p = Path(__file__).resolve().parent.parent / "openenv.yaml"
    if not p.exists():
        raise HTTPException(404, "openenv.yaml not found")
    return JSONResponse(yaml.safe_load(p.read_text()))

@app.get("/real_world_benchmarks")
def benchmarks():
    return JSONResponse({
        "texas_2021":         {"uptime_percent": 67.0,  "deaths": 246, "homes_affected": 4_500_000},
        "california_2022":    {"uptime_percent": 91.0,  "cost_billion": 1.4},
        "nerc_standard":      {"uptime_percent": 99.97, "standard": "NERC BAL-001-2"},
        "eu_renewable_target":{"renewable_percent": 42.5, "year": 2025},
    })
