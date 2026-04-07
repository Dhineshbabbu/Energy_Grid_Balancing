"""
models.py — Typed data models for Energy Grid Balancing OpenEnv
Real-world validated against NERC BAL-001, IEEE 1547, FERC Order 888
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    SOLAR    = "solar"
    WIND     = "wind"
    GAS      = "gas"
    COAL     = "coal"
    NUCLEAR  = "nuclear"
    HYDRO    = "hydro"
    BATTERY  = "battery"
    IMPORT   = "import"

class DisruptionType(str, Enum):
    NONE             = "none"
    PLANT_TRIP       = "plant_trip"        # unplanned generator failure
    WIND_DROP        = "wind_drop"         # sudden wind speed drop
    SOLAR_CLOUD      = "solar_cloud"       # cloud cover / storm
    DEMAND_SPIKE     = "demand_spike"      # heatwave / cold snap
    TRANSMISSION_FAULT = "transmission_fault"  # line failure
    FUEL_SHORTAGE    = "fuel_shortage"     # gas supply disruption

class ActionType(str, Enum):
    INCREASE_GAS_SMALL    = "increase_gas_small"     # +50 MW
    INCREASE_GAS_MEDIUM   = "increase_gas_medium"    # +150 MW
    INCREASE_GAS_LARGE    = "increase_gas_large"     # +300 MW
    DECREASE_GAS_SMALL    = "decrease_gas_small"     # -50 MW
    DECREASE_GAS_MEDIUM   = "decrease_gas_medium"    # -150 MW
    ACTIVATE_PEAKER       = "activate_peaker"         # +200 MW fast, expensive
    DEACTIVATE_PEAKER     = "deactivate_peaker"       # -200 MW
    CHARGE_BATTERY        = "charge_battery"          # store energy
    DISCHARGE_BATTERY     = "discharge_battery_half"  # +100 MW from storage
    DISCHARGE_BATTERY_FULL = "discharge_battery_full" # +200 MW from storage
    BUY_FROM_NEIGHBOR     = "buy_from_neighbor"       # +250 MW, expensive
    SELL_TO_NEIGHBOR      = "sell_to_neighbor"        # -150 MW, earn revenue
    SHED_LOAD_SMALL       = "shed_load_small"         # -100 MW, penalty
    SHED_LOAD_LARGE       = "shed_load_large"         # -300 MW, large penalty
    WAIT                  = "wait"                    # do nothing

class GridStatus(str, Enum):
    NORMAL    = "normal"       # freq 49.8–50.2 Hz
    WARNING   = "warning"      # freq 49.5–49.8 or 50.2–50.5 Hz
    CRITICAL  = "critical"     # freq 49.0–49.5 or 50.5–51.0 Hz
    BLACKOUT  = "blackout"     # freq < 49.0 or > 51.0 Hz


# ── Generation Sources ─────────────────────────────────────────────────────────

@dataclass
class GenerationSource:
    source_id:      str
    source_type:    SourceType
    name:           str
    capacity_mw:    float          # maximum output
    current_mw:     float          # current output
    min_mw:         float          # minimum stable output
    ramp_rate_mw_per_interval: float  # how fast can change per 15min
    cost_per_mwh:   float          # $/MWh operating cost
    co2_kg_per_mwh: float          # emissions intensity
    is_online:      bool = True
    is_renewable:   bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        return d


@dataclass
class BatteryStorage:
    capacity_mwh:   float          # total storage capacity
    current_mwh:    float          # current charge level
    max_charge_mw:  float          # max charge rate
    max_discharge_mw: float        # max discharge rate
    efficiency:     float = 0.92   # round-trip efficiency

    @property
    def soc(self) -> float:
        """State of charge 0.0–1.0"""
        return self.current_mwh / self.capacity_mwh

    @property
    def available_discharge_mw(self) -> float:
        return min(self.max_discharge_mw,
                   self.current_mwh * self.efficiency * 4)  # 4 = intervals/hour

    def to_dict(self) -> dict:
        d = asdict(self)
        d["soc"] = round(self.soc, 3)
        d["available_discharge_mw"] = round(self.available_discharge_mw, 1)
        return d


@dataclass
class Disruption:
    disruption_id:    str
    disruption_type:  DisruptionType
    affected_source:  str           # source_id or "demand"
    severity:         float         # 0.0–1.0
    start_interval:   int
    duration_intervals: int
    description:      str
    real_world_ref:   str           # e.g. "Texas Feb 2021"

    @property
    def end_interval(self) -> int:
        return self.start_interval + self.duration_intervals

    def to_dict(self) -> dict:
        d = asdict(self)
        d["disruption_type"] = self.disruption_type.value
        d["end_interval"]    = self.end_interval
        return d


# ── Reward Breakdown ───────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    """
    Named reward components per step.
    Maps to real-world grid reliability standards.
    """
    balance_signal:    float   # supply≈demand → NERC BAL-001
    frequency_penalty: float   # Hz deviation → NERC PRC-024
    cost_efficiency:   float   # generation cost vs baseline
    renewable_bonus:   float   # % renewable in mix
    shed_penalty:      float   # load shedding → IEEE 1547
    blackout_penalty:  float   # full blackout → catastrophic
    total:             float   # clamped [-1.0, +1.0]

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


# ── Action / Observation / StepResult ─────────────────────────────────────────

@dataclass
class Action:
    action_type: ActionType
    params:      dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"action_type": self.action_type.value, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(action_type=ActionType(d["action_type"]),
                   params=d.get("params", {}))


@dataclass
class HumanImpact:
    """
    Real-world translation of grid metrics.
    Validated against EIA and NERC public data.
    """
    homes_at_risk:      int     # households facing potential outage
    hospitals_on_backup: int    # medical facilities stressed
    co2_saved_kg:       float   # vs coal baseline
    cost_per_home_usd:  float   # daily electricity cost per household
    renewable_percent:  float   # % of mix from renewables
    blackout_minutes:   float   # cumulative blackout time this episode

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Observation:
    interval:           int            # 0–95 (96 × 15min = 24h)
    hour:               float          # 0.0–23.75
    frequency_hz:       float          # target 50.0 Hz
    grid_status:        str            # normal/warning/critical/blackout
    supply_mw:          float          # total generation
    demand_mw:          float          # total consumption
    balance_mw:         float          # supply - demand
    sources:            dict[str, dict] # source_id → GenerationSource.to_dict()
    battery:            dict            # BatteryStorage.to_dict()
    active_disruptions: list[dict]
    forecast:           dict            # next-interval predictions
    financials:         dict[str, float]
    kpis:               dict[str, float]
    human_impact:       dict            # HumanImpact.to_dict()
    situation_summary:  str             # plain English for LLM
    recent_events:      list[str]

    def to_dict(self) -> dict:
        return {
            "interval":           self.interval,
            "hour":               self.hour,
            "frequency_hz":       self.frequency_hz,
            "grid_status":        self.grid_status,
            "supply_mw":          self.supply_mw,
            "demand_mw":          self.demand_mw,
            "balance_mw":         self.balance_mw,
            "sources":            self.sources,
            "battery":            self.battery,
            "active_disruptions": self.active_disruptions,
            "forecast":           self.forecast,
            "financials":         self.financials,
            "kpis":               self.kpis,
            "human_impact":       self.human_impact,
            "situation_summary":  self.situation_summary,
            "recent_events":      self.recent_events,
        }


@dataclass
class StepResult:
    observation:      Observation
    reward:           float
    reward_breakdown: RewardBreakdown
    done:             bool
    info:             dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "observation":      self.observation.to_dict(),
            "reward":           round(self.reward, 4),
            "reward_breakdown": self.reward_breakdown.to_dict(),
            "done":             self.done,
            "info":             self.info,
        }


# ── Aliases for OpenEnv client compatibility ───────────────────────────────────
EnergyGridBalancingAction      = Action
EnergyGridBalancingObservation = Observation