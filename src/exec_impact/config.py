"""Typed configuration dataclasses and YAML loader for exec-impact."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "exec-impact"
    random_seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    year: int = 2024
    month: int = 1
    base_url: str = "https://data.binance.vision/data/spot/daily/aggTrades"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    bar_intervals: list[int] = field(default_factory=lambda: [1, 5, 15])


@dataclass(frozen=True)
class ImpactConfig:
    intervals: list[int] = field(default_factory=lambda: [1, 5, 15])
    delta_fixed: float = 0.5
    bootstrap_samples: int = 500
    train_weeks: list[int] = field(default_factory=lambda: [1, 2])
    test_weeks: list[int] = field(default_factory=lambda: [3, 4])
    permanent_lags: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 15])


@dataclass(frozen=True)
class ExecutionConfig:
    horizons_minutes: list[int] = field(default_factory=lambda: [30, 60, 240])
    order_sizes_adv_pct: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.50, 1.00])
    lambdas: list[float] = field(
        default_factory=lambda: [1.0e-8, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]
    )
    slice_interval_minutes: int = 1
    taker_fee_bps: float = 4.5
    volume_profile_lookback_days: int = 7


@dataclass(frozen=True)
class SimulationConfig:
    n_start_samples: int = 500
    min_warmup_days: int = 7
    monte_carlo_paths: int = 1000


@dataclass(frozen=True)
class ResultsConfig:
    dir: str = "results"
    metrics_file: str = "results/metrics.json"
    figures_dir: str = "results/figures"


@dataclass(frozen=True)
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    impact: ImpactConfig = field(default_factory=ImpactConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    results: ResultsConfig = field(default_factory=ResultsConfig)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        """Load configuration from YAML file or return defaults if file not found."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        return cls(
            project=ProjectConfig(**data.get("project", {})),
            data=DataConfig(**data.get("data", {})),
            impact=ImpactConfig(**data.get("impact", {})),
            execution=ExecutionConfig(**data.get("execution", {})),
            simulation=SimulationConfig(**data.get("simulation", {})),
            results=ResultsConfig(**data.get("results", {})),
        )
