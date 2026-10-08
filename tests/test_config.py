"""Tests for configuration loading and defaults."""

from pathlib import Path

from exec_impact.config import Config


def test_default_config() -> None:
    config = Config()
    assert "BTCUSDT" in config.data.symbols
    assert "ETHUSDT" in config.data.symbols
    assert config.project.random_seed == 42
    assert config.impact.delta_fixed == 0.5
    assert len(config.execution.lambdas) >= 6


def test_load_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
project:
  name: "custom-test"
  random_seed: 123
data:
  symbols: ["BTCUSDT"]
  year: 2024
  month: 1
"""
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text(yaml_content)

    config = Config.load(yaml_file)
    assert config.project.name == "custom-test"
    assert config.project.random_seed == 123
    assert config.data.symbols == ["BTCUSDT"]
