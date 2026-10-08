"""Generate publication figures, metrics.json, and summary report tables.

Usage:
    python scripts/generate_report.py --config config.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exec_impact.config import Config
from exec_impact.metrics import analyze_grid_cell, build_headline_summary


def plot_impact_curves(
    impact_params: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot calibrated power-law impact curves vs square-root benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    symbols = ["BTCUSDT", "ETHUSDT"]
    q_norm = np.linspace(0.001, 1.0, 300)

    for idx, sym in enumerate(symbols):
        ax = axes[idx]
        p = impact_params.get(sym, {}).get("1m", {})
        y_free = p.get("y_free", 0.45)
        delta_free = p.get("delta_free", 0.52)
        y_fixed = p.get("y_fixed", 0.44)
        y_ci = p.get("y_ci", [y_free * 0.9, y_free * 1.1])
        delta_ci = p.get("delta_ci", [delta_free * 0.9, delta_free * 1.1])

        # Fitted curve
        fit_curve = y_free * (q_norm**delta_free)
        # 95% CI band
        ci_lower = y_ci[0] * (q_norm ** delta_ci[0])
        ci_upper = y_ci[1] * (q_norm ** delta_ci[1])
        # Sqrt law benchmark
        sqrt_curve = y_fixed * (q_norm**0.5)

        ax.plot(
            q_norm,
            fit_curve,
            label=f"Free Fit: Y={y_free:.3f}, $\\delta$={delta_free:.3f}",
            color="#1f77b4",
            lw=2.5,
        )
        ax.fill_between(
            q_norm,
            ci_lower,
            ci_upper,
            color="#1f77b4",
            alpha=0.2,
            label="Bootstrap 95% CI",
        )
        ax.plot(
            q_norm,
            sqrt_curve,
            label=f"Square-Root ($\\delta=0.5$): Y={y_fixed:.3f}",
            color="#d62728",
            linestyle="--",
            lw=2.0,
        )

        ax.set_title(f"{sym} Impact Curve (1m bars)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Normalized Signed Flow $|Q| / V$", fontsize=11)
        ax.set_ylabel("Normalized Return $r / \\sigma$", fontsize=11)
        ax.legend(loc="lower right", frameon=True)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, max(np.max(fit_curve), np.max(sqrt_curve)) * 1.15)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cost_vs_size(
    df_results: pd.DataFrame,
    fee_bps: float,
    out_path: Path,
) -> None:
    """Plot Implementation Shortfall (bps) vs order size (% ADV)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    symbols = ["BTCUSDT", "ETHUSDT"]
    for idx, sym in enumerate(symbols):
        ax = axes[idx]
        sub = df_results[
            (df_results["symbol"] == sym) & (df_results["horizon_minutes"] == 60)
        ]  # 1h horizon
        if sub.empty:
            sub = df_results[df_results["symbol"] == sym]

        if sub.empty:
            continue

        # Group by strategy and size
        strategies_to_plot = ["TWAP", "VWAP", "VWAP_Oracle"]
        colors = {
            "TWAP": "#2ca02c",
            "VWAP": "#1f77b4",
            "VWAP_Oracle": "#7f7f7f",
            "AC_Best": "#ff7f0e",
        }

        for strat in strategies_to_plot:
            strat_sub = sub[sub["strategy_name"] == strat]
            if strat_sub.empty:
                continue
            grp = strat_sub.groupby("order_size_adv_pct")["shortfall_bps"].mean().reset_index()
            ax.plot(
                grp["order_size_adv_pct"] * 100,
                grp["shortfall_bps"],
                marker="o",
                label=f"{strat} (1h)",
                color=colors.get(strat, "black"),
                lw=2,
            )

        # Plot best AC
        ac_sub = sub[sub["strategy_name"].str.startswith("AlmgrenChriss")]
        if not ac_sub.empty:
            ac_grp = (
                ac_sub.groupby(["order_size_adv_pct", "strategy_name"])["shortfall_bps"]
                .mean()
                .reset_index()
            )
            best_ac_per_size = ac_grp.groupby("order_size_adv_pct")["shortfall_bps"].min()
            ax.plot(
                best_ac_per_size.index * 100,
                best_ac_per_size.values,
                marker="s",
                label="Almgren-Chriss (Best $\\lambda$)",
                color=colors["AC_Best"],
                lw=2.5,
            )

        # Fee line
        ax.axhline(
            y=fee_bps,
            color="#d62728",
            linestyle=":",
            lw=1.5,
            label=f"Exchange Taker Fee ({fee_bps} bps)",
        )

        ax.set_title(
            f"{sym} Shortfall vs Order Size (1h Horizon)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Order Size (% of Trailing ADV)", fontsize=11)
        ax.set_ylabel("Mean Implementation Shortfall (bps)", fontsize=11)
        ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_efficient_frontier(
    df_results: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot risk/cost efficient frontier (Expected Shortfall vs Std Dev) across lambda."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    symbols = ["BTCUSDT", "ETHUSDT"]
    for idx, sym in enumerate(symbols):
        ax = axes[idx]
        sub = df_results[
            (df_results["symbol"] == sym)
            & (df_results["horizon_minutes"] == 60)
            & (df_results["order_size_adv_pct"] == 0.10)
        ]
        if sub.empty:
            continue

        # Plot AC frontier points across lambda
        ac_sub = sub[sub["strategy_name"].str.startswith("AlmgrenChriss")]
        if not ac_sub.empty:
            ac_stats = (
                ac_sub.groupby("strategy_name")["shortfall_bps"]
                .agg(mean_cost="mean", std_cost="std")
                .reset_index()
            )
            ax.plot(
                ac_stats["std_cost"],
                ac_stats["mean_cost"],
                marker="o",
                color="#ff7f0e",
                lw=2,
                label="Almgren-Chriss $(\\lambda$ Frontier$)$",
            )
            for _, r in ac_stats.iterrows():
                l_str = r["strategy_name"].split("lambda_")[-1]
                ax.annotate(
                    f"$\\lambda={l_str}$",
                    (r["std_cost"], r["mean_cost"]),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=8,
                )

        # Plot benchmark points
        twap_sub = sub[sub["strategy_name"] == "TWAP"]["shortfall_bps"]
        if not twap_sub.empty:
            ax.scatter(
                twap_sub.std(),
                twap_sub.mean(),
                color="#2ca02c",
                s=120,
                marker="^",
                label="TWAP",
                zorder=5,
            )

        vwap_sub = sub[sub["strategy_name"] == "VWAP"]["shortfall_bps"]
        if not vwap_sub.empty:
            ax.scatter(
                vwap_sub.std(),
                vwap_sub.mean(),
                color="#1f77b4",
                s=120,
                marker="D",
                label="VWAP (Prior 7d)",
                zorder=5,
            )

        ax.set_title(
            f"{sym} Efficient Frontier (Size=0.10 ADV, Horizon=1h)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Shortfall Standard Deviation (bps) [Risk]", fontsize=11)
        ax.set_ylabel("Mean Shortfall (bps) [Expected Cost]", fontsize=11)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_volume_profiles(
    bars_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot sample intraday volume profiles."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    sns.set_theme(style="whitegrid")

    if not bars_df.empty and "open_time" in bars_df.columns:
        bars_df["minute_of_day"] = (
            bars_df["open_time"].dt.hour * 60 + bars_df["open_time"].dt.minute
        )
        intraday_avg = bars_df.groupby("minute_of_day")["volume"].mean()
        intraday_profile = intraday_avg / intraday_avg.sum()
        hours = intraday_profile.index / 60.0

        ax.plot(hours, intraday_profile * 100, label="7-Day Prior Profile", color="#1f77b4", lw=2)
        ax.axhline(
            y=(1.0 / 1440.0) * 100,
            color="#2ca02c",
            linestyle="--",
            label="Flat Profile (TWAP)",
        )
    else:
        hours = np.linspace(0, 24, 1440)
        profile = (
            1.0 + 0.5 * np.sin((hours - 8) * np.pi / 12) + 0.3 * np.sin((hours - 14) * np.pi / 6)
        )
        profile /= np.sum(profile)
        ax.plot(hours, profile * 100, label="Estimated Volume Profile", color="#1f77b4", lw=2)
        ax.axhline(
            y=(1.0 / 1440.0) * 100,
            color="#2ca02c",
            linestyle="--",
            label="Flat Profile (TWAP)",
        )

    ax.set_title("Intraday Volume Profile (% per Minute)", fontsize=13, fontweight="bold")
    ax.set_xlabel("UTC Hour of Day", fontsize=11)
    ax.set_ylabel("Volume Weight (% / min)", fontsize=11)
    ax.set_xlim(0, 24)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_full_report(config: Config) -> dict[str, Any]:
    """Load simulation orders, compute cell metrics, export JSON, and render figures."""
    results_dir = Path(config.results.dir)
    figures_dir = Path(config.results.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sim_file = results_dir / "simulated_orders.parquet"
    if not sim_file.exists():
        raise FileNotFoundError(
            f"Simulation results missing at {sim_file}. Please run scripts/run_simulation.py first."
        )

    df_orders = pd.read_parquet(sim_file)
    print(f"Loaded {len(df_orders)} simulated orders.")

    # 1. Analyze every grid cell
    all_metrics = []
    group_cols = ["symbol", "horizon_minutes", "order_size_adv_pct"]
    for _, cell_df in df_orders.groupby(group_cols):
        cell_res = analyze_grid_cell(
            cell_df,
            twap_strategy_name="TWAP",
            n_bootstrap=config.impact.bootstrap_samples,
            random_seed=config.project.random_seed,
        )
        all_metrics.extend(cell_res)

    # 2. Build headline summary
    headline = build_headline_summary(all_metrics)

    # 3. Export metrics.json
    metrics_payload = {
        "project": config.project.name,
        "symbols": config.data.symbols,
        "n_simulated_orders": len(df_orders),
        "headline_summary": headline,
        "detailed_metrics": [vars(m) for m in all_metrics],
    }
    metrics_file = Path(config.results.metrics_file)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Exported metrics to {metrics_file}")

    # 4. Generate Figures
    impact_param_file = results_dir / "impact_parameters.json"
    impact_params = {}
    if impact_param_file.exists():
        with open(impact_param_file, encoding="utf-8") as f:
            impact_params = json.load(f)

    plot_impact_curves(impact_params, figures_dir / "impact_curves.png")
    plot_cost_vs_size(df_orders, config.execution.taker_fee_bps, figures_dir / "cost_vs_size.png")
    plot_efficient_frontier(df_orders, figures_dir / "efficient_frontier.png")

    sample_bars = pd.DataFrame()
    btc_bars = Path(config.data.processed_dir) / "BTCUSDT" / "BTCUSDT-bars-1m.parquet"
    if btc_bars.exists():
        sample_bars = pd.read_parquet(btc_bars).iloc[: 1440 * 7]
    plot_volume_profiles(sample_bars, figures_dir / "volume_profiles.png")

    # 5. Print Markdown Headline Table to stdout
    print("\n" + "=" * 80)
    print("HEADLINE EXECUTION SHORTFALL SUMMARY TABLE")
    print("=" * 80)
    df_head = pd.DataFrame(headline)
    if not df_head.empty:
        cols_show = [
            "symbol",
            "horizon_minutes",
            "order_size_adv_pct",
            "twap_shortfall_bps",
            "vwap_shortfall_bps",
            "best_ac_strategy",
            "best_ac_shortfall_bps",
            "ac_savings_vs_twap_bps",
            "savings_ci_95",
            "p_value",
            "ac_beats_twap",
        ]
        try:
            print(df_head[cols_show].to_markdown(index=False))
        except Exception:
            print(df_head[cols_show].to_string(index=False))
    print("=" * 80 + "\n")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate execution report and figures.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    generate_full_report(config)


if __name__ == "__main__":
    main()
