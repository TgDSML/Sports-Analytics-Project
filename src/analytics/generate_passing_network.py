"""Generate successful-pass pitch maps from possession outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle


TEAMS_CONFIG = {
    "Team A": {
        "color": "#ff3366",
        "filename": "passing_network_team_a_tactical.png",
    },
    "Team B": {
        "color": "#00ffcc",
        "filename": "passing_network_team_b_tactical.png",
    },
}


def generate_passing_networks(
    possession_csv: Path,
    possession_debug_csv: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Create one successful-pass pitch map per team."""
    possession_df = pd.read_csv(possession_csv)
    debug_df = pd.read_csv(possession_debug_csv)
    debug_by_frame = debug_df.set_index("frame", drop=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = {}

    for team_name, config in TEAMS_CONFIG.items():
        passes = _successful_passes_for_team(
            possession_df=possession_df,
            debug_by_frame=debug_by_frame,
            team_name=team_name,
        )
        output_path = output_dir / config["filename"]
        _plot_successful_passes(
            passes=passes,
            team_name=team_name,
            color=config["color"],
            output_path=output_path,
        )
        created[team_name] = output_path

    return created


def _successful_passes_for_team(
    possession_df: pd.DataFrame,
    debug_by_frame: pd.DataFrame,
    team_name: str,
) -> list[dict]:
    active = possession_df[possession_df["team"].isin(["Team A", "Team B"])].copy()
    if active.empty:
        return []

    active = active.dropna(subset=["nearest_player_id"]).copy()
    if active.empty:
        return []

    active["nearest_player_id"] = active["nearest_player_id"].astype(int)
    player_changed = active["nearest_player_id"] != active["nearest_player_id"].shift()
    transfers = active[player_changed].copy()
    passes = []

    previous = None
    for _, current in transfers.iterrows():
        if previous is not None and previous["team"] == current["team"] == team_name:
            if int(previous["nearest_player_id"]) != int(current["nearest_player_id"]):
                start = _player_point(debug_by_frame, int(previous["frame"]))
                end = _player_point(debug_by_frame, int(current["frame"]))
                if start is not None and end is not None:
                    passes.append(
                        {
                            "from_player": int(previous["nearest_player_id"]),
                            "to_player": int(current["nearest_player_id"]),
                            "start": start,
                            "end": end,
                        }
                    )
        previous = current

    return passes


def _player_point(debug_by_frame: pd.DataFrame, frame: int) -> tuple[float, float] | None:
    if frame not in debug_by_frame.index:
        return None
    row = debug_by_frame.loc[frame]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    x = row.get("nearest_player_center_x")
    y = row.get("nearest_player_center_y")
    if pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _plot_successful_passes(
    passes: list[dict],
    team_name: str,
    color: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    _draw_pitch(ax)

    for pass_row in passes:
        x1, y1 = pass_row["start"]
        x2, y2 = pass_row["end"]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.8,
                "alpha": 0.72,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=3,
        )
        ax.scatter([x1], [y1], color="#ffffff", s=12, alpha=0.85, zorder=4)
        ax.scatter([x2], [y2], color=color, s=18, alpha=0.95, zorder=4)

    ax.set_title(
        f"{team_name} Successful Pass Map ({len(passes)})",
        color="white",
        fontsize=16,
        pad=14,
    )
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#1d6f3a")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _draw_pitch(ax) -> None:
    pitch_w = 1280
    pitch_h = 720
    line_color = "white"
    lw = 2

    ax.add_patch(
        Rectangle((0, 0), pitch_w, pitch_h, fill=False, edgecolor=line_color, linewidth=lw)
    )
    ax.plot([pitch_w / 2, pitch_w / 2], [0, pitch_h], color=line_color, linewidth=lw)
    ax.add_patch(
        Circle((pitch_w / 2, pitch_h / 2), 73, fill=False, edgecolor=line_color, linewidth=lw)
    )
    ax.add_patch(Circle((pitch_w / 2, pitch_h / 2), 3, color=line_color))

    box_h = 324
    six_h = 146
    penalty_y = (pitch_h - box_h) / 2
    six_y = (pitch_h - six_h) / 2

    ax.add_patch(Rectangle((0, penalty_y), 166, box_h, fill=False, edgecolor=line_color, linewidth=lw))
    ax.add_patch(Rectangle((0, six_y), 55, six_h, fill=False, edgecolor=line_color, linewidth=lw))
    ax.add_patch(Circle((110, pitch_h / 2), 3, color=line_color))
    ax.add_patch(
        Arc((110, pitch_h / 2), 146, 146, angle=0, theta1=310, theta2=50, color=line_color, linewidth=lw)
    )

    ax.add_patch(
        Rectangle((pitch_w - 166, penalty_y), 166, box_h, fill=False, edgecolor=line_color, linewidth=lw)
    )
    ax.add_patch(
        Rectangle((pitch_w - 55, six_y), 55, six_h, fill=False, edgecolor=line_color, linewidth=lw)
    )
    ax.add_patch(Circle((pitch_w - 110, pitch_h / 2), 3, color=line_color))
    ax.add_patch(
        Arc((pitch_w - 110, pitch_h / 2), 146, 146, angle=0, theta1=130, theta2=230, color=line_color, linewidth=lw)
    )

    ax.set_xlim(0, pitch_w)
    ax.set_ylim(pitch_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate successful-pass pitch maps")
    parser.add_argument(
        "--possession-csv",
        type=Path,
        default=Path("outputs/possession_30s_720p.csv"),
    )
    parser.add_argument(
        "--possession-debug-csv",
        type=Path,
        default=Path("outputs/possession_debug_30s_720p.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created = generate_passing_networks(
        possession_csv=args.possession_csv,
        possession_debug_csv=args.possession_debug_csv,
        output_dir=args.output_dir,
    )
    if created:
        print("Passing maps saved:")
        for output_path in created.values():
            print(f"- {output_path}")
    else:
        print("No passing maps created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
