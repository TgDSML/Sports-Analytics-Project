import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

PLAYER_SUMMARY_COLUMNS = [
    "player_id",
    "team",
    "carry_count",
    "total_carry_distance_px",
    "avg_carry_distance_px",
    "max_carry_distance_px",
    "avg_carry_duration_s",
    "max_carry_duration_s",
]

TEAM_FILE_MAP = {
    "Team A": "carry_map_team_a.png",
    "Team B": "carry_map_team_b.png",
}


def build_carry_maps(
    carry_events_csv_path: Path,
    output_dir: Path,
    team_a_color: str = "#00d4ff",
    team_b_color: str = "#ff5a5a",
    line_alpha: float = 0.65,
    min_distance_px: float = 0.0,
) -> dict:
    carry_rows = _read_csv(
        carry_events_csv_path,
        required={
            "carry_id", "player_id", "team", "start_ball_x", "start_ball_y",
            "end_ball_x", "end_ball_y", "ball_distance_px", "duration_seconds"
        },
    )
    filtered = [r for r in carry_rows if _float(r.get("ball_distance_px")) >= min_distance_px]
    by_team = defaultdict(list)
    for row in filtered:
        by_team[row["team"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Team A": team_a_color, "Team B": team_b_color}
    created = {}
    for team, filename in TEAM_FILE_MAP.items():
        rows = by_team.get(team, [])
        file_path = output_dir / filename
        _plot_team_carries(rows, team=team, color=colors[team], output_path=file_path)
        created[team] = file_path

    summary_rows = _build_player_summary(filtered)
    summary_path = output_dir / "carry_players_summary.csv"
    _write_rows(summary_rows, summary_path, PLAYER_SUMMARY_COLUMNS)
    created["summary_csv"] = summary_path
    print(f"Team A carry map saved to: {created['Team A']}")
    print(f"Team B carry map saved to: {created['Team B']}")
    print(f"Carry players summary saved to: {created['summary_csv']}")
    return created


def _read_csv(path: Path, required: set[str]) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")
        return list(reader)


def _write_rows(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_team_carries(rows: list[dict], team: str, color: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    _draw_pitch(ax)

    for row in rows:
        x1 = _float_or_none(row.get("start_ball_x"))
        y1 = _float_or_none(row.get("start_ball_y"))
        x2 = _float_or_none(row.get("end_ball_x"))
        y2 = _float_or_none(row.get("end_ball_y"))
        if None in {x1, y1, x2, y2}:
            continue
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.0, alpha=0.65)
        ax.scatter([x1], [y1], color="#ffffff", s=14, alpha=0.85, zorder=3)
        ax.scatter([x2], [y2], color=color, s=20, alpha=0.95, zorder=3)

    ax.set_title(f"{team} Carries", color="white", fontsize=16, pad=14)
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

    ax.add_patch(Rectangle((0, 0), pitch_w, pitch_h, fill=False, edgecolor=line_color, linewidth=lw))
    ax.plot([pitch_w / 2, pitch_w / 2], [0, pitch_h], color=line_color, linewidth=lw)
    ax.add_patch(Circle((pitch_w / 2, pitch_h / 2), 73, fill=False, edgecolor=line_color, linewidth=lw))
    ax.add_patch(Circle((pitch_w / 2, pitch_h / 2), 3, color=line_color))

    box_h = 324
    six_h = 146
    penalty_y = (pitch_h - box_h) / 2
    six_y = (pitch_h - six_h) / 2

    ax.add_patch(Rectangle((0, penalty_y), 166, box_h, fill=False, edgecolor=line_color, linewidth=lw))
    ax.add_patch(Rectangle((0, six_y), 55, six_h, fill=False, edgecolor=line_color, linewidth=lw))
    ax.add_patch(Circle((110, pitch_h / 2), 3, color=line_color))
    ax.add_patch(Arc((110, pitch_h / 2), 146, 146, angle=0, theta1=310, theta2=50, color=line_color, linewidth=lw))

    ax.add_patch(Rectangle((pitch_w - 166, penalty_y), 166, box_h, fill=False, edgecolor=line_color, linewidth=lw))
    ax.add_patch(Rectangle((pitch_w - 55, six_y), 55, six_h, fill=False, edgecolor=line_color, linewidth=lw))
    ax.add_patch(Circle((pitch_w - 110, pitch_h / 2), 3, color=line_color))
    ax.add_patch(Arc((pitch_w - 110, pitch_h / 2), 146, 146, angle=0, theta1=130, theta2=230, color=line_color, linewidth=lw))

    ax.set_xlim(0, pitch_w)
    ax.set_ylim(pitch_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def _build_player_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["player_id"], row["team"])].append(row)

    summary = []
    for (player_id, team), items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0][1], kv[0][0])):
        distances = [_float(r.get("ball_distance_px")) for r in items]
        durations = [_float(r.get("duration_seconds")) for r in items]
        summary.append(
            {
                "player_id": player_id,
                "team": team,
                "carry_count": len(items),
                "total_carry_distance_px": f"{sum(distances):.2f}",
                "avg_carry_distance_px": f"{(sum(distances) / len(distances)) if distances else 0.0:.2f}",
                "max_carry_distance_px": f"{max(distances) if distances else 0.0:.2f}",
                "avg_carry_duration_s": f"{(sum(durations) / len(durations)) if durations else 0.0:.3f}",
                "max_carry_duration_s": f"{max(durations) if durations else 0.0:.3f}",
            }
        )
    return summary


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create carry maps and player carry summary")
    parser.add_argument("--carry-events-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--team-a-color", type=str, default="#00d4ff")
    parser.add_argument("--team-b-color", type=str, default="#ff5a5a")
    parser.add_argument("--line-alpha", type=float, default=0.65)
    parser.add_argument("--min-distance-px", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_carry_maps(
        carry_events_csv_path=args.carry_events_csv,
        output_dir=args.output_dir,
        team_a_color=args.team_a_color,
        team_b_color=args.team_b_color,
        line_alpha=args.line_alpha,
        min_distance_px=args.min_distance_px,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())