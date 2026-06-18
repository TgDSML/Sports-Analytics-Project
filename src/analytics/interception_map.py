import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

PLAYER_SUMMARY_COLUMNS = [
    "player_id",
    "team",
    "interception_count",
    "avg_ball_x",
    "avg_ball_y",
]

TEAM_FILE_MAP = {
    "Team A": "interception_map_team_a.png",
    "Team B": "interception_map_team_b.png",
}


def build_interception_maps(
    interceptions_csv_path: Path,
    output_dir: Path,
    team_a_color: str = "#00d4ff",
    team_b_color: str = "#ff5a5a",
    point_alpha: float = 0.85,
) -> dict:
    rows = _read_csv(
        interceptions_csv_path,
        required={"winner_player_id", "winner_team", "ball_x", "ball_y"},
    )

    clean_rows = [
        row for row in rows
        if row.get("winner_team") in {"Team A", "Team B"}
        and _float_or_none(row.get("ball_x")) is not None
        and _float_or_none(row.get("ball_y")) is not None
    ]

    by_team = defaultdict(list)
    for row in clean_rows:
        by_team[row["winner_team"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Team A": team_a_color, "Team B": team_b_color}
    created = {}
    for team, filename in TEAM_FILE_MAP.items():
        team_rows = by_team.get(team, [])
        file_path = output_dir / filename
        _plot_team_interceptions(
            rows=team_rows,
            team=team,
            color=colors[team],
            point_alpha=point_alpha,
            output_path=file_path,
        )
        created[team] = file_path

    summary_rows = _build_player_summary(clean_rows)
    summary_path = output_dir / "interception_players_summary.csv"
    _write_rows(summary_rows, summary_path, PLAYER_SUMMARY_COLUMNS)
    created["summary_csv"] = summary_path

    print(f"Team A interception map saved to: {created['Team A']}")
    print(f"Team B interception map saved to: {created['Team B']}")
    print(f"Interception players summary saved to: {created['summary_csv']}")
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


def _plot_team_interceptions(
    rows: list[dict],
    team: str,
    color: str,
    point_alpha: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    _draw_pitch(ax)

    xs = []
    ys = []
    for row in rows:
        x = _float_or_none(row.get("ball_x"))
        y = _float_or_none(row.get("ball_y"))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    if xs and ys:
        ax.scatter(xs, ys, s=48, color=color, edgecolors="white", linewidths=0.8, alpha=point_alpha, zorder=3)

    ax.set_title(f"{team} Interceptions", color="white", fontsize=16, pad=14)
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
        grouped[(row["winner_player_id"], row["winner_team"])].append(row)

    summary = []
    for (player_id, team), items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0][1], kv[0][0])):
        xs = [_float_or_none(item.get("ball_x")) for item in items]
        ys = [_float_or_none(item.get("ball_y")) for item in items]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        summary.append(
            {
                "player_id": player_id,
                "team": team,
                "interception_count": len(items),
                "avg_ball_x": f"{(sum(xs) / len(xs)) if xs else 0.0:.2f}",
                "avg_ball_y": f"{(sum(ys) / len(ys)) if ys else 0.0:.2f}",
            }
        )
    return summary


def _float_or_none(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create interception maps and player interception summary")
    parser.add_argument("--interceptions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--team-a-color", type=str, default="#00d4ff")
    parser.add_argument("--team-b-color", type=str, default="#ff5a5a")
    parser.add_argument("--point-alpha", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_interception_maps(
        interceptions_csv_path=args.interceptions_csv,
        output_dir=args.output_dir,
        team_a_color=args.team_a_color,
        team_b_color=args.team_b_color,
        point_alpha=args.point_alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())