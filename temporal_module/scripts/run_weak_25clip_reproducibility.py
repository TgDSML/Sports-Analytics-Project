"""Rebuild and train non-gold weak-label BiGRU baselines for current clips."""
from __future__ import annotations

import argparse, csv, json, subprocess, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CLASSES = ["background", "carry", "pass", "turnover", "shot"]
EVENT_CLASSES = ["carry", "pass", "turnover", "shot"]
ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "temporal_module" / "data" / "derived"
OUTPUTS = ROOT / "outputs"
REPRO = ROOT / "temporal_module" / "reproducibility" / "weak_25clip_v1"
SEED = 42
PRIMARY_STRIDE = 0.5
STRIDES = [0.25, 0.5, 1.0, 2.0]
LEAKAGE_TOKENS = ["event", "label", "candidate", "confidence_tier", "quality", "review", "gold", "manual", "score", "source", "priority", "overlap", "refinement"]
READINESS_FIELDS = ["clip_id","readiness_status","temporal_frames_path","unified_candidates_path","tracks_path","ball_tracks_path","team_assignments_path","possession_path","frame_count","unified_event_rows","normalized_candidate_events","background_possible","feature_column_count_probe","gold_dependency_detected","warnings","reasons"]

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def read_csv(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        return pd.read_csv(path, low_memory=False), ""
    except Exception as e:
        return None, str(e)

def norm_event(value: Any) -> str:
    text = str(value).strip().casefold()
    if text.startswith("pass"): return "pass"
    if text.startswith("turnover") or text.startswith("interception"): return "turnover"
    if text.startswith("shot"): return "shot"
    if text.startswith("carry"): return "carry"
    return ""

def bad_path(path: Path) -> bool:
    text = str(path).replace("\\", "/").casefold()
    return any(token in text for token in ["gold", "cvat", "manual", "override"])

def audit_clip(clip_dir: Path) -> dict[str, Any]:
    clip = clip_dir.name
    out = OUTPUTS / clip
    paths = {
        "temporal_frames": clip_dir / "temporal_frames.csv",
        "unified_candidates": clip_dir / "events" / "event_candidates_unified.csv",
        "tracks": out / "tracks" / "tracks.csv",
        "ball_tracks": out / "tracks" / "ball_tracks.csv",
        "team_assignments": out / "teams" / "player_teams.csv",
        "possession": out / "possession" / "possession.csv",
    }
    reasons, warnings = [], []
    gold_dep = any(bad_path(p) for p in paths.values())
    if gold_dep: reasons.append("prohibited gold/CVAT/manual path dependency detected")
    for name, p in paths.items():
        if not p.exists(): reasons.append(f"missing {name}: {p}")
    frame_count = feature_probe = background_possible = event_rows = events_ok = 0
    if paths["temporal_frames"].exists():
        df, err = read_csv(paths["temporal_frames"])
        if df is None: reasons.append(f"corrupted temporal frame table: {err}")
        elif "frame" not in df.columns: reasons.append("temporal frame table missing frame column")
        else:
            frames = pd.to_numeric(df["frame"], errors="coerce").dropna()
            if frames.empty: reasons.append("temporal frame table has no numeric frames")
            else:
                frame_count = int(frames.max() - frames.min() + 1)
                background_possible = int(frame_count >= 64)
            feature_probe = int(sum(pd.to_numeric(df[c], errors="coerce").notna().any() for c in df.columns if c != "frame"))
            if feature_probe == 0: reasons.append("no usable numeric temporal features found")
    if paths["unified_candidates"].exists():
        df, err = read_csv(paths["unified_candidates"])
        if df is None: reasons.append(f"corrupted unified candidate table: {err}")
        else:
            missing = sorted({"event_type","start_frame","end_frame","center_frame"} - set(df.columns))
            if missing: reasons.append("unified candidate table missing required columns: " + ", ".join(missing))
            event_rows = len(df)
            if "event_type" in df.columns: events_ok = int(df["event_type"].map(norm_event).isin(EVENT_CLASSES).sum())
            if events_ok == 0 and not background_possible: reasons.append("no valid candidate events and not enough frames for background")
            elif events_ok == 0: warnings.append("no valid candidate events; background only")
    checks = {"tracks":{"frame","track_id"}, "ball_tracks":{"frame","center_x","center_y"}, "team_assignments":{"track_id","team"}, "possession":{"frame"}}
    for name, required in checks.items():
        p = paths[name]
        if not p.exists(): continue
        df, err = read_csv(p)
        if df is None: reasons.append(f"corrupted {name}: {err}"); continue
        miss = sorted(required - set(df.columns))
        if miss: reasons.append(f"{name} missing required columns: " + ", ".join(miss))
        if df.empty: reasons.append(f"{name} is empty")
    status = "NOT_READY" if reasons else ("READY_WITH_WARNINGS" if warnings else "READY")
    return {"clip_id":clip,"readiness_status":status,"temporal_frames_path":str(paths["temporal_frames"]),"unified_candidates_path":str(paths["unified_candidates"]),"tracks_path":str(paths["tracks"]),"ball_tracks_path":str(paths["ball_tracks"]),"team_assignments_path":str(paths["team_assignments"]),"possession_path":str(paths["possession"]),"frame_count":frame_count,"unified_event_rows":event_rows,"normalized_candidate_events":events_ok,"background_possible":background_possible,"feature_column_count_probe":feature_probe,"gold_dependency_detected":int(gold_dep),"warnings":";".join(warnings),"reasons":";".join(reasons)}

def ensure_dirs() -> None:
    for name in ["readiness","datasets","runs","reports","commands"]:
        (REPRO / name).mkdir(parents=True, exist_ok=True)

def write_readiness(rows: list[dict[str, Any]]) -> None:
    out = REPRO / "readiness"
    write_csv(out / "weak_clip_readiness.csv", rows, READINESS_FIELDS)
    counts = Counter(r["readiness_status"] for r in rows)
    write_json(out / "weak_clip_readiness_summary.json", {"created_at":now(),"expected_clip_source":"temporal_module/data/derived non-gold clip directories","clips_expected":len(rows),"clips_ready":counts.get("READY",0),"clips_ready_with_warnings":counts.get("READY_WITH_WARNINGS",0),"clips_not_ready":counts.get("NOT_READY",0),"gold_label_dependency_detected":bool(any(int(r["gold_dependency_detected"]) for r in rows)),"forbidden_sources_used":[]})
    lines = ["# Weak 25-Clip Readiness Report", "", "This audit uses only original non-gold outputs from `outputs/` and `temporal_module/data/derived/`.", "No CVAT exports, gold intervals, gold windows, or manual annotation artifacts are used as labels.", "", f"- Expected clips: {len(rows)}", f"- READY: {counts.get('READY',0)}", f"- READY_WITH_WARNINGS: {counts.get('READY_WITH_WARNINGS',0)}", f"- NOT_READY: {counts.get('NOT_READY',0)}", "", "| clip_id | status | warnings | reasons |", "| --- | --- | --- | --- |"]
    for r in rows: lines.append(f"| {r['clip_id']} | {r['readiness_status']} | {r['warnings']} | {r['reasons']} |")
    (out / "weak_clip_readiness_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["readiness_status"] in {"READY","READY_WITH_WARNINGS"}]

def split_counts(n: int) -> tuple[int,int,int]:
    if n < 3: raise RuntimeError("Need at least 3 clips for train/val/test")
    val = max(1, round(n * 0.2)); test = max(1, round(n * 0.2)); train = n - val - test
    return int(train), int(val), int(test)

def run_cmd(name: str, cmd: list[str]) -> None:
    cdir = REPRO / "commands"; cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{name}.txt").write_text(" ".join(cmd)+"\n", encoding="utf-8")
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    (cdir / f"{name}.stdout.txt").write_text(res.stdout, encoding="utf-8")
    (cdir / f"{name}.stderr.txt").write_text(res.stderr, encoding="utf-8")
    if res.returncode != 0: raise RuntimeError(f"{name} failed with exit code {res.returncode}; see {cdir / (name + '.stderr.txt')}")

def build_dataset(stride: float, out: Path, clip_list: Path, train: int, val: int, test: int) -> None:
    cmd = [sys.executable,"temporal_module/scripts/build_weak_event_gru_dataset.py","--derived-root",str(DERIVED.relative_to(ROOT)),"--output-dir",str(out.relative_to(ROOT)),"--window-seconds","8.0","--label-region-seconds","1.0","--stride-seconds",str(stride),"--background-center-margin-seconds","1.0","--seed",str(SEED),"--train-clips",str(train),"--val-clips",str(val),"--test-clips",str(test),"--balance-strategy","none","--clip-list",str(clip_list.relative_to(ROOT))]
    run_cmd("build_dataset_stride_" + str(stride).replace(".","_"), cmd)

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def verify_dataset(path: Path) -> None:
    schema = load_json(path / "weak_event_feature_schema.json")
    features = [str(x) for x in schema.get("selected_feature_columns", [])]
    leak = [f for f in features if any(tok in f.casefold() for tok in LEAKAGE_TOKENS)]
    if leak: raise RuntimeError("Leakage feature columns detected: " + ", ".join(leak))
    split = pd.read_csv(path / "clip_split_manifest.csv")
    if split["clip_id"].duplicated().any(): raise RuntimeError("Duplicate clip in split manifest")
    sets = {s:set(split.loc[split["split"]==s,"clip_id"].astype(str)) for s in ["train","val","test"]}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]: raise RuntimeError("Split is not clip-disjoint")

def stride_stats(stride: float, dset: Path, readiness: list[dict[str, Any]]) -> dict[str, Any]:
    summary = load_json(dset / "weak_event_dataset_summary.json")
    windows = pd.read_csv(dset / "weak_event_windows.csv", low_memory=False)
    rej = pd.read_csv(dset / "weak_event_window_rejections.csv", low_memory=False)
    frames = pd.read_csv(dset / "weak_event_clip_frame_diagnostics.csv", low_memory=False)
    sel = windows[pd.to_numeric(windows["selected_after_balancing"], errors="coerce").fillna(0).astype(int)==1]
    counts = Counter(sel["target_class"].astype(str)); rc = Counter(r["readiness_status"] for r in readiness)
    bg_rej = rej[rej["window_kind"].astype(str)=="background"] if not rej.empty else rej
    reasons = bg_rej.get("reason", pd.Series(dtype=str)).astype(str)
    overlap = int((reasons=="center_label_region_overlaps_event").sum()); margin = int((reasons=="center_label_region_too_close_to_event").sum()); context = int((reasons=="accepted_with_noncentral_context_event").sum())
    bg = int(counts.get("background",0))
    return {"stride_seconds":stride,"clips_expected":len(readiness),"clips_ready":int(rc.get("READY",0)),"clips_ready_with_warnings":int(rc.get("READY_WITH_WARNINGS",0)),"clips_not_ready":int(rc.get("NOT_READY",0)),"clips_used":int(summary.get("processable_clips_used",0)),"total_frames_available":int(pd.to_numeric(frames.get("frame_count",0), errors="coerce").fillna(0).sum()),"event_windows_before_filtering":int((windows["target_class"]!="background").sum()),"event_windows_selected":int((sel["target_class"]!="background").sum()),"background_windows_considered":bg+overlap+margin,"background_windows_selected":bg,"background_windows_rejected_event_overlap":overlap,"background_windows_rejected_event_margin":margin,"background_windows_with_noncentral_event_context":context,"carry_windows":int(counts.get("carry",0)),"pass_windows":int(counts.get("pass",0)),"turnover_windows":int(counts.get("turnover",0)),"shot_windows":int(counts.get("shot",0)),"background_windows":bg,"total_windows":int(len(sel)),"padded_event_windows":int(summary.get("padded_event_windows",0)),"total_padding_frames":int(summary.get("padded_event_window_padding_frames_total",0)),"maximum_padding_percent":float(summary.get("padded_event_window_padding_percent_max",0.0)),"feature_count":int(summary.get("feature_count",0)),"class_balance_strategy":str(summary.get("balance_strategy","none")),"split_strategy":"clip_disjoint_train_validation_test","seed":SEED,"warnings":";".join(str(x) for x in summary.get("warnings",[]))}

def clip_stats(dset: Path, readiness: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows = pd.read_csv(dset / "weak_event_windows.csv", low_memory=False)
    frames = pd.read_csv(dset / "weak_event_clip_frame_diagnostics.csv", low_memory=False)
    frame_map = {str(r.clip_id): r for r in frames.itertuples(index=False)}
    out = []
    for r in readiness:
        clip = r["clip_id"]
        cw = windows[windows["clip_id"].astype(str)==clip] if not windows.empty else pd.DataFrame()
        sel = cw[pd.to_numeric(cw.get("selected_after_balancing",0), errors="coerce").fillna(0).astype(int)==1] if not cw.empty else cw
        counts = Counter(sel.get("target_class", pd.Series(dtype=str)).astype(str)); fr = frame_map.get(clip)
        event_total = sum(int(counts.get(c,0)) for c in EVENT_CLASSES)
        out.append({"clip_id":clip,"readiness_status":r["readiness_status"],"included_in_dataset":int(r["readiness_status"] in {"READY","READY_WITH_WARNINGS"}),"fps":getattr(fr,"fps","") if fr else "","frame_count":getattr(fr,"frame_count",r.get("frame_count","")) if fr else r.get("frame_count",""),"event_count_total":event_total,"carry_event_count":int(counts.get("carry",0)),"pass_event_count":int(counts.get("pass",0)),"turnover_event_count":int(counts.get("turnover",0)),"shot_event_count":int(counts.get("shot",0)),"background_windows":int(counts.get("background",0)),"event_windows":event_total,"total_windows":int(len(sel)),"padded_event_windows":int((pd.to_numeric(sel.get("padding_frames",0), errors="coerce").fillna(0)>0).sum()) if not sel.empty else 0,"padding_frames":int(pd.to_numeric(sel.get("padding_frames",0), errors="coerce").fillna(0).sum()) if not sel.empty else 0,"warnings":r.get("warnings",""),"exclusion_reason":r.get("reasons","") if r["readiness_status"]=="NOT_READY" else ""})
    return out

def train_run(name: str, dset: Path, epochs: int, weighted: bool) -> None:
    run = REPRO / "runs" / name
    cmd = [sys.executable,"temporal_module/scripts/train_weak_event_gru.py","--dataset-dir",str(dset.relative_to(ROOT)),"--derived-root",str(DERIVED.relative_to(ROOT)),"--model-dir",str((run/"model").relative_to(ROOT)),"--report-dir",str((run/"report").relative_to(ROOT)),"--seed",str(SEED),"--epochs",str(epochs),"--batch-size","8","--hidden-size","32","--dropout","0.2","--learning-rate","0.001","--patience","8"]
    cmd += ["--loss","focal","--focal-gamma","2.0","--class-weighting","inverse_frequency","--sampling","weighted"] if weighted else ["--loss","cross_entropy","--class-weighting","none","--sampling","none"]
    run_cmd("train_"+name, cmd)

def model_summary(name: str, dset: Path, readiness: list[dict[str, Any]]) -> dict[str, Any]:
    run = REPRO / "runs" / name
    cfg = load_json(run / "model" / "experiment_config.json"); metrics = load_json(run / "report" / "evaluation_metrics.json")
    hist = pd.read_csv(run / "report" / "training_history.csv"); per = pd.read_csv(run / "report" / "per_class_metrics.csv")
    windows = pd.read_csv(dset / "weak_event_windows.csv", low_memory=False)
    sel = windows[pd.to_numeric(windows["selected_after_balancing"], errors="coerce").fillna(0).astype(int)==1]
    split = pd.read_csv(dset / "clip_split_manifest.csv"); rc = Counter(r["readiness_status"] for r in readiness); sc = Counter(sel["split"].astype(str))
    macro_col = "validation_macro_f1" if "validation_macro_f1" in hist.columns else "macro_f1"; acc_col = "validation_accuracy" if "validation_accuracy" in hist.columns else "val_accuracy"
    best = hist.iloc[int(hist[macro_col].idxmax())]; final = hist.iloc[-1]
    test = metrics.get("test",{}); tper = per[per["split"]=="test"]; f1 = {str(r.target_class):float(r.f1) for r in tper.itertuples(index=False)}
    return {"model_run":name,"label_source":"heuristic_unified_event_candidates_only","dataset_name":dset.name,"clips_expected":len(readiness),"clips_ready":int(rc.get("READY",0)+rc.get("READY_WITH_WARNINGS",0)),"clips_used":int(split["clip_id"].nunique()),"clips_excluded":int(rc.get("NOT_READY",0)),"feature_count":int(cfg.get("feature_count",0)),"window_duration_seconds":8.0,"label_region_seconds":1.0,"background_stride_seconds":PRIMARY_STRIDE,"total_windows":int(len(sel)),"train_windows":int(sc.get("train",0)),"validation_windows":int(sc.get("val",0)),"test_windows":int(sc.get("test",0)),"train_clips":int((split["split"]=="train").sum()),"validation_clips":int((split["split"]=="val").sum()),"test_clips":int((split["split"]=="test").sum()),"epochs_requested":int(cfg.get("epochs_requested",0)),"epochs_run":int(cfg.get("epochs_run",len(hist))),"best_validation_epoch":int(best.get("epoch",1)),"best_validation_accuracy":float(best.get(acc_col,0.0)),"best_validation_macro_f1":float(best.get(macro_col,0.0)),"final_validation_accuracy":float(final.get(acc_col,0.0)),"final_validation_macro_f1":float(final.get(macro_col,0.0)),"test_accuracy":float(test.get("accuracy",0.0)),"test_macro_f1":float(test.get("macro_f1",0.0)),"background_f1":f1.get("background",0.0),"carry_f1":f1.get("carry",0.0),"pass_f1":f1.get("pass",0.0),"turnover_f1":f1.get("turnover",0.0),"shot_f1":f1.get("shot",0.0),"loss":str(cfg.get("loss","")),"focal_gamma":float(cfg.get("focal_gamma",0.0)),"class_weighting":str(cfg.get("class_weighting","")),"sampling":str(cfg.get("sampling","")),"seed":SEED,"gold_label_dependency_detected":0,"warnings":";".join(str(x) for x in metrics.get("warnings",[]))}

def class_dist(names: list[str], dset: Path) -> list[dict[str, Any]]:
    w = pd.read_csv(dset / "weak_event_windows.csv", low_memory=False); s = w[pd.to_numeric(w["selected_after_balancing"], errors="coerce").fillna(0).astype(int)==1]
    rows=[]
    for name in names:
        for split in ["train","val","test"]:
            sw=s[s["split"]==split]; total=max(1,len(sw)); clips=int(sw["clip_id"].nunique()) if not sw.empty else 0; counts=Counter(sw["target_class"].astype(str))
            for cls in CLASSES: rows.append({"model_run":name,"split":split,"class_name":cls,"window_count":int(counts.get(cls,0)),"class_proportion":int(counts.get(cls,0))/total,"clip_count":clips})
    return rows

def final_report(model_rows: list[dict[str, Any]], stride_rows: list[dict[str, Any]], readiness: list[dict[str, Any]]) -> None:
    rc=Counter(r["readiness_status"] for r in readiness)
    lines=["# Weak 25-Clip Reproducibility Report","","Labels come only from heuristic original-pipeline candidate artifacts. No CVAT intervals, gold windows, gold annotations, manual overrides, or reviewed labels were used.","",f"- Expected clips: {len(readiness)}",f"- Usable clips: {rc.get('READY',0)+rc.get('READY_WITH_WARNINGS',0)}",f"- Excluded clips: {rc.get('NOT_READY',0)}","- Clip-disjoint split: verified","- Gold/manual dependency detected: false","","## Stride Diagnostics","","| stride_seconds | total_windows | background | carry | pass | turnover | shot |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in stride_rows: lines.append(f"| {r['stride_seconds']} | {r['total_windows']} | {r['background_windows']} | {r['carry_windows']} | {r['pass_windows']} | {r['turnover_windows']} | {r['shot_windows']} |")
    lines += ["","## Model Runs","","| run | loss | sampling | test_accuracy | test_macro_f1 | epochs_run |","| --- | --- | --- | ---: | ---: | ---: |"]
    for r in model_rows: lines.append(f"| {r['model_run']} | {r['loss']} | {r['sampling']} | {r['test_accuracy']:.6f} | {r['test_macro_f1']:.6f} | {r['epochs_run']} |")
    lines += ["","## Limitations","","These are weak-label baselines trained on heuristic labels. They are not production-ready and are not gold-supervised."]
    (REPRO/"reports"/"WEAK_25CLIP_REPRODUCIBILITY_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--skip-training", action="store_true"); p.add_argument("--epochs", type=int, default=40); args=p.parse_args()
    ensure_dirs(); readiness=[audit_clip(d) for d in sorted(x for x in DERIVED.iterdir() if x.is_dir())]; write_readiness(readiness)
    ready=ready_rows(readiness); inc=REPRO/"datasets"/"included_clips.txt"; exc=REPRO/"datasets"/"excluded_clips.txt"
    inc.write_text("\n".join(r["clip_id"] for r in ready)+"\n", encoding="utf-8"); exc.write_text("\n".join(r["clip_id"] for r in readiness if r["readiness_status"]=="NOT_READY")+"\n", encoding="utf-8")
    train,val,test=split_counts(len(ready)); primary=REPRO/"datasets"/"weak_event_gru_all_processable"; stride_rows=[]
    for stride in STRIDES:
        dset = primary if stride == PRIMARY_STRIDE else REPRO/"datasets"/f"stride_{str(stride).replace('.','_')}"
        build_dataset(stride,dset,inc,train,val,test); verify_dataset(dset); stride_rows.append(stride_stats(stride,dset,readiness))
    stride_fields=["stride_seconds","clips_expected","clips_ready","clips_ready_with_warnings","clips_not_ready","clips_used","total_frames_available","event_windows_before_filtering","event_windows_selected","background_windows_considered","background_windows_selected","background_windows_rejected_event_overlap","background_windows_rejected_event_margin","background_windows_with_noncentral_event_context","carry_windows","pass_windows","turnover_windows","shot_windows","background_windows","total_windows","padded_event_windows","total_padding_frames","maximum_padding_percent","feature_count","class_balance_strategy","split_strategy","seed","warnings"]
    write_csv(REPRO/"datasets"/"window_generation_statistics_by_stride.csv", stride_rows, stride_fields)
    write_csv(REPRO/"datasets"/"window_generation_statistics_by_clip.csv", clip_stats(primary,readiness), ["clip_id","readiness_status","included_in_dataset","fps","frame_count","event_count_total","carry_event_count","pass_event_count","turnover_event_count","shot_event_count","background_windows","event_windows","total_windows","padded_event_windows","padding_frames","warnings","exclusion_reason"])
    model_rows=[]; names=[]
    if not args.skip_training:
        train_run("weak_bigru_baseline", primary, args.epochs, False); train_run("weak_bigru_weighted_focal", primary, args.epochs, True); names=["weak_bigru_baseline","weak_bigru_weighted_focal"]
        model_rows=[model_summary(n,primary,readiness) for n in names]
        fields=["model_run","label_source","dataset_name","clips_expected","clips_ready","clips_used","clips_excluded","feature_count","window_duration_seconds","label_region_seconds","background_stride_seconds","total_windows","train_windows","validation_windows","test_windows","train_clips","validation_clips","test_clips","epochs_requested","epochs_run","best_validation_epoch","best_validation_accuracy","best_validation_macro_f1","final_validation_accuracy","final_validation_macro_f1","test_accuracy","test_macro_f1","background_f1","carry_f1","pass_f1","turnover_f1","shot_f1","loss","focal_gamma","class_weighting","sampling","seed","gold_label_dependency_detected","warnings"]
        write_csv(REPRO/"reports"/"weak_25clip_model_run_summary.csv", model_rows, fields); write_csv(REPRO/"reports"/"class_distribution_by_split.csv", class_dist(names,primary), ["model_run","split","class_name","window_count","class_proportion","clip_count"]); final_report(model_rows,stride_rows,readiness)
    print(f"Readiness report: {REPRO/'readiness'/'weak_clip_readiness_report.md'}"); print(f"Primary dataset: {primary}"); print(f"Stride stats: {REPRO/'datasets'/'window_generation_statistics_by_stride.csv'}")
    if model_rows: print(f"Run summary: {REPRO/'reports'/'weak_25clip_model_run_summary.csv'}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
