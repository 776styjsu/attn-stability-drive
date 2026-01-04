#!/usr/bin/env python3
"""
Temporal similarity analysis for saliency maps.

Compute similarity between consecutive frames' saliency maps under:
    <out_shap>/<TownXX>/<TownXX_XXXXXX>/saliency.png
Skip the next K frames after any respawn event. Optionally compute similarity
over the original image pairs.

Output CSV columns:
    town, prev_frame, curr_frame, prev_path, curr_path, prev_img_path, curr_img_path,
    saliency_metric, saliency_score, image_metric, image_score, steer_sim, note
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from attn_stability_drive.metrics.similarity import SimilarityComputer


def load_jsonl_records(jsonl_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Return (measurements, respawns) preserving insertion order."""
    measurements: List[Dict] = []
    respawns: List[Dict] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in obj and obj.get("event") == "respawn_teleport":
                if "frame" in obj and "town" in obj:
                    respawns.append(obj)
                continue
            # Measurement lines should have at least frame/town/image_path
            if "frame" in obj and "town" in obj and "image_path" in obj:
                measurements.append(obj)
    return measurements, respawns


def extract_frame_from_image_path(image_path: str) -> Optional[int]:
    """Extract frame (int) from '.../Town01_001313.png'."""
    m = re.search(r"_([0-9]+)\.(png|jpg|jpeg)$", image_path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def saliency_path(out_shap_root: Path, town: str, frame: int, pad: int,
                  filename: str = "saliency.png") -> Path:
    """Build path. Tries <out_shap>/<TownXX>/<TownXX_XXXXXX>/... and <out_shap>/<TownXX_XXXXXX>/..."""
    frame_dir = f"{town}__{frame:0{pad}d}"
    
    base = out_shap_root / town / frame_dir
    p = base / filename
    if p.exists():
        return p

    # If p does not exist, return the standard one (p) so the user sees what was expected
    return p

def resolve_image_path(jsonl_dir: Path, img_root: Optional[Path], image_path: str) -> Path:
    """Resolve relative image_path against --img-root or the JSONL's directory."""
    p = Path(image_path)
    if p.is_absolute():
        return p
    base = img_root if img_root is not None else jsonl_dir
    return (base / p).resolve()


def steer_similarity(s_prev: float, s_curr: float) -> float:
    """
    Similarity in [0,1] assuming steer ∈ [-1,1]: 1 - |Δ|/2.
    Clips inputs to [-1,1] and output to [0,1].
    """
    sp = max(-1.0, min(1.0, float(s_prev)))
    sc = max(-1.0, min(1.0, float(s_curr)))
    sim = 1.0 - abs(sc - sp) / 2.0
    return float(np.clip(sim, 0.0, 1.0))


def run_analysis(
    jsonl: Path,
    out_shap: Path,
    out_csv: Path,
    saliency_metric: str = "ssim",
    image_metric: str = "ssim",
    with_image_sim: bool = False,
    skip_after_respawn: int = 4,
    pad: int = 6,
    img_root: Optional[Path] = None,
    show_progress: bool = True,
) -> None:
    """
    Run temporal similarity analysis on saliency maps.
    
    Args:
        jsonl: Path to measurements.jsonl file.
        out_shap: Root directory of saliency outputs.
        out_csv: Where to write the output CSV.
        saliency_metric: Similarity metric for saliency maps ('ssim', 'fsim', 'lpips').
        image_metric: Similarity metric for original images ('ssim', 'fsim', 'lpips').
        with_image_sim: Also compute similarity on original image pairs.
        skip_after_respawn: Skip next K frames after respawn.
        pad: Zero-pad width for frame directories.
        img_root: Root for resolving relative image_path; default = jsonl parent.
        show_progress: Show tqdm progress bar.
    """
    jsonl_dir = jsonl.parent
    use_progress = show_progress
    with_image = with_image_sim
    image_metric_name = image_metric

    try:
        sal_metric = SimilarityComputer(saliency_metric)
    except (ImportError, ValueError) as exc:
        print(f"Failed to initialise saliency metric '{saliency_metric}': {exc}")
        return

    image_metric_computer: Optional[SimilarityComputer] = None
    if with_image:
        try:
            image_metric_computer = SimilarityComputer(image_metric_name)
        except (ImportError, ValueError) as exc:
            print(f"Failed to initialise image metric '{image_metric_name}': {exc}")
            return

    sal_prefer_gray = saliency_metric != "lpips"
    image_prefer_gray = (image_metric_name != "lpips")

    measurements, respawns = load_jsonl_records(jsonl)
    if not measurements:
        print("No measurement lines found; nothing to do.")
        return

    # Sort measurements by (town, frame). Use frame from record if present; else derive from image_path.
    # items: (town, frame, image_path, steer)
    items: List[Tuple[str, int, str, Optional[float]]] = []
    for m in measurements:
        town = m["town"]
        frame = m.get("frame", None)
        if frame is None:
            frame = extract_frame_from_image_path(m.get("image_path", ""))
        if frame is None:
            continue

        steer_val: Optional[float] = None
        ctrl = m.get("control", None)
        if isinstance(ctrl, dict) and "steer" in ctrl:
            try:
                steer_val = float(ctrl["steer"])
            except (TypeError, ValueError):
                steer_val = None

        items.append((town, int(frame), m["image_path"], steer_val))
    items.sort(key=lambda x: (x[0], x[1]))

    # Build a set of frames to skip per town after respawn.
    skip_after: Dict[str, Set[int]] = {}
    for e in respawns:
        town = e["town"]
        f = int(e["frame"])
        s = skip_after.setdefault(town, set())
        for k in range(1, skip_after_respawn + 1):
            s.add(f + k)

    rows: List[Dict] = []
    total_pairs = 0
    computed = 0
    skipped_respawn = 0
    skipped_missing = 0
    skipped_singleton = 0

    # Pre-group by town (so we can count candidate pairs for the progress bar)
    from itertools import groupby
    town2frames: Dict[str, List[Tuple[str, int, str, Optional[float]]]] = {}
    for town, group in groupby(items, key=lambda t: t[0]):
        town2frames[town] = list(group)

    candidate_pairs = sum(max(0, len(frames) - 1) for frames in town2frames.values())

    # Progress bar
    class _NoBar:
        def update(self, n): pass
        def close(self): pass

    pbar = tqdm(total=candidate_pairs, desc="Comparing frames", unit="pair",
                ncols=0, mininterval=0.3, disable=not use_progress) if use_progress else _NoBar()

    for town, frames in town2frames.items():
        if len(frames) < 2:
            skipped_singleton += 1
            continue

        # Preload paths/metadata for this town
        sal_paths: Dict[int, Path] = {}
        img_paths: Dict[int, Path] = {}
        steer_map: Dict[int, Optional[float]] = {}

        for _, fr, img_rel, steer_val in frames:
            sal_paths[fr] = saliency_path(out_shap, town, fr, pad)
            img_paths[fr] = resolve_image_path(jsonl_dir, img_root, img_rel)
            steer_map[fr] = steer_val

        # Compare consecutive frames
        for i in range(1, len(frames)):
            _, f_prev, _img_prev, _steer_prev = frames[i - 1]
            _, f_curr, _img_curr, _steer_curr = frames[i]
            total_pairs += 1

            # Skip window after respawn
            if f_curr in skip_after.get(town, set()):
                rows.append({
                    "town": town,
                    "prev_frame": f_prev,
                    "curr_frame": f_curr,
                    "prev_path": str(sal_paths.get(f_prev) or ""),
                    "curr_path": str(sal_paths.get(f_curr) or ""),
                    "prev_img_path": str(img_paths.get(f_prev, "")),
                    "curr_img_path": str(img_paths.get(f_curr, "")),
                    "saliency_metric": saliency_metric,
                    "saliency_score": "",
                    "image_metric": image_metric_name if with_image else "",
                    "image_score": "",
                    "steer_sim": "",
                    "note": "skipped: respawn_window"
                })
                skipped_respawn += 1
                pbar.update(1)
                continue

            p_prev = sal_paths.get(f_prev)
            p_curr = sal_paths.get(f_curr)
            note_parts: List[str] = []

            # Saliency similarity
            sal_val: Optional[float] = None
            if p_prev is None or p_curr is None or not (p_prev.exists() and p_curr.exists()):
                note_parts.append("missing_saliency")
            else:
                try:
                    sal_val = sal_metric.compute_pair(p_prev, p_curr, prefer_gray=sal_prefer_gray)
                except Exception:
                    note_parts.append("saliency_metric_error")

            # Image similarity (optional)
            img_val: Optional[float] = None
            if with_image:
                ip_prev = img_paths.get(f_prev)
                ip_curr = img_paths.get(f_curr)
                if ip_prev is None or ip_curr is None or not (ip_prev.exists() and ip_curr.exists()):
                    note_parts.append("missing_image")
                else:
                    try:
                        img_val = image_metric_computer.compute_pair(ip_prev, ip_curr, prefer_gray=image_prefer_gray) if image_metric_computer else None
                    except Exception:
                        note_parts.append("image_metric_error")

            # STEER similarity
            steer_val: Optional[float] = None
            s_prev = steer_map.get(f_prev, None)
            s_curr = steer_map.get(f_curr, None)
            if s_prev is None or s_curr is None:
                note_parts.append("missing_steer")
            else:
                try:
                    steer_val = steer_similarity(s_prev, s_curr)
                except Exception:
                    note_parts.append("steer_compute_error")

            has_any_metric = any([
                sal_val is not None,
                img_val is not None if with_image else False,
                steer_val is not None
            ])
            if has_any_metric:
                computed += 1
            else:
                skipped_missing += 1

            sal_score_str = "" if sal_val is None else f"{sal_val:.6f}"
            img_score_str = "" if img_val is None else f"{img_val:.6f}"

            rows.append({
                "town": town,
                "prev_frame": f_prev,
                "curr_frame": f_curr,
                "prev_path": str(p_prev or ""),
                "curr_path": str(p_curr or ""),
                "prev_img_path": str(img_paths.get(f_prev, "")),
                "curr_img_path": str(img_paths.get(f_curr, "")),
                "saliency_metric": saliency_metric,
                "saliency_score": sal_score_str,
                "image_metric": image_metric_name if with_image else "",
                "image_score": img_score_str,
                "steer_sim": "" if steer_val is None else f"{steer_val:.6f}",
                "note": ";".join(note_parts)
            })

            pbar.update(1)

    pbar.close()

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "town", "prev_frame", "curr_frame", "prev_path", "curr_path",
            "prev_img_path", "curr_img_path",
            "saliency_metric", "saliency_score",
            "image_metric", "image_score",
            "steer_sim", "note"
        ])
        w.writeheader()
        w.writerows(rows)

    # Summary
    print(f"Wrote: {out_csv}")
    print(f"Saliency metric: {sal_metric.display_name}")
    if with_image and image_metric_computer is not None:
        print(f"Image metric: {image_metric_computer.display_name}")
    print(f"Total candidate pairs: {candidate_pairs}")
    print(f"Total iterated pairs:  {total_pairs}")
    print(f"Computed (at least one metric): {computed}")
    print(f"Skipped (respawn):     {skipped_respawn}")
    print(f"Skipped (missing):     {skipped_missing}")
    print(f"Skipped (singleton towns): {skipped_singleton}")

