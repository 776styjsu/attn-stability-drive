"""
JSONL data format utilities for CARLA driving data.

This module defines the data schema produced by the data collection scripts
and consumed by analysis pipelines.

Measurement Record Schema:
{
    "frame": int,                    # CARLA world tick / frame number
    "location": {"x": float, "y": float, "z": float},
    "rotation": {"pitch": float, "yaw": float, "roll": float},
    "speed_kmh": float,
    "control": {
        "throttle": float,           # [0, 1]
        "steer": float,              # [-1, 1]
        "brake": float,              # [0, 1]
        "reverse": bool,
        "hand_brake": bool,
        "manual_gear_shift": bool,
        "gear": int
    },
    "image_path": str,               # Relative path: "images/Town01_000123.png"
    "mode": str,                     # "tm" | "basic" | "behavior"
    "town": str,                     # e.g., "Town01"
    "recording": bool                # Whether this frame was actively recorded
}

Respawn Event Schema:
{
    "event": "respawn_teleport",
    "frame": int,
    "from_spawn_id": int,
    "to_spawn_id": int,
    "town": str
}
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


# Type aliases
MeasurementRecord = Dict[str, Any]
RespawnEvent = Dict[str, Any]


def iter_jsonl(path: Path) -> Generator[Dict[str, Any], None, None]:
    """Iterate over records in a JSONL file, skipping invalid lines."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all records from a JSONL file."""
    return list(iter_jsonl(path))


def load_measurements_and_events(
    jsonl_path: Path,
) -> Tuple[List[MeasurementRecord], List[RespawnEvent]]:
    """
    Load and separate measurement records from event records.
    
    Returns:
        (measurements, respawn_events) tuple
    """
    measurements: List[MeasurementRecord] = []
    respawns: List[RespawnEvent] = []
    
    for record in iter_jsonl(jsonl_path):
        # Check if it's an event record
        if "event" in record:
            if record.get("event") == "respawn_teleport":
                respawns.append(record)
            continue
        
        # Measurement records must have these fields
        if "frame" in record and "town" in record and "image_path" in record:
            measurements.append(record)
    
    return measurements, respawns


def get_steer_value(record: MeasurementRecord) -> Optional[float]:
    """Extract steering value from a measurement record."""
    ctrl = record.get("control")
    if isinstance(ctrl, dict) and "steer" in ctrl:
        try:
            return float(ctrl["steer"])
        except (TypeError, ValueError):
            return None
    return None


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
