#!/usr/bin/env python3
"""
Collect driving data from CARLA using autopilot or navigation agents.

This script collects RGB camera images and vehicle telemetry while driving
autonomously in the CARLA simulator. The collected data can be used for
training driving models or analyzing saliency/attention maps.

Output Structure:
    <out>/<timestamp>/<TownXX>/
        images/
            Town01_000001.png
            Town01_000002.png
            ...
        measurements.jsonl

Usage:
    # Basic collection on current world
    python scripts/collect_data.py --out ./data

    # Collect on specific towns with behavior agent
    python scripts/collect_data.py --mode behavior --towns Town01,Town03 --steps 2000

    # Start CARLA automatically and collect on all towns
    python scripts/collect_data.py --start-carla --carla-path /opt/carla/CarlaUE4.sh --all-towns

Requirements:
    - CARLA simulator running (or use --start-carla)
    - CARLA Python API in PYTHONPATH
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carla

# =============================================================================
# Constants
# =============================================================================


def setup_carla_path(args: argparse.Namespace) -> None:
    """
    Setup sys.path to include CARLA 'agents' module.

    The 'agents' module is typically located in PythonAPI/carla/agents,
    so we need to add PythonAPI/carla to sys.path.
    """
    carla_root = os.environ.get("CARLA_ROOT")

    if not carla_root and args.carla_path:
        # args.carla_path points to CarlaUE4.sh, so root is the dir containing it
        carla_root = os.path.dirname(os.path.abspath(args.carla_path))

    if carla_root:
        # The agents package is inside PythonAPI/carla
        # e.g. .../CARLA_0.9.15/PythonAPI/carla/agents
        # So we add .../CARLA_0.9.15/PythonAPI/carla to sys.path
        agents_parent_dir = os.path.join(carla_root, "PythonAPI", "carla")
        if os.path.exists(agents_parent_dir) and agents_parent_dir not in sys.path:
            sys.path.append(agents_parent_dir)

PIDFILE_TMPL = "./carla_{port}.pid"
LOGFILE_TMPL = "./carla_{port}.log"

DEFAULT_RGB_WIDTH = 320
DEFAULT_RGB_HEIGHT = 180
DEFAULT_FPS = 10
DEFAULT_STEPS = 1200


# =============================================================================
# CARLA Server Management
# =============================================================================


def stop_carla(port: Optional[int] = None) -> None:
    """Stop a CARLA server we launched earlier (prefer pidfile; else restricted pkill)."""
    # 1) Try pidfile-based group kill (fast, no global scans)
    if port is not None:
        pidfile = PIDFILE_TMPL.format(port=port)
        if os.path.exists(pidfile):
            try:
                with open(pidfile) as f:
                    pid = int(f.read().strip())
                # Kill the whole process group started by Popen(start_new_session=True)
                os.killpg(pid, signal.SIGTERM)
                time.sleep(1.0)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                print(f"Stopped CARLA pgid {pid} (port {port})")
            except Exception as e:
                print(f"Warning: failed to kill CARLA by pidfile: {e}")
            finally:
                try:
                    os.remove(pidfile)
                except OSError:
                    pass
            return

    # 2) Fallback: restricted pkill (your user only, exact names), time-bounded
    user = os.environ.get("USER", "")
    for name in ("CarlaUE4-Linux-Shipping", "CarlaUE4.sh", "CarlaUE4"):
        for cmd_args in (
            ["pkill", "-u", user, "-x", name],
            ["pkill", "-9", "-u", user, "-x", name],
        ):
            try:
                subprocess.run(cmd_args, check=False, timeout=3)
            except subprocess.TimeoutExpired:
                print(f"pkill timed out for {name}; continuing")
    print("Stopped any existing CARLA servers (fallback)")


def _wait_for_carla(port: int, timeout: int = 120) -> bool:
    """Poll the RPC port until the server responds or timeout."""
    client = carla.Client("localhost", port)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.set_timeout(2.0)
            _ = client.get_server_version()
            return True
        except Exception:
            time.sleep(1.0)
    return False


def start_carla(args: argparse.Namespace) -> subprocess.Popen:
    """Start a CARLA server process."""
    carla_path = args.carla_path
    port = args.port
    gpu = args.gpu

    if not carla_path:
        raise ValueError(
            "You must pass --carla-path=/path/to/CarlaUE4.sh "
            "when --start-carla or --restart-carla is set."
        )

    cmd = [
        carla_path,
        f"-carla-rpc-port={port}",
        "-RenderOffScreen",
        "-nosound",
    ]

    if not gpu:
        cmd.append("-opengl")

    print(f"Starting CARLA: {' '.join(cmd)}")

    # Log to file so we can debug crashes
    log_path = LOGFILE_TMPL.format(port=port)
    logf = open(log_path, "ab", buffering=0)

    # Start in a new session so we can kill the whole group later
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
    )

    # Save pidfile for precise shutdowns
    pidfile = PIDFILE_TMPL.format(port=port)
    try:
        with open(pidfile, "w") as f:
            f.write(str(proc.pid))
    except Exception:
        pass

    print(f"Started CARLA on port {port}, PID={proc.pid}. Logs: {log_path}", flush=True)

    # Wait until the server is actually ready
    if not _wait_for_carla(port, timeout=120):
        raise RuntimeError(
            f"CARLA failed to come up on port {port}. Check log: {log_path}"
        )

    return proc


def restart_carla(args: argparse.Namespace) -> subprocess.Popen:
    """Restart CARLA server."""
    print("Restarting CARLA...", flush=True)
    stop_carla(args.port)
    return start_carla(args)


# =============================================================================
# Utility Functions
# =============================================================================


def ensure_dir(p: str) -> None:
    """Create directory if it doesn't exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def normalize_map_name(m: str) -> str:
    """Extract town name from full map path: '/Game/Carla/Maps/Town03' -> 'Town03'."""
    return Path(m).name


def disable_red_lights(world: carla.World) -> None:
    """Set all traffic lights to green permanently."""
    world.freeze_all_traffic_lights(True)
    for tl in world.get_actors().filter("traffic.traffic_light"):
        tl.set_state(carla.TrafficLightState.Green)
        tl.set_red_time(0.0)
        tl.set_yellow_time(0.0)
        tl.set_green_time(1e6)


# =============================================================================
# Actor Spawning
# =============================================================================


def spawn_ego(
    world: carla.World, bp_lib: carla.BlueprintLibrary
) -> Tuple[carla.Actor, int]:
    """Spawn ego vehicle at a random spawn point."""
    veh_bp = bp_lib.find("vehicle.tesla.model3")
    veh_bp.set_attribute("role_name", "hero")
    spawn_points = world.get_map().get_spawn_points()
    spawn_point_id = random.randint(0, len(spawn_points) - 1)
    print(f"Spawning ego at spawn point {spawn_point_id}")
    spawn = spawn_points[spawn_point_id]
    return world.spawn_actor(veh_bp, spawn), spawn_point_id


def attach_front_rgb(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    parent: carla.Actor,
    out_dir: str,
    width: int,
    height: int,
    fov: int = 90,
) -> Tuple[carla.Actor, deque, callable]:
    """Attach RGB camera to vehicle and set up image saving."""
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=parent)

    img_dir = os.path.join(out_dir, "images")
    ensure_dir(img_dir)
    q: deque = deque(maxlen=30)

    state = {"recording": True}

    def set_recording(flag: bool) -> None:
        state["recording"] = bool(flag)

    def _on_img(img: carla.Image) -> None:
        town_name = normalize_map_name(world.get_map().name)
        img.save_to_disk(os.path.join(img_dir, f"{town_name}_{img.frame:06d}.png"))
        q.append(img.frame)

    camera.listen(_on_img)
    return camera, q, set_recording


def spawn_npcs(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    tm: carla.TrafficManager,
    n: int = 40,
) -> List[carla.Actor]:
    """Spawn background traffic vehicles."""
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    actors: List[carla.Actor] = []
    for sp in spawn_points:
        if len(actors) >= n:
            break
        veh_bp = random.choice(bp_lib.filter("vehicle.*"))
        try:
            v = world.try_spawn_actor(veh_bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)
        except RuntimeError:
            pass
    return actors


# =============================================================================
# Navigation
# =============================================================================


def plan_random_route(agent: Any, grp: Any, spawn_point_id: int) -> int:
    """Plan a route from current spawn to a random destination."""
    spawn_points = agent._world.get_map().get_spawn_points()
    random_dest_id = random.randint(0, len(spawn_points) - 1)

    # Avoid picking the same point
    while random_dest_id == spawn_point_id:
        random_dest_id = random.randint(0, len(spawn_points) - 1)

    print(f"Planning route from sp {spawn_point_id} to sp {random_dest_id}")
    src = spawn_points[spawn_point_id].location
    dst = spawn_points[random_dest_id].location
    route = grp.trace_route(src, dst)
    agent.set_global_plan(route)
    return random_dest_id


def teleport_ego_to_random_spawn(
    world: carla.World, ego: carla.Actor, exclude_id: Optional[int] = None
) -> int:
    """
    Safely teleport the ego to a random spawn point.
    
    Resets velocity to avoid physics explosions.
    Returns the new spawn_point_id.
    """
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in map")

    new_id = random.randrange(len(spawn_points))
    if exclude_id is not None and len(spawn_points) > 1:
        while new_id == exclude_id:
            new_id = random.randrange(len(spawn_points))

    new_tf = spawn_points[new_id]

    # Temporarily disable physics, move, then re-enable + zero velocities
    try:
        ego.set_simulate_physics(False)
    except RuntimeError:
        pass

    ego.set_transform(new_tf)

    try:
        ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    except RuntimeError:
        pass

    try:
        ego.set_simulate_physics(True)
    except RuntimeError:
        pass

    # Let the world settle the teleport
    world.tick()
    world.tick()
    return new_id


# =============================================================================
# Data Collection
# =============================================================================


def collect_once_on_world(
    client: carla.Client,
    tm: carla.TrafficManager,
    args: argparse.Namespace,
    out_dir: str,
) -> None:
    """Run one data-collection episode on the current world."""
    world = client.get_world()
    town_name = normalize_map_name(world.get_map().name)

    # Enable synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()

    # Spawn ego vehicle and attach camera
    ego, spawn_point_id = spawn_ego(world, bp_lib)
    camera, img_queue, set_recording = attach_front_rgb(
        world, bp_lib, ego, out_dir, args.rgb_width, args.rgb_height
    )

    # Spawn background traffic
    traffic_actors = spawn_npcs(world, bp_lib, tm, n=args.n_npcs)

    # Set up control provider
    agent = None
    grp = None

    if args.mode == "tm":
        ego.set_autopilot(True, tm.get_port())
    elif args.mode == "basic":
        from agents.navigation.basic_agent import BasicAgent

        agent = BasicAgent(ego)
    elif args.mode == "behavior":
        from agents.navigation.behavior_agent import BehaviorAgent

        agent = BehaviorAgent(ego, behavior="normal")

    if agent is not None:
        agent.set_target_speed(args.target_speed_kmh)
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        grp = GlobalRoutePlanner(world.get_map(), 2.0)
        dest_id = plan_random_route(agent, grp, spawn_point_id)

    # Open measurements file
    meas_path = os.path.join(out_dir, "measurements.jsonl")
    meas_f = open(meas_path, "w", buffering=1)

    try:
        world.tick()  # Warm-up for sensors

        stall_timer = 0
        recording = True
        steps_done = 0
        ticks_seen = 0
        tick_cap = args.steps * 5

        while steps_done < args.steps and ticks_seen < tick_cap:
            # Run agent step if using navigation agent
            if agent is not None:
                if agent.done():
                    dest_id = plan_random_route(agent, grp, dest_id)
                control = agent.run_step()
                ego.apply_control(control)

            snapshot = world.tick()
            frame = snapshot
            ticks_seen += 1

            # Get vehicle state
            tf = ego.get_transform()
            vel = ego.get_velocity()
            spd_kmh = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6

            # Update stall timer
            if spd_kmh < args.min_speed_kmh:
                stall_timer += 1
            else:
                stall_timer = 0

            # Handle respawn on stall
            if args.respawn_on_stall and stall_timer >= args.stall_patience:
                prev_spawn = spawn_point_id
                print(
                    f"[stall] Respawning ego after {stall_timer} slow ticks "
                    f"(speed={spd_kmh:.2f} km/h).",
                    flush=True,
                )

                spawn_point_id = teleport_ego_to_random_spawn(
                    world, ego, exclude_id=prev_spawn
                )

                if agent is not None:
                    dest_id = plan_random_route(agent, grp, spawn_point_id)

                stall_timer = 0
                recording = True
                set_recording(True)

                # Log respawn event
                resp_evt = {
                    "event": "respawn_teleport",
                    "frame": frame,
                    "from_spawn_id": int(prev_spawn),
                    "to_spawn_id": int(spawn_point_id),
                    "town": town_name,
                }
                meas_f.write(json.dumps(resp_evt) + "\n")

            # Update recording state
            should_record = stall_timer < args.stall_patience
            if should_record != recording:
                recording = should_record
                set_recording(recording)

            # Wait for image to be saved
            t0 = time.time()
            while frame not in img_queue and time.time() - t0 < 2.0:
                time.sleep(0.001)

            # Log measurement record
            ctrl = ego.get_control()
            rec: Dict[str, Any] = {
                "frame": frame,
                "location": {
                    "x": tf.location.x,
                    "y": tf.location.y,
                    "z": tf.location.z,
                },
                "rotation": {
                    "pitch": tf.rotation.pitch,
                    "yaw": tf.rotation.yaw,
                    "roll": tf.rotation.roll,
                },
                "speed_kmh": spd_kmh,
                "control": {
                    "throttle": ctrl.throttle,
                    "steer": ctrl.steer,
                    "brake": ctrl.brake,
                    "reverse": ctrl.reverse,
                    "hand_brake": ctrl.hand_brake,
                    "manual_gear_shift": ctrl.manual_gear_shift,
                    "gear": ctrl.gear,
                },
                "image_path": f"images/{town_name}_{frame:06d}.png",
                "mode": args.mode,
                "town": town_name,
                "recording": recording,
            }
            meas_f.write(json.dumps(rec) + "\n")

            if recording:
                steps_done += 1

    finally:
        # Cleanup
        camera.stop()
        if agent is None:
            ego.set_autopilot(False)

        for a in traffic_actors:
            if a.is_alive:
                a.set_autopilot(False, tm.get_port())

        for a in traffic_actors:
            if a.is_alive:
                a.destroy()
        if ego.is_alive:
            ego.destroy()

        world.tick()
        meas_f.close()


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect driving data from CARLA using autopilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Control mode
    parser.add_argument(
        "--mode",
        choices=["tm", "basic", "behavior"],
        default="basic",
        help="Control mode: tm=TrafficManager, basic=BasicAgent, behavior=BehaviorAgent",
    )

    # Output
    parser.add_argument("--out", default="out_autopilot", help="Output directory")
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help="Number of frames to collect"
    )

    # Simulation
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Simulation FPS")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument(
        "--tm-port", type=int, default=8000, help="Traffic Manager port"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Traffic
    parser.add_argument(
        "--n-npcs", type=int, default=40, help="Number of NPC vehicles"
    )

    # Camera
    parser.add_argument(
        "--rgb-width", type=int, default=DEFAULT_RGB_WIDTH, help="Camera width"
    )
    parser.add_argument(
        "--rgb-height", type=int, default=DEFAULT_RGB_HEIGHT, help="Camera height"
    )

    # Towns
    parser.add_argument("--town", default=None, help="Run only on this town")
    parser.add_argument(
        "--towns", default=None, help="Comma-separated list of towns (e.g., Town01,Town03)"
    )
    parser.add_argument(
        "--all-towns", action="store_true", help="Run on every installed map"
    )

    # Traffic lights
    parser.add_argument(
        "--no-red-light", action="store_true", help="Set all traffic lights to green"
    )

    # Speed / stall handling
    parser.add_argument(
        "--min-speed-kmh",
        type=float,
        default=1.0,
        help="Minimum speed (km/h) to consider 'moving'",
    )
    parser.add_argument(
        "--target-speed-kmh",
        type=float,
        default=20,
        help="Target speed for navigation agents",
    )
    parser.add_argument(
        "--stall-patience",
        type=int,
        default=int(1e6),
        help="Consecutive slow frames before pausing recording",
    )
    parser.add_argument(
        "--respawn-on-stall",
        action="store_true",
        help="Respawn ego vehicle when stalled",
    )

    # CARLA server management
    parser.add_argument(
        "--start-carla", action="store_true", help="Start CARLA server"
    )
    parser.add_argument(
        "--restart-carla", action="store_true", help="Restart CARLA server"
    )
    parser.add_argument(
        "--gpu", type=bool, default=True, help="Use GPU rendering for CARLA"
    )
    parser.add_argument(
        "--carla-path", default=None, help="Path to CarlaUE4.sh script"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Ensure CARLA agents can be imported
    setup_carla_path(args)

    random.seed(args.seed)
    ensure_dir(args.out)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(args.out, timestamp_str)
    os.makedirs(root_out, exist_ok=True)

    # Start/restart CARLA if requested
    if args.restart_carla:
        restart_carla(args)
    elif args.start_carla:
        start_carla(args)

    # Connect to CARLA
    client = carla.Client("localhost", args.port)
    client.set_timeout(60.0)

    # Determine which towns to collect on
    if args.all_towns:
        towns = sorted({normalize_map_name(m) for m in client.get_available_maps()})
    elif args.towns:
        towns = [t.strip() for t in args.towns.split(",") if t.strip()]
    elif args.town:
        towns = [args.town]
    else:
        towns = [normalize_map_name(client.get_world().get_map().name)]

    # Set up Traffic Manager
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    tm.set_global_distance_to_leading_vehicle(2.5)

    try:
        for town in towns:
            print(f"\n{'='*60}")
            print(f"Collecting on {town}")
            print(f"{'='*60}")

            tm.set_synchronous_mode(True)

            world = client.load_world(town)
            time.sleep(10.0)
            world.tick()

            if args.no_red_light:
                disable_red_lights(world)

            town_out = os.path.join(root_out, town)
            ensure_dir(town_out)

            collect_once_on_world(client, tm, args, town_out)

    finally:
        print("\nCollection finished. Restoring async mode.")
        world = client.get_world()
        settings = world.get_settings()
        tm.set_synchronous_mode(False)
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

    print(f"\nSaved data to: {root_out}")


if __name__ == "__main__":
    main()
