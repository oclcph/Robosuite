"""
Generate a top-down SVG reachability map for MultiArmBlockLift.

Unlike a simple radius sketch, this script samples real robot joint
configurations inside MuJoCo joint limits, runs forward kinematics, and plots
the end-effector positions that fall in a requested height band above the
table.
"""

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np

import robosuite


COLORS = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]


def world_to_svg(x, y, bounds, canvas_size, margin):
    min_x, max_x, min_y, max_y = bounds
    width = canvas_size - 2 * margin
    height = canvas_size - 2 * margin
    sx = margin + (x - min_x) / (max_x - min_x) * width
    sy = margin + (max_y - y) / (max_y - min_y) * height
    return sx, sy


def fmt(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def svg_rect(parts, x, y, w, h, fill, stroke=None, opacity=1.0, extra=""):
    stroke_attr = "" if stroke is None else f' stroke="{stroke}"'
    parts.append(
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
        f'fill="{fill}" fill-opacity="{fmt(opacity)}"{stroke_attr} {extra}/>'
    )


def make_square_bounds(points, table_full_size, padding=0.12):
    table_x, table_y, _ = table_full_size
    xs = [-table_x / 2, table_x / 2]
    ys = [-table_y / 2, table_y / 2]
    for robot_points in points:
        if len(robot_points) > 0:
            xs.extend(robot_points[:, 0].tolist())
            ys.extend(robot_points[:, 1].tolist())

    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_y, max_y = min(ys) - padding, max(ys) + padding
    span = max(max_x - min_x, max_y - min_y)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    return mid_x - span / 2, mid_x + span / 2, mid_y - span / 2, mid_y + span / 2


def occupied_cells(points, resolution):
    if len(points) == 0:
        return set()
    cells = np.floor(points[:, :2] / resolution).astype(int)
    return {tuple(cell) for cell in cells}


def get_robot_collision_geom_ids(env, robot):
    names = list(robot.robot_model.contact_geoms)
    for arm in robot.arms:
        if robot.has_gripper[arm]:
            names.extend(robot.gripper[arm].contact_geoms)

    ids = []
    for name in names:
        try:
            ids.append(env.sim.model.geom_name2id(name))
        except ValueError:
            pass
    return np.array(sorted(set(ids)), dtype=int)


def sample_robot_reach(env, samples_per_robot, z_min, z_max, seed, camera_name=None):
    rng = np.random.default_rng(seed)
    sim = env.sim
    table_z = float(env.table_offset[2])
    robot_points = []
    robot_geom_points = []
    nearest_camera_distance = None
    camera_pos = None
    if camera_name is not None:
        try:
            camera_pos = np.array(sim.data.get_camera_xpos(camera_name), dtype=float)
        except ValueError:
            camera_pos = None

    for robot in env.robots:
        arm = robot.arms[0]
        qpos_indexes = np.array(robot._ref_arm_joint_pos_indexes, dtype=int)
        joint_indexes = np.array(robot._ref_arm_joint_indexes, dtype=int)
        joint_ranges = np.array(sim.model.jnt_range[joint_indexes], dtype=float)
        geom_ids = get_robot_collision_geom_ids(env, robot)

        low = joint_ranges[:, 0]
        high = joint_ranges[:, 1]
        limited = np.isfinite(low) & np.isfinite(high) & (high > low)
        if not np.all(limited):
            init_qpos = np.array(robot.init_qpos[: len(qpos_indexes)], dtype=float)
            low = np.where(limited, low, init_qpos - math.pi)
            high = np.where(limited, high, init_qpos + math.pi)

        accepted = []
        geom_points = []
        original_qpos = np.array(sim.data.qpos[qpos_indexes])
        for _ in range(samples_per_robot):
            sim.data.qpos[qpos_indexes] = rng.uniform(low, high)
            sim.forward()
            eef_pos = np.array(sim.data.site_xpos[robot.eef_site_id[arm]])
            if table_z + z_min <= eef_pos[2] <= table_z + z_max:
                accepted.append(eef_pos.copy())
            if len(geom_ids) > 0:
                centers = np.array(sim.data.geom_xpos[geom_ids], dtype=float)
                geom_points.append(centers)
                if camera_pos is not None:
                    distances = np.linalg.norm(centers - camera_pos, axis=1)
                    current_min = float(np.min(distances))
                    nearest_camera_distance = (
                        current_min
                        if nearest_camera_distance is None
                        else min(nearest_camera_distance, current_min)
                    )

        sim.data.qpos[qpos_indexes] = original_qpos
        sim.forward()
        robot_points.append(np.array(accepted))
        robot_geom_points.append(np.concatenate(geom_points, axis=0) if geom_points else np.empty((0, 3)))

    return robot_points, robot_geom_points, camera_pos, nearest_camera_distance


def write_svg(path, robot_points, args):
    canvas = args.canvas_size
    margin = 70
    table_x, table_y, _ = args.table_full_size
    bounds = make_square_bounds(robot_points, args.table_full_size)
    cell_px = (canvas - 2 * margin) * args.grid_resolution / (bounds[1] - bounds[0])

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas}" height="{canvas}" viewBox="0 0 {canvas} {canvas}">']
    parts.append('<rect width="100%" height="100%" fill="#f7f7f4"/>')

    table_left, table_top = world_to_svg(-table_x / 2, table_y / 2, bounds, canvas, margin)
    table_right, table_bottom = world_to_svg(table_x / 2, -table_y / 2, bounds, canvas, margin)
    svg_rect(parts, table_left, table_top, table_right - table_left, table_bottom - table_top, "#d8d2c2", "#343434")

    if args.cube_spawn_range is not None:
        spawn_x, spawn_y = args.cube_spawn_range
        sx0, sy0 = world_to_svg(-spawn_x, spawn_y, bounds, canvas, margin)
        sx1, sy1 = world_to_svg(spawn_x, -spawn_y, bounds, canvas, margin)
        svg_rect(
            parts,
            sx0,
            sy0,
            sx1 - sx0,
            sy1 - sy0,
            "#111827",
            "#111827",
            opacity=0.08,
            extra='stroke-dasharray="10 8" stroke-width="2"',
        )

    for i, points in enumerate(robot_points):
        color = COLORS[i % len(COLORS)]
        cells = occupied_cells(points, args.grid_resolution)
        for cx, cy in cells:
            x0 = cx * args.grid_resolution
            y1 = (cy + 1) * args.grid_resolution
            sx, sy = world_to_svg(x0, y1, bounds, canvas, margin)
            svg_rect(parts, sx, sy, cell_px, cell_px, color, opacity=0.22)

        if len(points) > 0:
            center = points[:, :2].mean(axis=0)
            sx, sy = world_to_svg(center[0], center[1], bounds, canvas, margin)
            parts.append(f'<text x="{fmt(sx)}" y="{fmt(sy)}" font-family="monospace" font-size="20" fill="{color}">robot{i}</text>')

    origin_x, origin_y = world_to_svg(0, 0, bounds, canvas, margin)
    parts.append(f'<line x1="{fmt(origin_x - 10)}" y1="{fmt(origin_y)}" x2="{fmt(origin_x + 10)}" y2="{fmt(origin_y)}" stroke="#111" stroke-width="2"/>')
    parts.append(f'<line x1="{fmt(origin_x)}" y1="{fmt(origin_y - 10)}" x2="{fmt(origin_x)}" y2="{fmt(origin_y + 10)}" stroke="#111" stroke-width="2"/>')

    title = f"FK reachability, table={fmt(table_x)} x {fmt(table_y)} m"
    z_band = f"EEF z in table_z + [{fmt(args.z_min)}, {fmt(args.z_max)}] m, samples/robot={args.samples_per_robot}"
    parts.append(f'<text x="40" y="42" font-family="monospace" font-size="24" fill="#111">{title}</text>')
    parts.append(f'<text x="40" y="74" font-family="monospace" font-size="20" fill="#333">{z_band}</text>')
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def downsample_points(points, max_points, rng):
    if len(points) <= max_points:
        return points
    indexes = rng.choice(len(points), size=max_points, replace=False)
    return points[indexes]


def write_3d_html(path, robot_geom_points, eef_points, camera_pos, nearest_camera_distance, args):
    rng = np.random.default_rng(args.seed + 1009)
    traces = []
    for i, points in enumerate(robot_geom_points):
        points = downsample_points(points, args.max_3d_points_per_robot, rng)
        traces.append(
            {
                "name": f"robot{i} collision geoms",
                "color": COLORS[i % len(COLORS)],
                "size": 2,
                "points": points.tolist(),
            }
        )
    for i, points in enumerate(eef_points):
        points = downsample_points(points, max(1, args.max_3d_points_per_robot // 5), rng)
        traces.append(
            {
                "name": f"robot{i} eef table-band",
                "color": "#111111",
                "size": 3,
                "points": points.tolist(),
            }
        )

    table_x, table_y, table_z_size = args.table_full_size
    table_top = 0.8
    table_bottom = table_top - table_z_size
    table_vertices = [
        [-table_x / 2, -table_y / 2, table_bottom],
        [table_x / 2, -table_y / 2, table_bottom],
        [table_x / 2, table_y / 2, table_bottom],
        [-table_x / 2, table_y / 2, table_bottom],
        [-table_x / 2, -table_y / 2, table_top],
        [table_x / 2, -table_y / 2, table_top],
        [table_x / 2, table_y / 2, table_top],
        [-table_x / 2, table_y / 2, table_top],
    ]
    camera = None if camera_pos is None else camera_pos.tolist()
    clearance = "unknown" if nearest_camera_distance is None else f"{nearest_camera_distance:.4f} m"
    title = html.escape(
        f"MultiArmBlockLift 3D sampled collision-geom sweep, camera={args.camera_name}, nearest sampled distance={clearance}"
    )

    traces_json = json.dumps(traces)
    table_vertices_json = json.dumps(table_vertices)
    camera_json = json.dumps(camera)

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f7f7f4; color: #111; }}
    #bar {{ padding: 10px 14px; border-bottom: 1px solid #ccc; }}
    #canvas {{ width: 100vw; height: calc(100vh - 64px); display: block; }}
  </style>
</head>
<body>
  <div id="bar">{title}<br>Drag to rotate, wheel to zoom. Points are sampled collision geom centers, not full swept volumes.</div>
  <canvas id="canvas"></canvas>
  <script>
    const traces = {traces_json};
    const tableVertices = {table_vertices_json};
    const cameraPoint = {camera_json};
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let yaw = -0.75, pitch = 0.65, zoom = 420, panX = 0, panY = 0;
    let dragging = false, lastX = 0, lastY = 0;

    function resize() {{
      canvas.width = canvas.clientWidth * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;
      draw();
    }}
    window.addEventListener("resize", resize);

    function project(p) {{
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * p[0] - sy * p[1];
      const y1 = sy * p[0] + cy * p[1];
      const z1 = p[2] - 0.8;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      const scale = zoom / (1 + 0.22 * z2);
      return [canvas.width / 2 + panX + x1 * scale, canvas.height / 2 + panY - y2 * scale, z2];
    }}

    function drawLine(a, b, color, width=1) {{
      const pa = project(a), pb = project(b);
      ctx.strokeStyle = color; ctx.lineWidth = width * devicePixelRatio;
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f7f7f4"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      for (const e of edges) drawLine(tableVertices[e[0]], tableVertices[e[1]], "#343434", 2);
      const all = [];
      for (const t of traces) for (const p of t.points) all.push([p, t.color, t.size]);
      if (cameraPoint) all.push([cameraPoint, "#d00000", 9]);
      all.sort((a,b) => project(a[0])[2] - project(b[0])[2]);
      for (const item of all) {{
        const p = project(item[0]);
        ctx.fillStyle = item[1];
        ctx.globalAlpha = item[1] === "#d00000" ? 1.0 : 0.42;
        ctx.beginPath();
        ctx.arc(p[0], p[1], item[2] * devicePixelRatio, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.globalAlpha = 1;
      if (cameraPoint) {{
        const p = project(cameraPoint);
        ctx.fillStyle = "#d00000";
        ctx.fillText("camera", p[0] + 10, p[1] - 10);
      }}
    }}
    canvas.addEventListener("mousedown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
    window.addEventListener("mouseup", () => dragging = false);
    window.addEventListener("mousemove", e => {{
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.006;
      pitch += (e.clientY - lastY) * 0.006;
      pitch = Math.max(-1.35, Math.min(1.35, pitch));
      lastX = e.clientX; lastY = e.clientY;
      draw();
    }});
    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      zoom *= Math.exp(-e.deltaY * 0.001);
      draw();
    }}, {{ passive: false }});
    resize();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-full-size", nargs=3, type=float, default=(0.9, 0.9, 0.05), metavar=("X", "Y", "Z"))
    parser.add_argument("--robots", nargs="+", default=("Panda", "Panda", "Panda", "Panda"))
    parser.add_argument("--arm-positions", nargs="+", default=("west", "south", "east", "north"))
    parser.add_argument("--position-radius-scale", type=float, default=1.0)
    parser.add_argument("--num-cubes", type=int, default=3)
    parser.add_argument("--cube-spawn-range", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--samples-per-robot", type=int, default=20000)
    parser.add_argument("--z-min", type=float, default=0.02, help="Minimum EEF height above table top in meters.")
    parser.add_argument("--z-max", type=float, default=0.35, help="Maximum EEF height above table top in meters.")
    parser.add_argument("--grid-resolution", type=float, default=0.025, help="Reach map cell size in meters.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("/tmp/multi_arm_fk_reach.svg"))
    parser.add_argument("--output-3d", type=Path, default=None)
    parser.add_argument("--camera-name", type=str, default="new_birdview")
    parser.add_argument("--max-3d-points-per-robot", type=int, default=25000)
    parser.add_argument("--canvas-size", type=int, default=1000)
    args = parser.parse_args()

    env_kwargs = {
        "robots": list(args.robots),
        "arm_positions": list(args.arm_positions),
        "position_radius_scale": args.position_radius_scale,
        "table_full_size": tuple(args.table_full_size),
        "num_cubes": args.num_cubes,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": False,
        "ignore_done": True,
        "initialization_noise": None,
    }
    if args.cube_spawn_range is not None:
        env_kwargs["cube_spawn_range"] = tuple(args.cube_spawn_range)

    env = robosuite.make("MultiArmBlockLift", **env_kwargs)
    try:
        env.reset()
        robot_points, robot_geom_points, camera_pos, nearest_camera_distance = sample_robot_reach(
            env=env,
            samples_per_robot=args.samples_per_robot,
            z_min=args.z_min,
            z_max=args.z_max,
            seed=args.seed,
            camera_name=args.camera_name,
        )
        write_svg(args.output, robot_points, args)
        if args.output_3d is not None:
            write_3d_html(args.output_3d, robot_geom_points, robot_points, camera_pos, nearest_camera_distance, args)
    finally:
        env.close()

    counts = ", ".join(f"robot{i}: {len(points)}" for i, points in enumerate(robot_points))
    print(f"Wrote {args.output}")
    if args.output_3d is not None:
        print(f"Wrote {args.output_3d}")
    print(f"Accepted samples in z band: {counts}")
    if nearest_camera_distance is not None:
        print(f"Nearest sampled robot collision-geom center to {args.camera_name}: {nearest_camera_distance:.4f} m")


if __name__ == "__main__":
    main()
