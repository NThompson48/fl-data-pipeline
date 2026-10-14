"""
release_view.py
---------------
Generate static stills of a pitcher's release frame and commit-point frame
from the hitter's perspective, per pitch type, using TrackMan/Statcast-style
9P trajectory parameters.

Pipeline:
    1.  pull_pitcher_pitches  -> DataFrame from Supabase `pitches` table
    2.  primary_fastball      -> pick anchor pitch type (four_seam | sinker | cutter)
    3.  get_reference_pitch   -> filter to middle-middle, average 9P params + release cloud
    4.  trajectory / time_to_plate / project_to_hitter_view  -> geometry
    5.  render_release_frame / render_commit_frame  -> matplotlib stills

Coordinate convention (Statcast — verified empirically with diagnose_coords):
    +x : 1B side from catcher's POV  (so RHP release_x < 0, LHP release_x > 0)
    +y : toward the mound from home  (release y0 ~ 50-55 ft, plate front = 1.417)
    +z : up                          (release z0 ~ 5-7 ft, ground = 0)

Hitter's eye stands on the opposite-handed side of the plate:
    RHH: eye_x < 0 (3B side)
    LHH: eye_x > 0 (1B side)
Hitter's head faces +y. Hitter's right = +x = 1B side. So screen +u = +x.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, PathPatch
import matplotlib.path as mpath
from matplotlib.lines import Line2D


# =====================================================================
# CONFIG
# =====================================================================

ANCHOR_PITCHES = ('four_seam', 'sinker', 'cutter')

# Middle-middle box (feet from plate center). Tighter = purer reference,
# smaller sample. 0.5 ft horizontal = 12" box, z range = 12" vertical.
MM_X_BOUND = 0.5
MM_Z_LOW = 2.0
MM_Z_HIGH = 3.0

# Minimum sample sizes
MIN_SAMPLE_ANCHOR = 10      # for the middle-middle anchor pitch
MIN_SAMPLE_SECONDARY = 8    # for secondaries (using competitive-zone fallback)

# Competitive zone (wider) for secondaries when middle-middle is too thin
COMP_X_BOUND = 0.83         # roughly edge of zone
COMP_Z_LOW = 1.5
COMP_Z_HIGH = 3.6

# Physics constants
BALL_RADIUS_FT = 0.12       # baseball radius
PLATE_FRONT_Y = 1.417       # front of home plate, Statcast coords
RUBBER_Y = 60.5             # back of pitcher's rubber
COMMIT_TIME_BEFORE_PLATE = 0.175  # seconds (typical MLB; FrL slightly more)


# =====================================================================
# DB CONNECTION
# =====================================================================

def get_engine(env_filename: str = 'FLCommand.env'):
    """Load creds from env file and return a SQLAlchemy engine."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    # Try common locations for the env file
    for p in [Path.cwd() / env_filename,
              Path.cwd().parent / env_filename,
              Path(__file__).parent / env_filename if '__file__' in globals() else None]:
        if p is not None and p.exists():
            load_dotenv(p)
            break

    url = URL.create(
        drivername='postgresql+psycopg2',
        username=os.environ['SUPABASE_USER'],
        password=os.environ['SUPABASE_PASSWORD'],
        host=os.environ['SUPABASE_HOST'],
        port=int(os.environ.get('SUPABASE_PORT', 5432)),
        database=os.environ.get('SUPABASE_DATABASE', 'postgres'),
    )
    return create_engine(url, pool_pre_ping=True)


# Columns we need for the release-view tool.
TRAJECTORY_COLS = ['x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0', 'ax0', 'ay0', 'az0']
PULL_COLS = [
    'pitcher', 'pitcher_id', 'pitcher_throws',
    'pitch_type', 'plate_x', 'plate_z',
    *TRAJECTORY_COLS, 'pfxx', 'pfxz',
    'year',
]


def pull_pitcher_pitches(engine, pitcher_name: str,
                         years: tuple = (2025, 2026)) -> pd.DataFrame:
    """Pull all pitches for one pitcher (by display-name ILIKE) across given years."""
    from sqlalchemy import text
    cols = ', '.join(PULL_COLS)
    sql = text(f"""
        SELECT {cols}
        FROM pitches
        WHERE pitcher ILIKE :name
          AND year = ANY(:years)
          AND x0 IS NOT NULL AND vy0 IS NOT NULL AND ay0 IS NOT NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            sql, conn,
            params={'name': f'%{pitcher_name}%', 'years': list(years)},
        )
    return df


# =====================================================================
# AGGREGATION
# =====================================================================

def primary_fastball(df: pd.DataFrame, min_n: int = 30) -> Optional[str]:
    """
    Pick this pitcher's primary fastball variant from {four_seam, sinker, cutter}.
    Returns the most-thrown one that meets min_n; falls back to most-thrown overall.
    """
    counts = df['pitch_type'].value_counts()
    candidates = [(pt, counts.get(pt, 0)) for pt in ANCHOR_PITCHES if counts.get(pt, 0) >= min_n]
    if candidates:
        return max(candidates, key=lambda t: t[1])[0]
    # Fallback: most-thrown pitch overall
    return counts.idxmax() if len(counts) > 0 else None


def get_reference_pitch(df: pd.DataFrame, pitch_type: str,
                        is_anchor: bool,
                        recency_weight: bool = True) -> Optional[dict]:
    """
    Filter to middle-middle (anchor) or competitive-zone (secondary), average
    the 9P trajectory parameters, and return the reference + the release-point
    scatter for cloud overlay.

    Returns dict with keys: params, release_cloud, n, location_filter
            or None if sample insufficient.
    """
    sub = df[df['pitch_type'] == pitch_type].copy()

    if is_anchor:
        loc_mask = (
            (sub['plate_x'].abs() < MM_X_BOUND)
            & sub['plate_z'].between(MM_Z_LOW, MM_Z_HIGH)
        )
        location_filter = 'middle-middle'
        min_n = MIN_SAMPLE_ANCHOR
    else:
        loc_mask = (
            (sub['plate_x'].abs() < COMP_X_BOUND)
            & sub['plate_z'].between(COMP_Z_LOW, COMP_Z_HIGH)
        )
        location_filter = 'competitive zone'
        min_n = MIN_SAMPLE_SECONDARY

    sub = sub[loc_mask]
    if len(sub) < min_n:
        return None

    # Weight 2026 pitches 3x relative to 2025 (when both present)
    if recency_weight and 'year' in sub.columns:
        weights = np.where(sub['year'] == sub['year'].max(), 3.0, 1.0)
    else:
        weights = np.ones(len(sub))

    params = {}
    for col in TRAJECTORY_COLS:
        params[col] = np.average(sub[col].values, weights=weights)

    return {
        'params': params,
        'release_cloud': sub[['x0', 'z0']].values,
        'n': len(sub),
        'location_filter': location_filter,
    }


# =====================================================================
# GEOMETRY: trajectory & camera
# =====================================================================

def trajectory(params: dict, t: float | np.ndarray) -> np.ndarray:
    """3D ball position at time t from release (seconds). Returns (3,) or (3, N)."""
    x = params['x0'] + params['vx0'] * t + 0.5 * params['ax0'] * t ** 2
    y = params['y0'] + params['vy0'] * t + 0.5 * params['ay0'] * t ** 2
    z = params['z0'] + params['vz0'] * t + 0.5 * params['az0'] * t ** 2
    return np.array([x, y, z])


def time_to_plate(params: dict, plate_y: float = PLATE_FRONT_Y) -> float:
    """Solve 0.5*ay0*t^2 + vy0*t + (y0 - plate_y) = 0 for the smaller positive root."""
    a = 0.5 * params['ay0']
    b = params['vy0']
    c = params['y0'] - plate_y
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        raise ValueError("No real solution to plate-arrival time — check trajectory params.")
    sq = np.sqrt(disc)
    # vy0 is negative (Statcast); want the smaller positive root
    candidates = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
    pos = [t for t in candidates if t > 0]
    return min(pos)


def eye_position(height_in: float, hitter_handedness: str) -> np.ndarray:
    """
    Hitter's eye in Statcast coords.

    +x = 1B side, so RHH stands on -x side of plate (3B side), LHH on +x side.
    Eye is ~7" below top of head (4" eye-to-crown + ~3" stance compression).
    Eye y ~ 0 (behind back tip of plate by stance depth).
    """
    height_ft = height_in / 12.0
    eye_z = height_ft - 7.0 / 12.0
    eye_x = -1.7 if hitter_handedness.upper().startswith('R') else 1.7
    eye_y = -1.5   # ~18 inches behind back tip of plate; tunable
    return np.array([eye_x, eye_y, eye_z])


def focal_length_from_fov(image_width_px: int, horizontal_fov_deg: float) -> float:
    """Pinhole focal length such that the visible width subtends `horizontal_fov_deg`."""
    return (image_width_px / 2.0) / np.tan(np.radians(horizontal_fov_deg / 2.0))


def project_to_hitter_view(point_xyz: np.ndarray, eye_xyz: np.ndarray,
                           focal_length: float) -> Optional[tuple]:
    """
    Pinhole projection from Statcast world coords to hitter's screen.

    Camera looks straight forward along +y (no tilt). Hitter's right = +x.
    For the release-view use case the optical axis is essentially horizontal —
    the release point is barely above eye height, so tilt is < 1°. Keeping the
    camera fixed gives a consistent frame across pitch types.

    Returns (u, v, apparent_radius_px) or None if the point is behind hitter.
    """
    rel = point_xyz - eye_xyz
    if rel[1] <= 0.1:
        return None
    u = focal_length * rel[0] / rel[1]
    v = focal_length * rel[2] / rel[1]
    apparent_radius = focal_length * BALL_RADIUS_FT / rel[1]
    return u, v, apparent_radius


# =====================================================================
# PITCHER SILHOUETTE (arm-angle-driven)
# =====================================================================

def estimate_shoulder(release_xyz: np.ndarray, arm_angle_deg: float,
                      pitcher_height_ft: float, throwing_hand: str) -> tuple:
    """
    Estimate throwing-shoulder position from release point + arm angle.

    Arm angle convention: 90° = directly overhead, 0° = sidearm/horizontal.
    Arm length ~ 0.35 * height (anthropometric rough rule).

    Returns (shoulder_xyz, arm_length_ft).
    """
    arm_length = 0.35 * pitcher_height_ft
    theta = np.radians(arm_angle_deg)
    # Throwing-arm-side direction in Statcast coords:
    # RHP releases from -x (3B side; pitcher faces -y so right hand points -x).
    # LHP releases from +x.
    side = -1 if throwing_hand.upper().startswith('R') else 1
    arm_vec = np.array([
        side * arm_length * np.cos(theta),
        0.0,
        arm_length * np.sin(theta),
    ])
    shoulder = release_xyz - arm_vec
    return shoulder, arm_length


def silhouette_keypoints(release_xyz: np.ndarray, arm_angle_deg: float,
                         pitcher_height_ft: float, throwing_hand: str) -> dict:
    """
    Build a pitcher silhouette in release / early-follow-through pose.

    Key biomechanical features modeled:
    - Lateral spine tilt toward glove side (throwing shoulder is HIGHER).
      Tilt magnitude scales with arm slot — overhand tilts most, sidearm least.
    - Forward trunk lean toward plate.
    - Lead leg planted forward and slightly to glove side, knee bent.
    - Trail leg pushing off / elevated behind hip in follow-through.
    - Glove tucked across body near opposite-hip / chest.

    Returns keypoints (3D world coords), segment list, and head_radius_ft.
    """
    side = 1 if throwing_hand.upper().startswith('L') else -1
    h = pitcher_height_ft

    # Anthropometric proportions (rough)
    arm_length = 0.35 * h
    shoulder_width = 0.22 * h
    torso_len = 0.30 * h
    hip_width = 0.16 * h
    head_radius = 0.06 * h

    # Lean angles vary with arm slot
    if arm_angle_deg > 65:
        lateral_tilt_deg = 24
    elif arm_angle_deg > 40:
        lateral_tilt_deg = 15
    else:
        lateral_tilt_deg = 6
    forward_tilt_deg = 18

    lat = np.radians(lateral_tilt_deg)
    fwd = np.radians(forward_tilt_deg)
    theta = np.radians(arm_angle_deg)

    # Throwing shoulder: arm extends from shoulder to release
    arm_vec = np.array([side * arm_length * np.cos(theta), 0.0,
                        arm_length * np.sin(theta)])
    throw_sh = release_xyz - arm_vec

    # Glove shoulder: across body, lower (lateral tilt rotates the shoulder line)
    glove_sh = throw_sh + np.array([
        -side * shoulder_width * np.cos(lat),
        0.0,
        -shoulder_width * np.sin(lat),
    ])

    # Spine top = midpoint of shoulders
    spine_top = (throw_sh + glove_sh) / 2.0

    # Spine vector (top -> hips): laterally tilted toward THROW side at the
    # bottom (because spine tilts top toward glove side -> bottom goes opposite)
    # and forward toward plate (-y).
    spine_vec = np.array([
        side * torso_len * np.sin(lat),
        -torso_len * np.sin(fwd),
        -torso_len * np.cos(lat) * np.cos(fwd),
    ])
    hip_center = spine_top + spine_vec

    # Hips
    throw_hip = hip_center + np.array([side * hip_width / 2.0, 0.0, 0.0])
    glove_hip = hip_center + np.array([-side * hip_width / 2.0, 0.0, 0.0])

    # Head: above spine_top, slightly toward glove side (head leans with spine)
    head_center = spine_top + np.array([
        -side * 0.20 * h * np.sin(lat),
        -0.04 * h,
        0.16 * h,
    ])

    # Lead leg (on glove side, lands forward).
    # Foot lands ~0.4 ft above true ground (still on mound slope is OK).
    foot_z = 0.3
    stride = 0.50 * h
    lead_foot = glove_hip + np.array([
        -side * 0.08 * h,
        -stride,
        foot_z - hip_center[2],
    ])
    # Knee a bit closer to hip than midway, knee forward of leg axis (bent)
    lead_knee = glove_hip + np.array([
        -side * 0.04 * h,
        -stride * 0.45,
        (foot_z - hip_center[2]) * 0.45,
    ])

    # Trail leg (throw side): pushing off / elevated behind hip
    trail_foot = throw_hip + np.array([
        side * 0.15 * h,
        0.30 * h,           # behind hip toward mound
        0.18 * h,           # elevated above hip
    ])
    trail_knee = throw_hip + np.array([
        side * 0.10 * h,
        0.12 * h,
        0.05 * h,
    ])

    # Glove arm: tucked across body, glove near throw-side hip / chest
    glove_hand = glove_sh + np.array([
        side * 0.06 * h,    # slightly across body toward throw side
        -0.05 * h,           # forward of shoulder
        -0.18 * h,           # below shoulder (down at chest)
    ])
    glove_elbow = (glove_sh + glove_hand) / 2.0 + np.array([
        -side * 0.02 * h,
        -0.04 * h,
        0.0,
    ])

    keypoints = {
        'release': release_xyz,
        'shoulder': throw_sh,
        'opp_shoulder': glove_sh,
        'head': head_center,
        'hip_throw': throw_hip,
        'hip_opp': glove_hip,
        'lead_knee': lead_knee,
        'lead_foot': lead_foot,
        'trail_knee': trail_knee,
        'trail_foot': trail_foot,
        'glove_elbow': glove_elbow,
        'glove_hand': glove_hand,
    }
    segments = [
        ('shoulder', 'release'),         # throwing arm at full extension
        ('shoulder', 'opp_shoulder'),    # shoulder line (visibly tilted)
        ('shoulder', 'hip_throw'),       # torso, throw side
        ('opp_shoulder', 'hip_opp'),     # torso, glove side
        ('hip_throw', 'hip_opp'),        # hips
        ('hip_opp', 'lead_knee'),        # lead thigh
        ('lead_knee', 'lead_foot'),      # lead shin
        ('hip_throw', 'trail_knee'),     # trail thigh
        ('trail_knee', 'trail_foot'),    # trail shin
        ('opp_shoulder', 'glove_elbow'), # glove upper arm
        ('glove_elbow', 'glove_hand'),   # glove forearm
    ]
    return {
        'keypoints': keypoints,
        'segments': segments,
        'head_radius_ft': head_radius,
    }


# =====================================================================
# RENDERING
# =====================================================================

DEFAULTS = {
    'image_size': 900,
    'fov_deg': 16.0,      # narrow zoom focus to clear vision training view
    'ball_scale': 2.5,    # prominent visibility scaling
    'bg_color': '#111612',
    'field_color': '#162b1d',
    'ball_face': '#ffffff',
    'ball_edge': '#a81b1b',
    'seam_color': '#cc2424',
    'cloud_face': '#ffffff',
    'cloud_alpha': 0.12,  # softened to blend smoothly when stacked
    'silhouette_color': '#d1ded6',
    'jersey_color': '#1d3324',
    'pants_color': '#e6e6e6',
    'glove_color': '#4a3319',
    'silhouette_alpha': 0.70,
    'silhouette_lw': 3.5,
    'plate_color': '#3a4f3f',
}


def _project_point(p: np.ndarray, eye: np.ndarray, focal: float,
                   image_size: int) -> Optional[tuple]:
    proj = project_to_hitter_view(p, eye, focal)
    if proj is None:
        return None
    u, v, r = proj
    return image_size / 2 + u, image_size / 2 + v, r


def _draw_home_plate(ax, eye: np.ndarray, focal: float, image_size: int, style: dict):
    """Projects home plate from 3D world space onto the hitter's screen perspective."""
    plate_pts = [
        np.array([0.0, 0.0, 0.0]),          # Back tip
        np.array([-0.708, 0.708, 0.0]),     # Left corner
        np.array([-0.708, 1.417, 0.0]),     # Left front
        np.array([0.708, 1.417, 0.0]),      # Right front
        np.array([0.708, 0.708, 0.0]),      # Right corner
    ]
    proj_pts = []
    for pt in plate_pts:
        proj = _project_point(pt, eye, focal, image_size)
        if proj is not None:
            proj_pts.append([proj[0], proj[1]])
    if len(proj_pts) == 5:
        ax.add_patch(Polygon(proj_pts, facecolor='none', edgecolor=style['plate_color'], lw=1.5, alpha=0.7, zorder=1))


def _draw_stadium_background(ax, eye: np.ndarray, focal: float, image_size: int, style: dict):
    """
    Minimal background: a single faint horizon line at eye-height.

    At the hitter's POV, anything in world coords at z = eye_z projects to
    image vertical center. Drawing it gives an unobtrusive sense of "ground
    level out there." The previously-rendered mound / outfield wall / batter's-
    eye created perspective distortion at narrow FOV and got dropped.
    """
    horizon_y = image_size / 2  # z = eye_z projects to v = 0 -> screen center
    ax.add_line(Line2D(
        [0, image_size], [horizon_y, horizon_y],
        color='#1f2c22', lw=1, alpha=0.5, zorder=0,
    ))


def _draw_baseball(ax, center_x: float, center_y: float, radius: float, style: dict):
    """Draws a crisp baseball circle with stylized red seam lines."""
    ax.add_patch(Circle((center_x, center_y), radius, facecolor=style['ball_face'], edgecolor=style['ball_edge'], lw=1.5, zorder=10))
    seam_offset = radius * 0.45

    left_path = mpath.Path([
        (center_x - seam_offset, center_y - radius * 0.8),
        (center_x - radius * 0.1, center_y),
        (center_x - seam_offset, center_y + radius * 0.8)
    ], [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])

    right_path = mpath.Path([
        (center_x + seam_offset, center_y - radius * 0.8),
        (center_x + radius * 0.1, center_y),
        (center_x + seam_offset, center_y + radius * 0.8)
    ], [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])

    for path in [left_path, right_path]:
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor=style['seam_color'], lw=1.2, ls=':', zorder=11))


def _draw_silhouette(ax, silhouette: dict, eye: np.ndarray, focal: float,
                     image_size: int, style: dict):
    """Draw the pitcher silhouette projected into the hitter's view."""
    kp = silhouette['keypoints']
    head_r_ft = silhouette['head_radius_ft']
    is_lhp = kp['release'][0] > 0

    # Project all keypoints once
    proj = {k: _project_point(v, eye, focal, image_size) for k, v in kp.items()}

    # 1. Filled jersey trapezoid (shoulders -> hips). With lateral spine tilt,
    # this comes out as a tilted quadrilateral — gives the body its lean.
    if all(proj[k] is not None for k in ['shoulder', 'opp_shoulder', 'hip_opp', 'hip_throw']):
        poly = [
            [proj['shoulder'][0],     proj['shoulder'][1]],
            [proj['opp_shoulder'][0], proj['opp_shoulder'][1]],
            [proj['hip_opp'][0],      proj['hip_opp'][1]],
            [proj['hip_throw'][0],    proj['hip_throw'][1]],
        ]
        ax.add_patch(Polygon(
            poly, facecolor=style['jersey_color'], edgecolor='none',
            alpha=style['silhouette_alpha'], zorder=2,
        ))

    # 2. Limbs. Different limb types get different styling.
    LEG_SEGS = {('hip_opp', 'lead_knee'), ('lead_knee', 'lead_foot'),
                ('hip_throw', 'trail_knee'), ('trail_knee', 'trail_foot')}
    THROW_ARM_SEGS = {('shoulder', 'release')}
    GLOVE_ARM_SEGS = {('opp_shoulder', 'glove_elbow'), ('glove_elbow', 'glove_hand')}

    for (a, b) in silhouette['segments']:
        pa, pb = proj[a], proj[b]
        if pa is None or pb is None:
            continue

        if (a, b) in LEG_SEGS:
            color = style['pants_color']
            lw = style['silhouette_lw'] * 1.1
        elif (a, b) in THROW_ARM_SEGS:
            color = style['silhouette_color']
            lw = style['silhouette_lw'] * 1.15
        elif (a, b) in GLOVE_ARM_SEGS:
            color = style['silhouette_color']
            lw = style['silhouette_lw'] * 0.95
        else:
            # Body outline (shoulder-to-hip rails, shoulder bar, hip bar) —
            # already filled by the jersey poly, so render thin
            color = style['jersey_color']
            lw = style['silhouette_lw'] * 0.6

        ax.add_line(Line2D(
            [pa[0], pb[0]], [pa[1], pb[1]],
            color=color, alpha=style['silhouette_alpha'],
            lw=lw, solid_capstyle='round', zorder=3,
        ))

    # 3. Head (circle scaled by distance) — use silhouette_color so it reads
    # clearly against the dark field background (jersey/field blend otherwise).
    head_apparent = None
    if proj['head'] is not None:
        head_apparent = focal * head_r_ft / max(0.1, (kp['head'] - eye)[1])
        ax.add_patch(Circle(
            (proj['head'][0], proj['head'][1]), head_apparent * 1.1,
            facecolor=style['silhouette_color'], edgecolor='none',
            alpha=style['silhouette_alpha'], zorder=4,
        ))

    # 4. Glove (small dark circle at glove_hand)
    if proj['glove_hand'] is not None and head_apparent is not None:
        ax.add_patch(Circle(
            (proj['glove_hand'][0], proj['glove_hand'][1]),
            head_apparent * 0.85,
            facecolor=style['glove_color'], edgecolor='none',
            alpha=style['silhouette_alpha'], zorder=4,
        ))


def _render_blank_frame(image_size: int, style: dict):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=style['bg_color'])
    ax.set_facecolor(style['field_color'])
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def render_release_frame(ref: dict, *,
                         hitter_height_in: float,
                         hitter_hand: str,
                         pitcher_height_ft: float,
                         arm_angle_deg: float,
                         throwing_hand: str,
                         pitcher_name: str = '',
                         pitch_type_label: str = '',
                         style: Optional[dict] = None):
    """
    Render the release frame: pitcher silhouette + release-point cloud + ball at release.
    """
    style = {**DEFAULTS, **(style or {})}
    image_size = style['image_size']
    focal = focal_length_from_fov(image_size, style['fov_deg'])
    eye = eye_position(hitter_height_in, hitter_hand)

    params = ref['params']
    release_xyz = np.array([params['x0'], params['y0'], params['z0']])

    fig, ax = _render_blank_frame(image_size, style)

    # Ground and Wall overlays
    _draw_stadium_background(ax, eye, focal, image_size, style)
    _draw_home_plate(ax, eye, focal, image_size, style)

    # Pitcher silhouette structure
    sil = silhouette_keypoints(release_xyz, arm_angle_deg, pitcher_height_ft, throwing_hand)
    _draw_silhouette(ax, sil, eye, focal, image_size, style)

    # Release-point cloud
    mean_y = params['y0']
    for xi, zi in ref['release_cloud']:
        proj = project_to_hitter_view(np.array([xi, mean_y, zi]), eye, focal)
        if proj is None:
            continue
        u, v, r = proj
        ax.add_patch(Circle(
            (image_size / 2 + u, image_size / 2 + v),
            r * style['ball_scale'],
            facecolor=style['cloud_face'],
            edgecolor='none',
            alpha=style['cloud_alpha'],
            zorder=5
        ))

    # Average ball at release
    proj = project_to_hitter_view(release_xyz, eye, focal)
    if proj is not None:
        u, v, r = proj
        _draw_baseball(ax, image_size / 2 + u, image_size / 2 + v, r * style['ball_scale'], style)

    title = f"{pitcher_name}  |  {pitch_type_label}  |  release"
    sub = f"vs {hitter_hand.upper()}HH  |  n={ref['n']}  ({ref['location_filter']})"
    ax.text(image_size / 2, image_size - 30, title,
            color='white', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(image_size / 2, image_size - 55, sub,
            color='#cfd6cf', ha='center', va='top', fontsize=10)

    return fig


def render_commit_frame(ref: dict, *,
                        hitter_height_in: float,
                        hitter_hand: str,
                        pitcher_name: str = '',
                        pitch_type_label: str = '',
                        style: Optional[dict] = None):
    """
    Render the commit-point frame: ball at t = (time_to_plate - 0.175s).
    """
    style = {**DEFAULTS, **(style or {})}
    image_size = style['image_size']
    focal = focal_length_from_fov(image_size, style['fov_deg'])
    eye = eye_position(hitter_height_in, hitter_hand)

    params = ref['params']
    t_plate = time_to_plate(params)
    t_commit = max(0.0, t_plate - COMMIT_TIME_BEFORE_PLATE)
    ball_xyz = trajectory(params, t_commit)

    fig, ax = _render_blank_frame(image_size, style)

    # Ballpark ground continuity
    _draw_stadium_background(ax, eye, focal, image_size, style)
    _draw_home_plate(ax, eye, focal, image_size, style)

    # Faint horizon line at strike-zone bottom (z=1.5) for reference
    horizon = trajectory(params, t_commit).copy()
    horizon[2] = 0.0
    proj_h = project_to_hitter_view(horizon, eye, focal)
    if proj_h is not None:
        _, vh, _ = proj_h
        ax.add_line(Line2D([0, image_size], [image_size / 2 + vh, image_size / 2 + vh],
                           color='#2c4a2c', lw=1, alpha=0.6, zorder=1))

    proj = project_to_hitter_view(ball_xyz, eye, focal)
    if proj is not None:
        u, v, r = proj
        _draw_baseball(ax, image_size / 2 + u, image_size / 2 + v, r * style['ball_scale'], style)

    title = f"{pitcher_name}  |  {pitch_type_label}  |  commit point"
    distance_ft = (ball_xyz - eye)[1]
    sub = (f"vs {hitter_hand.upper()}HH  |  {COMMIT_TIME_BEFORE_PLATE*1000:.0f} ms "
           f"before plate  |  {distance_ft:.1f} ft from hitter")
    ax.text(image_size / 2, image_size - 30, title,
            color='white', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(image_size / 2, image_size - 55, sub,
            color='#cfd6cf', ha='center', va='top', fontsize=10)

    return fig


# =====================================================================
# DIAGNOSTICS
# =====================================================================

def diagnose_coords(df: pd.DataFrame) -> None:
    """
    Print sign and magnitude of release_pos_x relative to throwing hand
    to verify the coordinate convention matches assumptions.
    """
    if 'pitcher_throws' not in df.columns:
        print("[diagnose] no pitcher_throws column; skipping")
        return
    for hand, sub in df.groupby('pitcher_throws'):
        if len(sub) == 0:
            continue
        mx, mz, my = sub['x0'].mean(), sub['z0'].mean(), sub['y0'].mean()
        print(f"[diagnose] {hand}: n={len(sub)}, mean x0={mx:+.2f}, "
              f"y0={my:+.2f}, z0={mz:+.2f}")
    print("  Statcast expects: R hand -> x0 negative (3B side), "
          "L hand -> x0 positive (1B side).")


# =====================================================================
# SYNTHETIC DATA  (for testing without DB)
# =====================================================================

def make_synthetic_pitcher(pitcher_name: str = 'Figueredo, Kevin',
                           pitcher_throws: str = 'L',
                           n_pitches: int = 250,
                           seed: int = 42) -> pd.DataFrame:
    """
    Generate a plausible Frontier-League-ish LHP pitch sample for end-to-end testing.
    """
    rng = np.random.default_rng(seed)
    hand_sign = 1 if pitcher_throws.upper().startswith('L') else -1

    rel_x_mu = hand_sign * 2.0
    rel_z_mu = 6.0
    rel_y_mu = 54.0

    pitch_mix = {
        'four_seam': 0.45,
        'sinker':    0.10,
        'slider':    0.25,
        'changeup':  0.15,
        'curveball': 0.05,
    }
    pitches = []
    for _ in range(n_pitches):
        pt = rng.choice(list(pitch_mix.keys()), p=list(pitch_mix.values()))
        velo_mu = {'four_seam': 92, 'sinker': 90, 'slider': 83,
                   'changeup': 82, 'curveball': 76}[pt]
        velo = rng.normal(velo_mu, 1.2)
        velo_fps = velo * 1.467

        x0 = rng.normal(rel_x_mu, 0.25)
        z0 = rng.normal(rel_z_mu, 0.15)
        y0 = rng.normal(rel_y_mu, 0.3)

        spread = {'four_seam': (0.6, 0.7), 'sinker': (0.7, 0.7),
                  'slider': (0.8, 0.9), 'changeup': (0.8, 0.9),
                  'curveball': (0.8, 1.1)}[pt]
        aim_x = rng.normal(0, spread[0])
        aim_z = rng.normal(2.5, spread[1])

        flight_t = (y0 - PLATE_FRONT_Y) / velo_fps
        vy0 = -(y0 - PLATE_FRONT_Y) / flight_t
        vx0 = (aim_x - x0) / flight_t
        vz0 = (aim_z - z0) / flight_t + 0.5 * 32.17 * flight_t

        pfx_z_mu, pfx_x_mu = {
            'four_seam': (15, hand_sign * 8),
            'sinker':    (8, hand_sign * 14),
            'slider':    (1, -hand_sign * 4),
            'changeup':  (8, hand_sign * 12),
            'curveball': (-8, -hand_sign * 6),
        }[pt]
        pfxz = rng.normal(pfx_z_mu, 1.5) / 12.0
        pfxx = rng.normal(pfx_x_mu, 1.5) / 12.0

        ax0 = 2 * pfxx / flight_t ** 2
        az0_extra = 2 * pfxz / flight_t ** 2
        az0 = -32.17 + az0_extra
        ay0 = rng.normal(28, 2)

        plate_x = aim_x + rng.normal(0, 0.15)
        plate_z = aim_z + rng.normal(0, 0.20)

        pitches.append({
            'pitcher': pitcher_name,
            'pitcher_id': 12345,
            'pitcher_throws': pitcher_throws,
            'pitch_type': pt,
            'plate_x': plate_x,
            'plate_z': plate_z,
            'x0': x0, 'y0': y0, 'z0': z0,
            'vx0': vx0, 'vy0': vy0, 'vz0': vz0,
            'ax0': ax0, 'ay0': ay0, 'az0': az0,
            'pfxx': pfxx, 'pfxz': pfxz,
            'year': rng.choice([2025, 2026], p=[0.6, 0.4]),
        })
    return pd.DataFrame(pitches)