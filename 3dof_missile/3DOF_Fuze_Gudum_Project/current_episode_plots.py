from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from air_to_air_3dof_core import (
    NORTH, EAST, DOWN, SPEED, CHI, GAMMA, POSITION,
    SimulationResult, VehicleSpec, velocity_from_state,
)

def _set_axes_equal_3d(axis):
    x_limits = axis.get_xlim3d()
    y_limits = axis.get_ylim3d()
    z_limits = axis.get_zlim3d()
    ranges = np.array([np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)], dtype=float)
    centers = np.array([np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)], dtype=float)
    radius = 0.5 * max(float(ranges.max()), 1e-6)
    axis.set_xlim3d(centers[0]-radius, centers[0]+radius)
    axis.set_ylim3d(centers[1]-radius, centers[1]+radius)
    axis.set_zlim3d(centers[2]-radius, centers[2]+radius)

def _resolve_target_name(result, missile_name):
    if missile_name not in result.target_of:
        raise KeyError(f"{missile_name!r} için target_of kaydı bulunamadı.")
    return result.target_of[missile_name]

def _derive_relative_metrics(result, missile_name, target_name):
    missile = np.asarray(result.states[missile_name], dtype=float)
    target = np.asarray(result.states[target_name], dtype=float)
    n = min(len(result.time_s), len(missile), len(target))
    t = np.asarray(result.time_s[:n], dtype=float)
    missile = missile[:n]
    target = target[:n]
    r = target[:, POSITION] - missile[:, POSITION]
    vrel = np.vstack([
        velocity_from_state(target[i]) - velocity_from_state(missile[i])
        for i in range(n)
    ])
    rng = np.linalg.norm(r, axis=1)
    safe = np.maximum(rng, 1e-9)
    los_u = r / safe[:, None]
    closing = -np.sum(vrel * los_u, axis=1)
    los_rate = np.linalg.norm(np.cross(r, vrel) / safe[:, None]**2, axis=1)

    north, east, down = r[:,0], r[:,1], r[:,2]
    horiz = np.sqrt(north**2 + east**2)
    az = np.unwrap(np.arctan2(east, north))
    el = np.unwrap(np.arctan2(-down, horiz))
    if n >= 3:
        az_rate = np.gradient(az, t)
        el_rate = np.gradient(el, t)
    else:
        az_rate = np.zeros(n)
        el_rate = np.zeros(n)

    return {
        "time_s": t,
        "missile_state": missile,
        "target_state": target,
        "range_m": rng,
        "closing_speed_mps": closing,
        "los_rate_radps": los_rate,
        "los_azimuth_rate_deg_s": np.rad2deg(az_rate),
        "los_elevation_rate_deg_s": np.rad2deg(el_rate),
    }

def print_episode_summary(result: SimulationResult, *, missile_name="missile_1"):
    target_name = _resolve_target_name(result, missile_name)
    m = _derive_relative_metrics(result, missile_name, target_name)
    end = min(result.terminal_index.get(missile_name, len(m["time_s"])-1), len(m["time_s"])-1)
    rng = m["range_m"][:end+1]
    idx = int(np.argmin(rng))
    print("="*68)
    print("CURRENT EPISODE")
    print("="*68)
    print(f"Missile        : {missile_name}")
    print(f"Target         : {target_name}")
    print(f"Status         : {result.status.get(missile_name, 'unknown')}")
    print(f"Terminal time  : {result.terminal_time_s.get(missile_name, np.nan):.3f} s")
    print(f"Minimum range  : {float(rng[idx]):.4f} m")
    print(f"CPA time       : {m['time_s'][idx]:.3f} s")
    print(f"Hit radius     : {result.hit_radius_m:.3f} m")
    print(f"Initial range  : {rng[0]:.2f} m")
    print(f"Initial closing: {m['closing_speed_mps'][0]:.2f} m/s")
    print("="*68)

def plot_episode(
    result: SimulationResult,
    vehicles: Iterable[VehicleSpec],
    *,
    missile_name="missile_1",
    title=None,
    save_path=None,
    show=True,
):
    specs = {v.name: v for v in vehicles}
    if missile_name not in result.states:
        raise KeyError(f"{missile_name!r} result.states içinde yok.")
    target_name = _resolve_target_name(result, missile_name)
    if target_name not in result.states:
        raise KeyError(f"Hedef {target_name!r} result.states içinde yok.")

    m = _derive_relative_metrics(result, missile_name, target_name)
    t = m["time_s"]
    missile = m["missile_state"]
    target = m["target_state"]
    rng = m["range_m"]

    end = int(min(result.terminal_index.get(missile_name, len(t)-1), len(t)-1))
    sl = slice(0, end+1)
    tp = t[sl]
    mp = missile[sl]
    gp = target[sl]
    rp = rng[sl]
    cpa_i = int(np.argmin(rp))
    min_r = float(rp[cpa_i])
    status = result.status.get(missile_name, "unknown")

    if title is None:
        title = f"{missile_name} → {target_name} | {status} | min range = {min_r:.3f} m"

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14)

    # 1) 3D
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(mp[:,NORTH]/1000, mp[:,EAST]/1000, -mp[:,DOWN]/1000, linewidth=2, label=missile_name)
    ax.plot(gp[:,NORTH]/1000, gp[:,EAST]/1000, -gp[:,DOWN]/1000, "--", linewidth=2, label=target_name)
    ax.scatter(mp[0,NORTH]/1000, mp[0,EAST]/1000, -mp[0,DOWN]/1000, s=45)
    ax.scatter(gp[0,NORTH]/1000, gp[0,EAST]/1000, -gp[0,DOWN]/1000, s=45)
    ax.scatter(mp[cpa_i,NORTH]/1000, mp[cpa_i,EAST]/1000, -mp[cpa_i,DOWN]/1000, marker="X", s=100, label="Closest approach")
    ax.scatter(gp[cpa_i,NORTH]/1000, gp[cpa_i,EAST]/1000, -gp[cpa_i,DOWN]/1000, marker="X", s=100)
    ax.plot(
        [mp[cpa_i,NORTH]/1000, gp[cpa_i,NORTH]/1000],
        [mp[cpa_i,EAST]/1000, gp[cpa_i,EAST]/1000],
        [-mp[cpa_i,DOWN]/1000, -gp[cpa_i,DOWN]/1000],
        ":", linewidth=1.2,
    )
    ax.set_xlabel("North [km]"); ax.set_ylabel("East [km]"); ax.set_zlabel("Altitude [km]")
    ax.set_title("3D trajectory"); ax.legend(fontsize=8)
    _set_axes_equal_3d(ax)

    # 2) Range
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(tp, rp, linewidth=2)
    ax.axhline(result.hit_radius_m, linestyle="--", label=f"Hit radius ({result.hit_radius_m:g} m)")
    ax.scatter(tp[cpa_i], rp[cpa_i], marker="X", s=70, label=f"Minimum = {min_r:.3f} m")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Range [m]"); ax.set_title("Missile-target range")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8)

    # 3) LOS rate
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(tp, np.rad2deg(m["los_rate_radps"][sl]), linewidth=2, label="LOS-rate magnitude")
    ax.plot(tp, m["los_azimuth_rate_deg_s"][sl], linewidth=1.1, alpha=.75, label="Azimuth component")
    ax.plot(tp, m["los_elevation_rate_deg_s"][sl], linewidth=1.1, alpha=.75, label="Elevation component")
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("LOS rate [deg/s]"); ax.set_title("Line-of-sight rate")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8)

    # 4) Command
    ax = fig.add_subplot(2, 3, 4)
    cmd = np.asarray(result.command_g[missile_name], dtype=float)
    ncmd = min(len(cmd), end+1, len(result.time_s))
    ax.plot(result.time_s[:ncmd], cmd[:ncmd], linewidth=2)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Normal acceleration [g]"); ax.set_title("Guidance command magnitude")
    ax.grid(True, alpha=.3)

    # 5) Speed
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(tp, mp[:,SPEED], linewidth=2, label=missile_name)
    ax.plot(tp, gp[:,SPEED], "--", linewidth=2, label=target_name)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Speed [m/s]"); ax.set_title("Vehicle speeds")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8)

    # 6) chi + gamma
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(tp, np.rad2deg(np.unwrap(mp[:,CHI])), linewidth=1.8, label="Missile χ")
    ax.plot(tp, np.rad2deg(np.unwrap(gp[:,CHI])), "--", linewidth=1.8, label="Target χ")
    ax.plot(tp, np.rad2deg(mp[:,GAMMA]), linewidth=1.4, label="Missile γ")
    ax.plot(tp, np.rad2deg(gp[:,GAMMA]), "--", linewidth=1.4, label="Target γ")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Angle [deg]"); ax.set_title("Heading and flight-path angle")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8, ncol=2)

    fig.tight_layout(rect=[0,0,1,.95])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return fig
