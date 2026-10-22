"""Layer 1: real-time server-side validation (O(1) per event) with a violation budget."""
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class MoveRules:
    max_speed: float = 7.5            # units/s, from game design
    tolerance: float = 1.15           # slack for lag/physics
    max_vertical_speed: float = 12.0


@dataclass
class WeaponCfg:
    min_interval_ms: float = 100.0
    max_range: float = 150.0


@dataclass
class State:
    t_ms: float
    pos: tuple
    ammo: int = 30
    last_shot_ms: float = -1e9
    in_vehicle: bool = False


def validate_move(prev: State, cur: State, rules: MoveRules = MoveRules()):
    dt = (cur.t_ms - prev.t_ms) / 1000
    if dt <= 0:
        return "bad_timestamp"
    dx, dy, dz = (cur.pos[i] - prev.pos[i] for i in range(3))
    if math.hypot(dx, dy) / dt > rules.max_speed * rules.tolerance and not cur.in_vehicle:
        return "speed_violation"
    if abs(dz) / dt > rules.max_vertical_speed and not cur.in_vehicle:
        return "fly_or_teleport"
    return None


def validate_fire(player: State, shot_t_ms: float, weapon: WeaponCfg):
    if player.ammo <= 0:
        return "no_ammo"
    if shot_t_ms - player.last_shot_ms < weapon.min_interval_ms * 0.9:
        return "fire_rate_violation"
    return None


class ViolationBudget:
    """Never act on a single violation (jitter causes false flags). `budget` violations within
    `window_ms` => 'rubber_band' (correct the player's position and log). Repeated rubber-banding
    (`escalate_after` times) is server-proven evidence a moderator can act on."""

    def __init__(self, budget=5, window_ms=10_000, escalate_after=3):
        self.budget, self.window_ms, self.escalate_after = budget, window_ms, escalate_after
        self._v = defaultdict(deque)
        self.rubber_bands = defaultdict(int)

    def record(self, player_id, t_ms, reason):
        q = self._v[player_id]
        q.append(t_ms)
        while q and t_ms - q[0] > self.window_ms:
            q.popleft()
        if len(q) >= self.budget:
            q.clear()
            self.rubber_bands[player_id] += 1
            return "rubber_band"
        return "ignore"

    def server_proven(self):
        return {p for p, n in self.rubber_bands.items() if n >= self.escalate_after}


class PositionHistory:
    """Ring buffer of a player's positions for lag compensation."""

    def __init__(self, keep_ms=1000):
        self.keep_ms, self.buf = keep_ms, deque()

    def add(self, t_ms, pos):
        self.buf.append((t_ms, pos))
        while self.buf and t_ms - self.buf[0][0] > self.keep_ms:
            self.buf.popleft()

    def at(self, t_ms):
        if not self.buf:
            return None
        prev = self.buf[0]
        for cur in self.buf:
            if cur[0] >= t_ms:
                if cur[0] == prev[0] or t_ms <= prev[0]:
                    return cur[1]
                f = (t_ms - prev[0]) / (cur[0] - prev[0])
                return tuple(prev[1][i] + f * (cur[1][i] - prev[1][i]) for i in range(3))
            prev = cur
        return self.buf[-1][1]


def validate_hit(now_ms, shooter_view_ms, origin, direction, target_hist: PositionHistory,
                 weapon: WeaponCfg, radius=0.6, max_rewind_ms=200, los=None):
    """Server re-runs the shot against the target's position at the shooter's (bounded) view time.
    Never trust a client's 'I hit him'. Returns ('hit', dist) or ('rejected', reason)."""
    view = max(shooter_view_ms, now_ms - max_rewind_ms)         # bound the rewind
    c = target_hist.at(view)
    if c is None:
        return "rejected", "no_target_history"
    n = math.sqrt(sum(d * d for d in direction))
    if n < 1e-9:
        return "rejected", "bad_direction"
    d = tuple(x / n for x in direction)
    oc = tuple(c[i] - origin[i] for i in range(3))
    proj = sum(oc[i] * d[i] for i in range(3))
    if proj < 0:
        return "rejected", "target_behind"
    dist2 = sum(x * x for x in oc) - proj * proj
    if dist2 > radius * radius:
        return "rejected", "miss"
    if proj > weapon.max_range:
        return "rejected", "out_of_range"
    if los is not None and not los(origin, c):
        return "rejected", "no_line_of_sight"
    return "hit", proj
