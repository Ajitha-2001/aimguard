"""Layer 2: information hiding. The server only replicates enemies a player could plausibly see,
so wallhack/radar cheats have little hidden data to display. 2D grid + potentially-visible-set.

KNOWN LEAKS (by design or limitation): (1) teammates are always replicated; (2) enemies within `peek_radius`
of a corner edge may be replicated; (3) audio cues reveal a coarse direction + loudness of hidden enemies;
(4) 2D walls only: no floors, doors, props or dynamic occluders. A cheat can still exploit what is replicated."""
import math
from dataclasses import dataclass


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d):
    return (_ccw(a, c, d) * _ccw(b, c, d) < 0) and (_ccw(a, b, c) * _ccw(a, b, d) < 0)


class World:
    def __init__(self, width, height, walls, cell=8.0, margin_cells=1):
        """walls: list of ((x1,y1),(x2,y2)) segments. margin_cells widens the visible set so
        enemies don't 'pop in' around corners (tune with playtests)."""
        self.cell, self.margin = cell, margin_cells
        self.nx, self.ny, self.walls = math.ceil(width / cell), math.ceil(height / cell), walls
        self.pvs = self._build()

    def cell_of(self, pos):
        return (min(max(int(pos[0] // self.cell), 0), self.nx - 1), min(max(int(pos[1] // self.cell), 0), self.ny - 1))

    def center(self, c):
        return ((c[0] + 0.5) * self.cell, (c[1] + 0.5) * self.cell)

    def los(self, a, b):
        a, b = (a[0], a[1]), (b[0], b[1])
        return not any(segments_intersect(a, b, w[0], w[1]) for w in self.walls)

    def _build(self):
        cells = [(x, y) for x in range(self.nx) for y in range(self.ny)]
        base = {c: {o for o in cells if o == c or self.los(self.center(c), self.center(o))} for c in cells}
        out = {}
        for c, vis in base.items():                          # widen by the margin ring
            wide = set(vis)
            for v in vis:
                for dx in range(-self.margin, self.margin + 1):
                    for dy in range(-self.margin, self.margin + 1):
                        n = (v[0] + dx, v[1] + dy)
                        if 0 <= n[0] < self.nx and 0 <= n[1] < self.ny:
                            wide.add(n)
            out[c] = wide
        return out


@dataclass
class Entity:
    id: str
    team: str
    pos: tuple


def _offsets(pos, r):
    x, y = pos[0], pos[1]
    return [(x, y), (x + r, y), (x - r, y), (x, y + r), (x, y - r)]


def could_see(world: World, a, b, peek_radius=1.5):
    """True if a line of sight exists between the two positions OR between points within `peek_radius` of each.
    The margin avoids enemies 'popping in' as someone peeks around a corner. It is a DELIBERATE, BOUNDED leak:
    an enemy hidden only by a corner edge, within `peek_radius` of being visible, may be replicated."""
    return any(world.los(p, q) for p in _offsets(a, peek_radius) for q in _offsets(b, peek_radius))


def entities_for_player(player: Entity, entities, world: World, peek_radius=1.5):
    """Replicate ONLY: teammates, and enemies inside the player's PVS that have (peek-margin) line of sight.
    There is NO distance exception: an enemy behind a wall is never replicated, however close.
    Hidden-but-near enemies are conveyed by audio_cues (direction only) instead."""
    visible = world.pvs[world.cell_of(player.pos)]
    return [e for e in entities if e.id != player.id and
            (e.team == player.team or (world.cell_of(e.pos) in visible and could_see(world, player.pos, e.pos, peek_radius)))]


def audio_cues(player: Entity, entities, world: World, radius=40.0, buckets=8):
    """Hidden-but-audible enemies become direction-only sound events, never positions."""
    shown = {e.id for e in entities_for_player(player, entities, world)}
    cues = []
    for e in entities:
        if e.id == player.id or e.id in shown or e.team == player.team:
            continue
        d = math.dist(player.pos[:2], e.pos[:2])
        if d <= radius:
            ang = math.atan2(e.pos[1] - player.pos[1], e.pos[0] - player.pos[0]) % (2 * math.pi)
            cues.append({"type": "footstep", "dir_bucket": int(ang / (2 * math.pi) * buckets) % buckets,
                         "loudness": round(1 - d / radius, 2)})
    return cues
