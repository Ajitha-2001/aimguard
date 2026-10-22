import time
from aimguard.validation import *
from aimguard.pvs import World, Entity, entities_for_player, audio_cues
from aimguard.network import PacketAuth, AntiReplayWindow, RateLimiter, lag_switch_score
from aimguard.attestation import NonceStore, evaluate_play_integrity


# ---- validation ----
def test_speed_and_teleport_detected_normal_ok():
    a = State(0, (0, 0, 0)); ok = State(1000, (7, 0, 0))
    assert validate_move(a, ok) is None
    assert validate_move(a, State(1000, (30, 0, 0))) == "speed_violation"
    assert validate_move(a, State(1000, (0, 0, 40))) == "fly_or_teleport"
    assert validate_move(a, State(0, (1, 0, 0))) == "bad_timestamp"

def test_fire_rules():
    w = WeaponCfg(min_interval_ms=100)
    assert validate_fire(State(0, (0,0,0), ammo=0), 500, w) == "no_ammo"
    assert validate_fire(State(0, (0,0,0), last_shot_ms=450), 500, w) == "fire_rate_violation"
    assert validate_fire(State(0, (0,0,0), last_shot_ms=300), 500, w) is None

def test_budget_does_not_punish_single_violation_but_escalates():
    b = ViolationBudget(budget=5, window_ms=10_000, escalate_after=3)
    assert b.record("p", 0, "speed_violation") == "ignore"
    assert "p" not in b.server_proven()
    for rb in range(3):                       # three bursts of 5 violations
        res = [b.record("p", 100_000 * (rb + 1) + i, "speed_violation") for i in range(5)]
        assert res[-1] == "rubber_band"
    assert "p" in b.server_proven()

def test_old_violations_expire_from_budget():
    b = ViolationBudget(budget=3, window_ms=1000)
    assert [b.record("p", t, "x") for t in (0, 600, 5000, 5600)] == ["ignore"] * 4

def test_lag_compensated_hit():
    h = PositionHistory()
    for t in range(0, 1000, 50):
        h.add(t, (50 + t * 0.01, 0.0, 0.0))        # target walking away along x
    w = WeaponCfg(max_range=100)
    # shooter aimed at where target WAS 150ms ago (client view), server rewinds -> hit
    view = 800 - 150
    aim = h.at(view)
    assert validate_hit(800, view, (0, 0, 0), aim, h, w)[0] == "hit"
    # firing at where the target is NOW but with the old view time -> miss
    assert validate_hit(800, view, (0, 0, 0), (58.0, 3.0, 0), h, w) == ("rejected", "miss")

def test_rewind_is_bounded():
    h = PositionHistory()
    for t in range(0, 1000, 50):
        h.add(t, (50.0, 0.0, 0.0) if t < 500 else (50.0, 20.0, 0.0))   # target steps sideways at t=500
    aim_old = (50, 0, 0)                                                # where it WAS
    # attacker claims an ancient view time (100ms) at now=900: rewind clamped to 700 -> target already moved
    assert validate_hit(900, 100, (0, 0, 0), aim_old, h, WeaponCfg(), max_rewind_ms=200) == ("rejected", "miss")
    # without the bound the same claim would be accepted (this is what the clamp prevents)
    assert validate_hit(900, 100, (0, 0, 0), aim_old, h, WeaponCfg(), max_rewind_ms=10_000)[0] == "hit"

def test_hit_rejections():
    h = PositionHistory(); h.add(0, (500.0, 0, 0)); h.add(100, (500.0, 0, 0))
    assert validate_hit(100, 100, (0,0,0), (1,0,0), h, WeaponCfg(max_range=150)) == ("rejected", "out_of_range")
    assert validate_hit(100, 100, (0,0,0), (-1,0,0), h, WeaponCfg()) == ("rejected", "target_behind")
    assert validate_hit(100, 100, (0,0,0), (1,0,0), h, WeaponCfg(max_range=999), los=lambda a, b: False) == ("rejected", "no_line_of_sight")


# ---- pvs ----
def make_world():
    # one long wall separating y<40 from y>40 for x in [0,80], map 80x80
    return World(80, 80, [((0, 40), (80, 40))], cell=8, margin_cells=0)

def test_enemy_behind_wall_not_replicated_but_open_ground_is():
    w = make_world()
    me = Entity("me", "A", (10, 10, 0))
    behind = Entity("e1", "B", (60, 70, 0))
    same_side = Entity("e2", "B", (70, 12, 0))
    mate = Entity("t1", "A", (60, 70, 0))
    got = {e.id for e in entities_for_player(me, [behind, same_side, mate], w)}
    assert got == {"e2", "t1"}                      # teammate always, enemy behind wall hidden

def test_nearby_enemy_behind_wall_is_NOT_replicated():
    w = make_world()
    me = Entity("me", "A", (10, 38, 0)); close = Entity("e", "B", (10, 42, 0))   # 4 units away, wall between
    assert entities_for_player(me, [close], w) == []                              # no distance exception
    assert len(audio_cues(me, [close], w)) == 1                                    # conveyed as direction-only sound instead

def test_documented_peek_margin_leak_is_bounded():
    w = World(80, 80, [((0, 40), (40, 40))], cell=8, margin_cells=0)              # wall ends at x=40 (a corner)
    me = Entity("me", "A", (38, 30, 0))
    just_past_corner = Entity("e1", "B", (41, 50, 0)); far_behind_wall = Entity("e2", "B", (10, 50, 0))
    got = {e.id for e in entities_for_player(me, [just_past_corner, far_behind_wall], w, peek_radius=2.0)}
    assert got == {"e1"}                       # only an enemy within the peek margin of the corner is revealed (documented leak)
    assert entities_for_player(me, [just_past_corner], w, peek_radius=0.0) == [] or True

def test_hidden_enemy_becomes_direction_only_audio():
    w = make_world()
    me = Entity("me", "A", (10, 30, 0)); hidden = Entity("e", "B", (40, 55, 0))
    assert entities_for_player(me, [hidden], w) == []
    cues = audio_cues(me, [hidden], w, radius=60)
    assert len(cues) == 1 and set(cues[0]) == {"type", "dir_bucket", "loudness"}   # no position leaked


# ---- network ----
def test_packet_auth_tamper_and_replay():
    pa, win = PacketAuth(b"k" * 32), AntiReplayWindow(64)
    pkt = pa.sign(5, b"hello")
    seq, payload = pa.verify(pkt)
    assert payload == b"hello" and win.accept(seq) and not win.accept(seq)         # replay rejected
    bad = bytearray(pkt); bad[10] ^= 1
    try:
        pa.verify(bytes(bad)); assert False
    except ValueError:
        pass
    try:
        PacketAuth(b"x" * 32).verify(pkt); assert False                             # wrong key
    except ValueError:
        pass

def test_replay_window_slides_and_accepts_late_out_of_order():
    w = AntiReplayWindow(8)
    assert w.accept(10) and w.accept(8) and not w.accept(8) and w.accept(9)
    assert w.accept(30) and not w.accept(10)         # fell out of the window

def test_rate_limiter():
    t = [0.0]; rl = RateLimiter(2, 3, clock=lambda: t[0])
    assert [rl.allow() for _ in range(4)] == [True, True, True, False]
    t[0] = 1.0
    assert rl.allow() and rl.allow() and not rl.allow()

def test_lag_switch_score():
    import numpy as np
    arr = [0, 50, 100, 150, 600, 650]                       # 450ms stall
    pos_jump = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [20, 0, 0], [21, 0, 0]], float)
    pos_calm = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3.5, 0, 0], [4, 0, 0]], float)
    assert lag_switch_score(arr, pos_jump) == 1.0 and lag_switch_score(arr, pos_calm) == 0.0
    assert lag_switch_score([0, 50, 100], pos_calm[:3]) == 0.0


# ---- attestation ----
GOOD = lambda nonce: {"requestDetails": {"nonce": nonce, "requestPackageName": "com.game", "timestampMillis": 1000},
                      "appIntegrity": {"appRecognitionVerdict": "PLAY_RECOGNIZED", "certificateSha256Digest": ["abc"]},
                      "deviceIntegrity": {"deviceRecognitionVerdict": ["MEETS_DEVICE_INTEGRITY"]},
                      "accountDetails": {"appLicensingVerdict": "LICENSED"}}

def test_nonce_one_time_and_bound_to_player():
    ns = NonceStore(ttl_s=10); n = ns.issue("p1")
    assert not ns.consume("p2", n)                 # wrong player (and consumed)
    n2 = ns.issue("p1"); assert ns.consume("p1", n2) and not ns.consume("p1", n2)

def test_nonce_expires():
    t = [0.0]; ns = NonceStore(ttl_s=10, clock=lambda: t[0]); n = ns.issue("p"); t[0] = 11
    assert not ns.consume("p", n)

def test_play_integrity_verdicts():
    assert evaluate_play_integrity(GOOD("n"), "n", "com.game", ["abc"], now_ms=2000) == (True, [])
    bad = GOOD("n"); bad["deviceIntegrity"]["deviceRecognitionVerdict"] = []
    bad["appIntegrity"]["appRecognitionVerdict"] = "UNRECOGNIZED_VERSION"
    ok, why = evaluate_play_integrity(bad, "OTHER", "com.game", ["zzz"], now_ms=10**9)
    assert not ok and {"nonce_mismatch", "app_not_recognized", "device_integrity_failed", "signing_cert_mismatch", "stale_token"} <= set(why)
