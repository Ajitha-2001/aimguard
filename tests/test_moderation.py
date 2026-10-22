"""Regression tests for the moderation workflow (each one was a reproduced bug)."""
import json
import pandas as pd
import pytest
from aimguard.store import Store, ENG_DB_COLS
from aimguard.moderation import Moderation

H = 3600


def mk():
    clock = [1_000_000.0]
    s = Store(clock=lambda: clock[0])
    return s, Moderation(s), clock


def flag(mod, pid):
    mod.create_flags(pd.DataFrame([dict(player_id=pid, action="human_review", fired={}, p_headshot=1e-9,
                                        p_preaim=1.0, hard_evidence=False, risk_score=0.99)]))


def ban_and_enforce(mod, clock, pid, mods=("m1", "m2")):
    flag(mod, pid)
    for m in mods:
        mod.submit_review(m, pid, "cheater")
    eid = mod.request_ban(mods[0], pid)
    clock[0] += 73 * H
    assert mod.run_ban_wave()["enforced"] == 1
    return eid


def overturn(mod, pid, eid, resolver="m3"):
    aid = mod.file_appeal(pid, eid, "I did not cheat")
    mod.resolve_appeal(resolver, aid, "overturned")


def status(s, eid):
    return s.q("SELECT status FROM enforcement WHERE id=?", (eid,))[0]["status"]


# ---- BUG 1: overturned ban re-scheduled with old confirmations ----
def test_overturned_ban_cannot_be_rescheduled_with_old_confirmations():
    s, mod, clock = mk(); eid = ban_and_enforce(mod, clock, "p"); overturn(mod, "p", eid)
    with pytest.raises(ValueError, match="no open case"):
        mod.request_ban("m1", "p")

def test_fresh_reviews_alone_are_not_new_evidence():
    s, mod, clock = mk(); eid = ban_and_enforce(mod, clock, "p"); overturn(mod, "p", eid)
    mod.submit_review("m1", "p", "cheater"); mod.submit_review("m2", "p", "cheater")   # fresh reviews, but nothing new
    with pytest.raises(ValueError, match="no open case"):
        mod.request_ban("m1", "p")

def test_new_flag_needs_fresh_reviews_old_ones_do_not_count():
    s, mod, clock = mk(); eid = ban_and_enforce(mod, clock, "p"); overturn(mod, "p", eid)
    flag(mod, "p")                                                                      # new evidence arrives
    mod.submit_review("m1", "p", "cheater")                                             # only ONE fresh review
    with pytest.raises(ValueError, match="2 distinct"):
        mod.request_ban("m1", "p")                                                      # old m2 confirmation is gone
    mod.submit_review("m2", "p", "cheater")
    assert mod.request_ban("m1", "p")                                                   # new case, new reviews: allowed

def test_hard_evidence_from_the_closed_case_does_not_carry_over():
    s, mod, clock = mk(); s.add_evidence("p", "server_violation", "old")
    eid = ban_and_enforce(mod, clock, "p", mods=("m1",)); overturn(mod, "p", eid)
    assert "p" not in s.hard_evidence_players()
    flag(mod, "p"); mod.submit_review("m1", "p", "cheater")
    with pytest.raises(ValueError, match="2 distinct"):                                 # now statistics-only again
        mod.request_ban("m1", "p")

def test_new_hard_evidence_opens_a_new_case():
    s, mod, clock = mk(); eid = ban_and_enforce(mod, clock, "p"); overturn(mod, "p", eid)
    assert s.add_evidence("p", "server_violation", "new") is not None
    mod.submit_review("m1", "p", "cheater")
    assert mod.request_ban("m1", "p")                                                   # hard evidence: one fresh review suffices

def test_matches_before_the_overturn_no_longer_count_as_evidence():
    s, mod, clock = mk()
    row = lambda mid: pd.DataFrame([dict(player_id="p", match_id=mid, match_ts=1.0, bracket=0, shots=1, hits=1,
                                         headshots=1, react_ms=100.0, linearity=0.9, omega_peak=300.0,
                                         preaim_err=5.0, null_preaim_err=30.0)])[ENG_DB_COLS]
    s.add_engagements(row("old")); assert len(s.load_engagements()) == 1
    eid = ban_and_enforce(mod, clock, "p"); overturn(mod, "p", eid)
    assert s.load_engagements().empty                                                   # old match ignored
    clock[0] += 10; s.add_engagements(row("new")); assert s.load_engagements().match_id.tolist() == ["new"]
    assert s.q("SELECT risk_score, action FROM players WHERE player_id='p'")[0] == {"risk_score": 0.0, "action": "none"}   # stale score cleared


# ---- BUG 2: scheduled ban executes after a reviewer flips to legit ----
def scheduled(mod, pid):
    flag(mod, pid); mod.submit_review("m1", pid, "cheater"); mod.submit_review("m2", pid, "cheater")
    return mod.request_ban("m1", pid)

def test_reviewer_flip_to_legit_cancels_the_scheduled_ban():
    s, mod, clock = mk(); eid = scheduled(mod, "q")
    mod.submit_review("m2", "q", "legit")
    assert status(s, eid) == "cancelled"
    clock[0] += 73 * H; assert mod.run_ban_wave() == {"enforced": 0, "cancelled": 0}
    assert mod.player_notice("q") is None

def test_ban_wave_rechecks_approval_even_if_the_row_changed_behind_the_workflow():
    s, mod, clock = mk(); eid = scheduled(mod, "q")
    s.x("INSERT INTO reviews(player_id,moderator_id,decision,notes,reviewed_at,epoch) VALUES('q','m2','legit','',0,0)")   # bypass submit_review
    assert status(s, eid) == "scheduled"
    clock[0] += 73 * H
    assert mod.run_ban_wave() == {"enforced": 0, "cancelled": 1}
    assert status(s, eid) == "cancelled" and mod.player_notice("q") is None
    assert "ban_cancelled" in {r["action"] for r in s.q("SELECT action FROM audit_log")}

def test_flip_to_unsure_dropping_below_required_confirmations_cancels():
    s, mod, clock = mk(); eid = scheduled(mod, "q")
    mod.submit_review("m2", "q", "unsure")
    assert status(s, eid) == "cancelled"

def test_valid_ban_still_enforced_and_extra_confirmation_survives_one_flip():
    s, mod, clock = mk(); flag(mod, "q")
    for m in ("m1", "m2", "m3"): mod.submit_review(m, "q", "cheater")
    eid = mod.request_ban("m1", "q"); mod.submit_review("m3", "q", "unsure")             # still 2 confirmations
    assert status(s, eid) == "scheduled"
    clock[0] += 73 * H; assert mod.run_ban_wave() == {"enforced": 1, "cancelled": 0}

def test_ban_wave_does_not_run_early():
    s, mod, clock = mk(); scheduled(mod, "q"); clock[0] += 10 * H
    assert mod.run_ban_wave() == {"enforced": 0, "cancelled": 0}


# ---- evidence trust levels ----
def test_reported_hash_is_soft_until_a_moderator_verifies_it():
    s, mod, clock = mk()
    eid = s.add_evidence("p", "file_hash_reported", "abc")
    assert "p" not in s.hard_evidence_players() and "p" in s.players_with(("file_hash_reported",))
    with pytest.raises(ValueError):
        mod.verify_hash_evidence("m1", "other_player", eid)                             # wrong player
    mod.verify_hash_evidence("m1", "p", eid)
    assert "p" in s.hard_evidence_players()
    with pytest.raises(ValueError, match="already verified"):
        mod.verify_hash_evidence("m1", "p", eid)
    assert "evidence_verified" in {r["action"] for r in s.q("SELECT action FROM audit_log")}

def test_old_databases_are_upgraded():
    import sqlite3, tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "old.db")
    c = sqlite3.connect(path); c.executescript("CREATE TABLE players(player_id TEXT PRIMARY KEY, bracket INTEGER);"
        "CREATE TABLE reviews(review_id INTEGER PRIMARY KEY, player_id TEXT, flag_id INTEGER, moderator_id TEXT, decision TEXT, notes TEXT, reviewed_at REAL);"
        "CREATE TABLE evidence(id INTEGER PRIMARY KEY, player_id TEXT, kind TEXT, ref TEXT, at REAL);"); c.close()
    s = Store(path)
    assert {"case_epoch", "evidence_cutoff"} <= {r["name"] for r in s.q("PRAGMA table_info(players)")}
