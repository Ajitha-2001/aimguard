"""Moderator workflow: flags -> reviews -> (four-eyes) ban request -> delayed ban wave -> appeals -> labels.

Safety properties (each covered by tests):
 * Approval is RE-CHECKED when the ban wave runs; a reviewer changing to 'legit' cancels a scheduled ban.
 * An overturned appeal closes the case. Old reviews/evidence/matches no longer count; another ban needs
   new evidence (new flag or new hard evidence) AND fresh moderator reviews.
 * Statistics alone need independent confirmations; mass reports never trigger action.
"""
import json
import pandas as pd

PLAYER_NOTICE = "Your account was found to have violated the fair play rules."   # never reveal detection details


class Moderation:
    def __init__(self, store, required_confirmations=2, ban_delay_s=72 * 3600, appeal_window_s=14 * 86400, clock=None):
        self.s, self.required, self.delay, self.appeal_window = store, required_confirmations, ban_delay_s, appeal_window_s
        self.clock = clock or store.clock

    # ---- flags ----
    def create_flags(self, risk: pd.DataFrame):
        n = 0
        for r in risk.itertuples():
            if r.action not in ("human_review", "ban_eligible_after_review"):
                continue
            ev = json.dumps({"signals": r.fired, "p_headshot": r.p_headshot, "p_preaim": r.p_preaim,
                             "hard_evidence": bool(r.hard_evidence)}, default=int)
            open_ = self.s.q("SELECT flag_id FROM flags WHERE player_id=? AND status='open'", (r.player_id,))
            if open_:                                    # refresh, never duplicate
                self.s.x("UPDATE flags SET severity=?, evidence=? WHERE flag_id=?", (float(r.risk_score), ev, open_[0]["flag_id"]))
            else:
                self.s.x("INSERT INTO flags(player_id,created_at,detector,severity,evidence) VALUES(?,?,?,?,?)",
                         (r.player_id, self.clock(), "aimguard", float(r.risk_score), ev)); n += 1
        return n

    def queue(self, status="open", min_severity=0.0):
        rows = self.s.q("SELECT * FROM flags WHERE status=? AND severity>=? ORDER BY severity DESC", (status, min_severity))
        for r in rows:
            r["evidence"] = json.loads(r["evidence"])
            r["weighted_reports"] = self.s.weighted_report_score(r["player_id"])
        return rows

    # ---- approval rules (single source of truth, used at request time AND at execution time) ----
    def _latest_by_moderator(self, player_id):
        out = {}
        for r in self.s.q("SELECT moderator_id, decision FROM reviews WHERE player_id=? AND epoch=? "
                          "AND moderator_id NOT LIKE 'system:%' ORDER BY review_id", (player_id, self.s.epoch(player_id))):
            out[r["moderator_id"]] = r["decision"]
        return out

    def _approval(self, player_id):
        """(ok, reason). Only reviews/evidence/flags of the player's CURRENT case count."""
        latest = self._latest_by_moderator(player_id)
        if "legit" in latest.values():
            return False, "moderators disagree (a 'legit' decision is outstanding); resolve before banning"
        hard = player_id in self.s.hard_evidence_players()
        if not hard and not self.s.q("SELECT 1 FROM flags WHERE player_id=? AND status='open'", (player_id,)):
            return False, "no open case: a ban needs new evidence (a new flag or new hard evidence)"
        need = 1 if hard else self.required            # statistics alone need independent confirmations
        have = sum(d == "cheater" for d in latest.values())
        if have < need:
            return False, (f"ban needs {need} distinct moderator confirmation(s); has {have}"
                           + ("" if hard else " (statistical evidence only)"))
        return True, ""

    # ---- reviews ----
    def submit_review(self, moderator, player_id, decision, notes="", flag_id=None):
        if decision not in ("cheater", "legit", "unsure"):
            raise ValueError("decision must be cheater, legit or unsure")
        rid = self.s.x("INSERT INTO reviews(player_id,flag_id,moderator_id,decision,notes,reviewed_at,epoch) VALUES(?,?,?,?,?,?,?)",
                       (player_id, flag_id, moderator, decision, notes, self.clock(), self.s.epoch(player_id)))
        if decision == "legit":
            self.s.x("UPDATE flags SET status='closed' WHERE player_id=? AND status='open'", (player_id,))
        self.s.audit(moderator, "review", player_id, {"decision": decision, "review_id": rid}, self.clock())
        self._recheck_scheduled(player_id, f"review changed by {moderator}")
        return rid

    def _recheck_scheduled(self, player_id, why):
        ok, reason = self._approval(player_id)
        if ok:
            return 0
        rows = self.s.q("SELECT id FROM enforcement WHERE player_id=? AND status='scheduled'", (player_id,))
        for e in rows:
            self.s.x("UPDATE enforcement SET status='cancelled' WHERE id=?", (e["id"],))
            self.s.audit("system", "ban_cancelled", player_id, {"enforcement_id": e["id"], "why": why, "reason": reason}, self.clock())
        return len(rows)

    # ---- bans ----
    def request_ban(self, moderator, player_id):
        if self.s.q("SELECT 1 FROM enforcement WHERE player_id=? AND status IN ('scheduled','active')", (player_id,)):
            raise ValueError("player already has a scheduled or active ban")
        ok, reason = self._approval(player_id)
        if not ok:
            raise ValueError(reason)
        ep = self.s.epoch(player_id)
        refs = {"epoch": ep, "reviews": [r["review_id"] for r in self.s.q(
                    "SELECT review_id FROM reviews WHERE player_id=? AND decision='cheater' AND epoch=?", (player_id, ep))],
                "evidence": [e["kind"] for e in self.s.evidence_rows(player_id)]}
        now = self.clock()
        eid = self.s.x("INSERT INTO enforcement(player_id,action,reason,evidence_refs,status,requested_by,scheduled_at,execute_after) "
                       "VALUES(?,?,?,?,?,?,?,?)", (player_id, "ban", "fair_play_violation", json.dumps(refs), "scheduled", moderator, now, now + self.delay))
        self.s.audit(moderator, "ban_requested", player_id, {"enforcement_id": eid}, now)
        return eid

    def run_ban_wave(self):
        """Enforce due bans together (so cheat makers can't tell what was detected when). Approval is RE-CHECKED
        here: a ban whose approval no longer holds is cancelled, never executed."""
        now, done, cancelled = self.clock(), 0, 0
        for e in self.s.q("SELECT id, player_id FROM enforcement WHERE status='scheduled' AND execute_after<=?", (now,)):
            ok, reason = self._approval(e["player_id"])
            if not ok:
                self.s.x("UPDATE enforcement SET status='cancelled' WHERE id=?", (e["id"],))
                self.s.audit("system", "ban_cancelled", e["player_id"], {"enforcement_id": e["id"], "at": "ban_wave", "reason": reason}, now)
                cancelled += 1
                continue
            self.s.x("UPDATE enforcement SET status='active', appealable_until=? WHERE id=?", (now + self.appeal_window, e["id"]))
            self.s.audit("system", "ban_enforced", e["player_id"], {"enforcement_id": e["id"]}, now); done += 1
        return {"enforced": done, "cancelled": cancelled}

    def player_notice(self, player_id):
        return PLAYER_NOTICE if self.s.q("SELECT 1 FROM enforcement WHERE player_id=? AND status='active'", (player_id,)) else None

    # ---- evidence verification ----
    def verify_hash_evidence(self, moderator, player_id, evidence_id):
        """A client-REPORTED file-hash match is soft (clients can forge or omit reports). A moderator who has independently
        confirmed it (sample analysis, replay, a trusted re-scan) promotes it to hard evidence. Audited."""
        ev = self.s.q("SELECT * FROM evidence WHERE id=? AND player_id=? AND kind='file_hash_reported' AND epoch=?",
                      (evidence_id, player_id, self.s.epoch(player_id)))
        if not ev:
            raise ValueError("no such unverified file-hash report for this player on the current case")
        vid = self.s.add_evidence(player_id, "file_hash", ev[0]["ref"])
        if vid is None:
            raise ValueError("already verified")
        self.s.audit(moderator, "evidence_verified", player_id, {"evidence_id": evidence_id, "verified_id": vid}, self.clock())
        return vid

    # ---- appeals ----
    def file_appeal(self, player_id, enforcement_id, text):
        e = self.s.q("SELECT * FROM enforcement WHERE id=? AND player_id=?", (enforcement_id, player_id))
        if not e or e[0]["status"] != "active":
            raise ValueError("no active enforcement to appeal")
        if self.clock() > e[0]["appealable_until"]:
            raise ValueError("appeal window has closed")
        if self.s.q("SELECT 1 FROM appeals WHERE enforcement_id=? AND status='open'", (enforcement_id,)):
            raise ValueError("an appeal is already open")
        aid = self.s.x("INSERT INTO appeals(enforcement_id,player_id,text,created_at) VALUES(?,?,?,?)",
                       (enforcement_id, player_id, text[:4000], self.clock()))
        self.s.audit(player_id, "appeal_filed", player_id, {"appeal_id": aid}, self.clock())
        return aid

    def resolve_appeal(self, moderator, appeal_id, outcome, notes=""):
        if outcome not in ("upheld", "overturned"):
            raise ValueError("outcome must be upheld or overturned")
        a = self.s.q("SELECT * FROM appeals WHERE id=? AND status='open'", (appeal_id,))
        if not a:
            raise ValueError("no such open appeal")
        a = a[0]
        involved = {r["moderator_id"] for r in self.s.q("SELECT moderator_id FROM reviews WHERE player_id=?", (a["player_id"],))}
        involved.add(self.s.q("SELECT requested_by FROM enforcement WHERE id=?", (a["enforcement_id"],))[0]["requested_by"])
        if moderator in involved:
            raise ValueError("appeal must be resolved by a moderator who was not involved in the original decision")
        now = self.clock()
        self.s.x("UPDATE appeals SET status=?, resolved_by=?, resolved_at=?, notes=? WHERE id=?", (outcome, moderator, now, notes, appeal_id))
        if outcome == "overturned":
            self.s.x("UPDATE enforcement SET status='reverted' WHERE id=?", (a["enforcement_id"],))
            self.s.x("UPDATE flags SET status='closed' WHERE player_id=?", (a["player_id"],))
            self.s.close_case(a["player_id"], now)          # old reviews/evidence/matches stop counting
            self.s.x("INSERT INTO reviews(player_id,moderator_id,decision,notes,reviewed_at,epoch) VALUES(?,?,?,?,?,?)",
                     (a["player_id"], f"system:appeal:{appeal_id}", "legit", "ban overturned on appeal", now, self.s.epoch(a["player_id"])))
        self.s.audit(moderator, "appeal_resolved", a["player_id"], {"appeal_id": appeal_id, "outcome": outcome}, now)

    def export_labels(self) -> pd.DataFrame:
        """Latest confirmed label per player (cheater/legit) for retraining / threshold tuning.
        Appeal outcomes override earlier moderator decisions."""
        rows = self.s.q("SELECT player_id, decision FROM reviews WHERE decision!='unsure' ORDER BY review_id")
        last = {r["player_id"]: r["decision"] for r in rows}
        return pd.DataFrame([{"player_id": p, "label": d} for p, d in last.items()], columns=["player_id", "label"])
