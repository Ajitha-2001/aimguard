"""SQLite persistence (stdlib). Swap for PostgreSQL/ClickHouse at scale; the interface stays the same."""
import hashlib, json, secrets, sqlite3, threading, time
from contextlib import contextmanager
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS players(player_id TEXT PRIMARY KEY, bracket INTEGER, risk_score REAL DEFAULT 0,
  action TEXT DEFAULT 'none', fired TEXT DEFAULT '{}', risk_updated REAL,
  case_epoch INTEGER DEFAULT 0, evidence_cutoff REAL);
CREATE TABLE IF NOT EXISTS ingested(match_id TEXT, player_id TEXT, at REAL, PRIMARY KEY(match_id, player_id));
CREATE TABLE IF NOT EXISTS engagements(player_id TEXT, match_id TEXT, match_ts REAL, bracket INTEGER,
  shots INTEGER, hits INTEGER, headshots INTEGER, react_ms REAL, linearity REAL, omega_peak REAL,
  preaim_err REAL, null_preaim_err REAL);
CREATE INDEX IF NOT EXISTS eng_player ON engagements(player_id);
CREATE TABLE IF NOT EXISTS evidence(id INTEGER PRIMARY KEY AUTOINCREMENT, player_id TEXT, kind TEXT, ref TEXT, at REAL, epoch INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS flags(flag_id INTEGER PRIMARY KEY AUTOINCREMENT, player_id TEXT, created_at REAL,
  detector TEXT, severity REAL, evidence TEXT, status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS reviews(review_id INTEGER PRIMARY KEY AUTOINCREMENT, player_id TEXT, flag_id INTEGER,
  moderator_id TEXT, decision TEXT CHECK(decision IN ('cheater','legit','unsure')), notes TEXT, reviewed_at REAL, epoch INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS enforcement(id INTEGER PRIMARY KEY AUTOINCREMENT, player_id TEXT, action TEXT,
  reason TEXT, evidence_refs TEXT, status TEXT, requested_by TEXT, scheduled_at REAL, execute_after REAL,
  appealable_until REAL);
CREATE TABLE IF NOT EXISTS appeals(id INTEGER PRIMARY KEY AUTOINCREMENT, enforcement_id INTEGER, player_id TEXT,
  text TEXT, status TEXT DEFAULT 'open', created_at REAL, resolved_by TEXT, resolved_at REAL, notes TEXT);
CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY AUTOINCREMENT, reporter_id TEXT, target_id TEXT, reason TEXT, at REAL);
CREATE TABLE IF NOT EXISTS tokens(token_hash TEXT PRIMARY KEY, role TEXT, subject TEXT);
CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, actor TEXT, action TEXT, target TEXT, details TEXT);
"""
# Trust levels. HARD = observed by the server itself, or verified by a moderator. Anything a client merely
# CLAIMS (file hashes, loaded modules, attestation failures) is SOFT: clients can be modified to forge or omit it.
HARD_KINDS = ("file_hash", "server_violation")          # 'file_hash' here means moderator-VERIFIED
SOFT_KINDS = ("unknown_modules", "attestation_failed")
REPORTED_HASH = "file_hash_reported"                    # client-reported blocklist match, unverified
MIGRATIONS = [("players", "case_epoch", "INTEGER DEFAULT 0"), ("players", "evidence_cutoff", "REAL"),
              ("evidence", "epoch", "INTEGER DEFAULT 0"), ("reviews", "epoch", "INTEGER DEFAULT 0")]
ENG_DB_COLS = ["player_id", "match_id", "match_ts", "bracket", "shots", "hits", "headshots", "react_ms",
               "linearity", "omega_peak", "preaim_err", "null_preaim_err"]


class Store:
    def __init__(self, path=":memory:", clock=time.time):
        self.clock = clock
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)
            for table, col, ddl in MIGRATIONS:      # upgrade databases created by older versions
                if col not in {r['name'] for r in self.db.execute(f'PRAGMA table_info({table})')}:
                    self.db.execute(f'ALTER TABLE {table} ADD COLUMN {col} {ddl}')

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).lastrowid

    @contextmanager
    def tx(self):
        with self.lock:
            self.db.execute("BEGIN")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def audit(self, actor, action, target="", details=None, now=None):
        self.x("INSERT INTO audit_log(at,actor,action,target,details) VALUES(?,?,?,?,?)",
               (now or self.clock(), actor, action, str(target), json.dumps(details or {}, default=str)))

    # ---- auth ----
    @staticmethod
    def _h(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def create_token(self, role, subject):
        t = secrets.token_urlsafe(32)
        self.x("INSERT INTO tokens VALUES(?,?,?)", (self._h(t), role, subject))
        return t

    def auth(self, token):
        r = self.q("SELECT role, subject FROM tokens WHERE token_hash=?", (self._h(token),))
        return (r[0]["role"], r[0]["subject"]) if r else None

    # ---- telemetry ----
    def add_engagements(self, df: pd.DataFrame, now=None) -> int:
        """Idempotent per (match_id, player_id): re-uploading a match never double-counts evidence."""
        added = 0
        with self.tx():
            for (pid, mid), g in df.groupby(["player_id", "match_id"]):
                if self.q("SELECT 1 FROM ingested WHERE match_id=? AND player_id=?", (mid, pid)):
                    continue
                g = g[ENG_DB_COLS].astype(object).where(g[ENG_DB_COLS].notna(), None)
                self.db.executemany("INSERT INTO engagements VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                    [tuple(r) for r in g.itertuples(index=False)])
                self.db.execute("INSERT INTO ingested VALUES(?,?,?)", (mid, pid, now or self.clock()))
                self.db.execute("INSERT INTO players(player_id,bracket) VALUES(?,?) "
                                "ON CONFLICT(player_id) DO UPDATE SET bracket=excluded.bracket", (pid, int(g.bracket.iloc[0])))
                added += len(g)
        return added

    def load_engagements(self) -> pd.DataFrame:
        with self.lock:
            # after an overturned ban, matches ingested before the overturn no longer count as evidence
            return pd.read_sql_query(
                "SELECT e.* FROM engagements e JOIN ingested i ON i.match_id=e.match_id AND i.player_id=e.player_id "
                "LEFT JOIN players p ON p.player_id=e.player_id "
                "WHERE p.evidence_cutoff IS NULL OR i.at > p.evidence_cutoff", self.db)

    def purge_before(self, ts):
        """Retention: delete raw engagement rows older than ts (features/evidence for confirmed cases stay)."""
        with self.tx():
            n = self.db.execute("DELETE FROM engagements WHERE match_ts < ?", (ts,)).rowcount
            self.db.execute("DELETE FROM ingested WHERE NOT EXISTS (SELECT 1 FROM engagements e "
                            "WHERE e.match_id=ingested.match_id AND e.player_id=ingested.player_id)")
        return n

    # ---- risk & evidence ----
    def set_risk(self, pid, score, action, fired, now=None):
        self.x("INSERT INTO players(player_id,risk_score,action,fired,risk_updated) VALUES(?,?,?,?,?) "
               "ON CONFLICT(player_id) DO UPDATE SET risk_score=excluded.risk_score, action=excluded.action, "
               "fired=excluded.fired, risk_updated=excluded.risk_updated",
               (pid, float(score), action, json.dumps(fired, default=int), now or self.clock()))

    def epoch(self, pid):
        r = self.q("SELECT case_epoch FROM players WHERE player_id=?", (pid,))
        return r[0]["case_epoch"] if r else 0

    def add_evidence(self, pid, kind, ref="", now=None):
        """Returns the new evidence id, or None if identical evidence is already on this case."""
        ep = self.epoch(pid)
        if self.q("SELECT 1 FROM evidence WHERE player_id=? AND kind=? AND ref=? AND epoch=?", (pid, kind, ref, ep)):
            return None
        return self.x("INSERT INTO evidence(player_id,kind,ref,at,epoch) VALUES(?,?,?,?,?)", (pid, kind, ref, now or self.clock(), ep))

    def evidence_rows(self, pid, kinds=None):
        rows = self.q("SELECT id, kind, ref, at FROM evidence WHERE player_id=? AND epoch=?", (pid, self.epoch(pid)))
        return [r for r in rows if kinds is None or r["kind"] in kinds]

    def players_with(self, kinds):
        """Players with evidence of these kinds ON THEIR CURRENT CASE (evidence from closed cases is ignored)."""
        qs = ",".join("?" * len(kinds))
        return {r["player_id"] for r in self.q(
            f"SELECT DISTINCT e.player_id FROM evidence e LEFT JOIN players p ON p.player_id=e.player_id "
            f"WHERE e.kind IN ({qs}) AND e.epoch=COALESCE(p.case_epoch,0)", tuple(kinds))}

    def hard_evidence_players(self):
        return self.players_with(HARD_KINDS)

    def soft_signal_players(self):
        return self.players_with(SOFT_KINDS)

    def close_case(self, pid, now=None):
        """Appeal overturned: start a new case. Old reviews/evidence stop counting, pre-overturn matches
        stop counting, and the stale risk score is cleared. A new ban needs NEW evidence + fresh reviews."""
        self.x("INSERT INTO players(player_id,case_epoch,evidence_cutoff) VALUES(?,1,?) ON CONFLICT(player_id) DO UPDATE SET "
               "case_epoch=case_epoch+1, evidence_cutoff=excluded.evidence_cutoff, risk_score=0, action='none', fired='{}'",
               (pid, now or self.clock()))

    # ---- reports ----
    def add_report(self, reporter, target, reason, now=None):
        self.x("INSERT INTO reports(reporter_id,target_id,reason,at) VALUES(?,?,?,?)", (reporter, target, reason, now or self.clock()))

    def reporter_reliability(self, reporter):
        """(1+confirmed)/(2+confirmed+refuted) from moderator outcomes on this reporter's targets."""
        c = r = 0
        for t in {x["target_id"] for x in self.q("SELECT target_id FROM reports WHERE reporter_id=?", (reporter,))}:
            last = self.q("SELECT decision FROM reviews WHERE player_id=? AND decision!='unsure' ORDER BY review_id DESC LIMIT 1", (t,))
            if last:
                c, r = c + (last[0]["decision"] == "cheater"), r + (last[0]["decision"] == "legit")
        return (1 + c) / (2 + c + r)

    def weighted_report_score(self, target):
        return round(sum(self.reporter_reliability(x["reporter_id"]) for x in
                         self.q("SELECT DISTINCT reporter_id FROM reports WHERE target_id=?", (target,))), 3)
