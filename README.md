# AimGuard: server-side anti-cheat PROTOTYPE

**Status: prototype. Not ready for real enforcement.** The server-side components below exist and pass their tests on
synthetic data, but the system has never run on a real game, has had no security review, and several parts (client
protection, game integration, scale) are outside this repo. Do not ban real players with it.

## What is implemented
| Layer (from the design) | Module | Notes |
|---|---|---|
| Raw telemetry -> features | `ticks.py` | crosshair ticks + shot + visibility events -> reaction time, aim path linearity, peak aim speed, headshots, pre-aim error vs a decoy null |
| Real-time validation | `validation.py` | speed/teleport/fire-rate rules, violation budget (no single-violation punishment), lag-compensated hit re-check with bounded rewind |
| Information hiding | `pvs.py` | grid PVS + line of sight. Enemies without line of sight are not replicated, **however close** (no distance exception); hidden-but-near enemies become direction-only audio cues. Deliberate, bounded leaks: teammates, enemies within `peek_radius` of a corner edge, coarse audio direction. 2D walls only (no floors/doors/props) |
| Network | `network.py` | packet HMAC, sliding anti-replay window, rate limiter, lag-switch score |
| Attestation (server side) | `attestation.py` | one-time nonces + decision logic for a decoded Play Integrity verdict |
| Behaviour detection | `detectors.py`, `risk.py`, `pipeline.py` | bracket-robust z-scores, binomial headshot test, pre-aim test, Isolation Forest calibrated on held-out clean players, log-odds risk with decay |
| Known-cheat signatures | `signatures.py`, `localscan.py` | hash blocklist, unknown-module signal, scoped local PC scan. Client-reported matches are SOFT evidence (see trust levels) |
| Database | `store.py` | SQLite: engagements, players, flags, reviews, enforcement, appeals, reports, tokens, audit log; idempotent ingest; retention purge |
| Moderator workflow | `moderation.py` | review queue, four-eyes ban rule, delayed ban waves, appeals with independent resolver, label export |
| HTTP API | `api.py` | all endpoints from the design, token auth, roles, rate-limited reports |
| Operations | `manage.py` | demo, train baseline, make tokens, serve, score, ban wave |

## Enforcement rules (each is covered by regression tests)
- Statistics alone never ban: two independent moderators must confirm (one if HARD evidence exists).
- Approval is **re-checked when the ban wave runs**. If a reviewer changes to `legit` (or confirmations drop below the
  requirement) the scheduled ban is cancelled, never executed.
- An overturned appeal **closes the case**: old reviews, old evidence and pre-overturn matches stop counting and the stale
  risk score is cleared. Another ban needs NEW evidence (a new flag from new matches, or new hard evidence) plus FRESH reviews.
- Appeals are resolved by a moderator not involved in the original decision. Mass reports never trigger action. Every step is audited.
- `risk_score` is an experimental ranking, NOT a calibrated probability.

## Evidence trust levels
| Evidence | Source | Level |
|---|---|---|
| Server-observed violation (`server_violation`) | the game server itself | HARD |
| File-hash match **verified by a moderator** (`POST /v1/evidence/verify`) | independent confirmation (sample analysis, replay, trusted re-scan) | HARD |
| File-hash match **reported by the client** | the player's own device, relayed by the game server | SOFT (+3 weight, never reduces the confirmations needed) |
| Unknown modules, failed attestation | client | SOFT (+1) |
Authentication only proves the report came from your game server, not that the hash was really observed on that player's device:
a modified client can forge or omit reports, and a framing attempt could fabricate one. That is why client claims are soft.
`score_real.py` follows the same rule: `--reports` adds soft evidence; `--verified-players ids.txt` supplies hard evidence.

## Run it (Windows PowerShell, from the project folder with the venv active)
    pip install -r requirements.txt
    python manage.py demo                    # trains, ingests raw ticks, scores, prints the review queue
    python -m pytest                         # all tests (takes ~2-3 minutes)
Run a real API server:
    python manage.py train-baseline --reference clean_engagements.csv --out baseline.pkl
    python manage.py make-token --db aimguard.db --role admin --subject you
    python manage.py make-token --db aimguard.db --role game_server --subject server1
    python manage.py make-token --db aimguard.db --role moderator --subject alice
    python manage.py serve --db aimguard.db --baseline baseline.pkl --blocklist blocklist.txt --port 8080
Endpoints (all need `Authorization: Bearer <token>`): `POST /v1/matches/{id}/telemetry`, `GET /v1/players/{id}/risk`,
`GET /v1/flags?status=open&min_severity=`, `POST /v1/reviews`, `POST /v1/bans`, `POST /v1/reports`,
`POST /v1/attestation/nonce|verify`, `POST /v1/evidence`, `POST /v1/evidence/verify` (moderator), `POST /v1/appeals`, `POST /v1/appeals/{id}/resolve`,
`GET /v1/me/status`, `POST /v1/admin/score`, `POST /v1/admin/ban_wave`.

## Telemetry your game server must send
Per player per match (columns as lists): `ticks` (t_ms, yaw, pitch), `shots` (t_ms, target, hit, hitbox),
`vis` (t_ms, target, target_yaw, target_pitch, decoy_yaw, decoy_pitch) plus the player's `bracket`.
`vis` = when an enemy first became visible; `target_*` = direction to it; `decoy_*` = direction to another plausible
enemy position (your integration must supply this; it forms the null for the wallhack test).

## What you still need (not implementable generically)
- **Game integration:** the code inside YOUR game/engine that emits the telemetry above and applies `validation`/`pvs` in the netcode.
- **Client-side protection:** native client checks, integrity of the game binary, and (PC) any driver-level component. `client/` and `localscan.py` are scoped examples only, easily bypassed by a determined cheat.
- **Play Integrity decoding** needs your Google credentials (inject a `decoder`); **Apple App Attest** is not implemented. Packet confidentiality (DTLS/TLS/AEAD) is not implemented, only integrity.
- **Scale:** SQLite + stdlib server suit a prototype. For production use PostgreSQL/ClickHouse, Kafka, worker queues, a production web server/TLS, and secrets management.
- **Real-data validation:** every accuracy figure comes from simulated data. Expect to re-tune thresholds on real matches, run in shadow mode for weeks, and measure precision with moderator labels before enforcing anything.

## Known limitations
- Tests were run in a sandbox with a minimal pytest stand-in; run `python -m pytest` yourself. Passing tests do not establish real-world safety: this has had no independent security review.
- IDs are always treated as strings (`aimguard/io_utils.py`), so `007` in a CSV matches `007` in JSON. Databases from older versions are upgraded automatically on open.
- After an overturned ban a player needs a full new window of matches before they can be scored again.
- Weak/subtle cheating is partly missed (simulated cheats at strength 0.2-0.35 were detected roughly 50-70% of the time); that trades recall for near-zero false positives.
- Baselines need verified-clean reference players (>= ~65 per bracket, same match window); the scoring window is fixed (default 6 matches).
- Legal/privacy: disclose telemetry and any client scanning in your terms, use pseudonymous IDs, set retention, honour local data-protection law.
