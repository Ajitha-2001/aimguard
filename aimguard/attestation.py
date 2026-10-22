"""Layer 5 (server side): nonce binding + evaluation of an already-DECODED Google Play Integrity verdict.
Decoding the token needs Google credentials and network access, so it is injected as `decoder`
(see api.py); this module only decides what the verdict means. Client checks can be bypassed:
treat the result as a risk signal combined with server-side evidence, never as the only basis for a ban."""
import secrets, threading, time


class NonceStore:
    def __init__(self, ttl_s=300, clock=time.time):
        self.ttl, self.clock, self._n, self._lock = ttl_s, clock, {}, threading.Lock()

    def issue(self, player_id):
        n = secrets.token_urlsafe(24)
        with self._lock:
            self._n[n] = (player_id, self.clock())
        return n

    def consume(self, player_id, nonce):
        """One-time use; must belong to this player and be fresh."""
        with self._lock:
            rec = self._n.pop(nonce, None)
        return bool(rec) and rec[0] == player_id and self.clock() - rec[1] <= self.ttl


def evaluate_play_integrity(verdict: dict, expected_nonce, expected_package, allowed_cert_sha256=None,
                            now_ms=None, max_age_ms=300_000):
    """Returns (ok, reasons). `verdict` is the decoded `tokenPayloadExternal` dict."""
    reasons = []
    rd = verdict.get("requestDetails", {})
    if rd.get("nonce") != expected_nonce:
        reasons.append("nonce_mismatch")
    if rd.get("requestPackageName") != expected_package:
        reasons.append("package_mismatch")
    if now_ms is not None and now_ms - int(rd.get("timestampMillis", 0)) > max_age_ms:
        reasons.append("stale_token")
    app = verdict.get("appIntegrity", {})
    if app.get("appRecognitionVerdict") != "PLAY_RECOGNIZED":
        reasons.append("app_not_recognized")
    if allowed_cert_sha256 is not None and not (set(app.get("certificateSha256Digest", [])) & set(allowed_cert_sha256)):
        reasons.append("signing_cert_mismatch")
    if "MEETS_DEVICE_INTEGRITY" not in verdict.get("deviceIntegrity", {}).get("deviceRecognitionVerdict", []):
        reasons.append("device_integrity_failed")
    if verdict.get("accountDetails", {}).get("appLicensingVerdict") != "LICENSED":
        reasons.append("unlicensed")
    return (not reasons), reasons
