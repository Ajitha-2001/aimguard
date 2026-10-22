"""Layer 4: packet integrity, anti-replay, rate limiting, lag-switch detection (stdlib only).
NOTE: this gives integrity + replay protection. For confidentiality use DTLS/TLS or an AEAD library
(e.g. ChaCha20-Poly1305 from `cryptography`); do not invent your own cipher."""
import hashlib, hmac, struct, time
import numpy as np


class PacketAuth:
    TAG = 16

    def __init__(self, session_key: bytes):
        self.key = session_key

    def sign(self, seq: int, payload: bytes) -> bytes:
        head = struct.pack(">Q", seq)
        return head + payload + hmac.new(self.key, head + payload, hashlib.sha256).digest()[:self.TAG]

    def verify(self, packet: bytes):
        if len(packet) < 8 + self.TAG:
            raise ValueError("packet too short")
        head, payload, tag = packet[:8], packet[8:-self.TAG], packet[-self.TAG:]
        if not hmac.compare_digest(hmac.new(self.key, head + payload, hashlib.sha256).digest()[:self.TAG], tag):
            raise ValueError("bad packet tag")
        return struct.unpack(">Q", head)[0], payload


class AntiReplayWindow:
    """Sliding window over sequence numbers: rejects duplicates and packets older than the window."""

    def __init__(self, size=64):
        self.size, self.top, self.mask = size, -1, 0

    def accept(self, seq: int) -> bool:
        if seq > self.top:
            shift = seq - self.top
            self.mask = ((self.mask << shift) | 1) & ((1 << self.size) - 1) if self.top >= 0 else 1
            self.top = seq
            return True
        off = self.top - seq
        if off >= self.size or (self.mask >> off) & 1:
            return False
        self.mask |= 1 << off
        return True


class RateLimiter:
    def __init__(self, rate_per_s, burst, clock=time.monotonic):
        self.rate, self.burst, self.clock, self.tokens, self.last = rate_per_s, burst, clock, burst, clock()

    def allow(self, cost=1.0) -> bool:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


def lag_switch_score(arrivals_ms, positions, stall_ms=300, jump=5.0):
    """Fraction of long stalls followed by a suspicious position jump (catch-up teleport).
    Compare with the player's own history: genuine bad connections degrade gradually."""
    arrivals_ms, positions = np.asarray(arrivals_ms, float), np.asarray(positions, float)
    stalls = np.where(np.diff(arrivals_ms) > stall_ms)[0]
    if not len(stalls):
        return 0.0
    return sum(np.linalg.norm(positions[i + 1] - positions[i]) > jump for i in stalls) / len(stalls)
