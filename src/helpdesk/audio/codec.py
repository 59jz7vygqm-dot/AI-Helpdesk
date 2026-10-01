"""G.711 (PCMU/PCMA) codecs and resampling.

The stdlib ``audioop`` module was removed in Python 3.13, so the companding
tables are built here with numpy.  Both directions are pure table lookups, which
makes a 20 ms frame cost a few microseconds -- the telephony leg must never be
the thing that adds latency.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

_SIGN_BIT = 0x80
_QUANT_MASK = 0x0F
_SEG_MASK = 0x70
_SEG_SHIFT = 4
_BIAS = 0x84


def _build_ulaw_decode() -> np.ndarray:
    out = np.empty(256, dtype=np.int16)
    for byte in range(256):
        u = ~byte & 0xFF
        t = ((u & _QUANT_MASK) << 3) + _BIAS
        t <<= (u & _SEG_MASK) >> _SEG_SHIFT
        out[byte] = (_BIAS - t) if (u & _SIGN_BIT) else (t - _BIAS)
    return out


def _build_alaw_decode() -> np.ndarray:
    out = np.empty(256, dtype=np.int16)
    for byte in range(256):
        a = byte ^ 0x55
        t = (a & _QUANT_MASK) << 4
        seg = (a & _SEG_MASK) >> _SEG_SHIFT
        if seg == 0:
            t += 8
        elif seg == 1:
            t += 0x108
        else:
            t = (t + 0x108) << (seg - 1)
        out[byte] = t if (a & _SIGN_BIT) else -t
    return out


def _build_encode_table(decode: np.ndarray) -> np.ndarray:
    """Nearest-codeword table for every int16 input.

    Indexed with ``sample + 32768``, so encoding a frame is one gather.  Using
    minimum distance rather than the reference bit-twiddling keeps the quantiser
    optimal and removes any chance of a sign/segment bug.
    """
    order = np.argsort(decode, kind="stable")
    levels = decode[order].astype(np.int32)
    samples = np.arange(-32768, 32768, dtype=np.int32)
    idx = np.searchsorted(levels, samples)
    idx = np.clip(idx, 1, len(levels) - 1)
    lower = levels[idx - 1]
    upper = levels[idx]
    pick_lower = (samples - lower) <= (upper - samples)
    chosen = np.where(pick_lower, idx - 1, idx)
    return order[chosen].astype(np.uint8)


ULAW_DECODE = _build_ulaw_decode()
ALAW_DECODE = _build_alaw_decode()
ULAW_ENCODE = _build_encode_table(ULAW_DECODE)
ALAW_ENCODE = _build_encode_table(ALAW_DECODE)

# G.711 has two codes for zero; the standards use positive zero for idle fill
# (0xFF for u-law, 0xD5 for A-law) and some gateways squelch the negative one.
ULAW_SILENCE = 0xFF
ALAW_SILENCE = 0xD5


def ulaw_decode(payload: bytes) -> np.ndarray:
    return ULAW_DECODE[np.frombuffer(payload, dtype=np.uint8)]


def alaw_decode(payload: bytes) -> np.ndarray:
    return ALAW_DECODE[np.frombuffer(payload, dtype=np.uint8)]


def ulaw_encode(pcm: np.ndarray) -> bytes:
    return ULAW_ENCODE[np.asarray(pcm, dtype=np.int16).astype(np.int32) + 32768].tobytes()


def alaw_encode(pcm: np.ndarray) -> bytes:
    return ALAW_ENCODE[np.asarray(pcm, dtype=np.int16).astype(np.int32) + 32768].tobytes()


def decode(payload: bytes, encoding: str) -> np.ndarray:
    if encoding == "PCMU":
        return ulaw_decode(payload)
    if encoding == "PCMA":
        return alaw_decode(payload)
    raise ValueError(f"unsupported payload encoding: {encoding}")


def encode(pcm: np.ndarray, encoding: str) -> bytes:
    if encoding == "PCMU":
        return ulaw_encode(pcm)
    if encoding == "PCMA":
        return alaw_encode(pcm)
    raise ValueError(f"unsupported payload encoding: {encoding}")


def silence_byte(encoding: str) -> int:
    return ULAW_SILENCE if encoding == "PCMU" else ALAW_SILENCE


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample, returning int16.

    Used for 8 k -> 16 k on the way into the recogniser and for TTS output
    (22.05 k / 24 k) on the way back out to the call.
    """
    if src_rate == dst_rate:
        return np.asarray(pcm, dtype=np.int16)
    if pcm.size == 0:
        return np.zeros(0, dtype=np.int16)
    g = np.gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    out = resample_poly(np.asarray(pcm, dtype=np.float32), up, down)
    return np.clip(out, -32768, 32767).astype(np.int16)


def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return np.asarray(pcm, dtype=np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    scaled = np.asarray(audio, dtype=np.float32) * 32767.0
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def rms_dbfs(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(np.asarray(pcm, dtype=np.float64)))))
    if rms < 1e-6:
        return -120.0
    return 20.0 * float(np.log10(rms / 32768.0))
