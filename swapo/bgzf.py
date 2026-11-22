"""Read a slice of a remote BGZF file without downloading all of it.

Population resources are too large to fetch for a measurement: the gnomAD v4.1
chr21 genomes sites VCF is 7.2 GB, and we need its coordinates, not its
annotations. BGZF — the block-gzip format every bgzipped VCF uses — makes a
slice cheap, because the file is a sequence of **independent gzip members**,
each at most 64 KiB uncompressed, with the member size recorded in its header.

So an HTTP range request that begins on a member boundary is itself a valid
gzip stream. The only work is finding a boundary, which this module does by
scanning for the BGZF magic and validating it against the size field.

The result is a contiguous genomic window at the file's true record density,
which is exactly what a density measurement needs. It is **not** a uniform
sample of the file, and nothing here should be used to estimate genome-wide
totals.

Reference: the BGZF specification in the SAM/BAM format spec, section 4.1.
"""

from __future__ import annotations

import gzip
import io
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Iterator

__all__ = ["find_block_boundary", "block_size", "fetch_range", "stream_lines",
           "RangeSlice"]

#: A BGZF member starts with the gzip magic, deflate method, and FEXTRA set.
BGZF_MAGIC = b"\x1f\x8b\x08\x04"


@dataclass
class RangeSlice:
    """What a slice actually covered, so a measurement can state its scope."""

    url: str
    requested_start: int
    boundary_start: int
    compressed_bytes: int
    uncompressed_bytes: int
    lines: int
    truncated_tail: bool


def _http_range(url: str, start: int, length: int, timeout: int = 600) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{start + length - 1}",
                      "User-Agent": "swapo/0.1 (measurement study; python-urllib)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed URLs
        return r.read()


def block_size(buf: bytes, off: int) -> int:
    """Total compressed length of the BGZF member at ``off``, or -1.

    BGZF records this in the BC subfield of the gzip extra field, so a member
    can be sliced exactly without decompressing anything first.
    """
    if off + 18 > len(buf) or not buf.startswith(BGZF_MAGIC, off):
        return -1
    xlen = struct.unpack_from("<H", buf, off + 10)[0]
    extra = buf[off + 12:off + 12 + xlen]
    j = 0
    while j + 4 <= len(extra):
        si1, si2, slen = extra[j], extra[j + 1], struct.unpack_from("<H", extra, j + 2)[0]
        if si1 == 66 and si2 == 67 and slen == 2:            # 'B','C'
            return struct.unpack_from("<H", extra, j + 4)[0] + 1
        j += 4 + slen
    return -1


def find_block_boundary(buf: bytes, from_offset: int = 0) -> int:
    """Offset of the first valid BGZF member at or after ``from_offset``.

    A candidate is accepted only when its declared block size lands on another
    BGZF magic, or on the end of the buffer. Checking the successor rules out
    the magic appearing by chance inside compressed data.
    """
    i = from_offset
    n = len(buf)
    while True:
        i = buf.find(BGZF_MAGIC, i)
        if i < 0 or i + 18 > n:
            return -1
        bsize = block_size(buf, i)
        if bsize > 0:
            nxt = i + bsize
            # Validate against the successor so the magic cannot match by
            # chance inside compressed data.
            if nxt >= n or buf.startswith(BGZF_MAGIC, nxt):
                return i
        i += 1


def fetch_range(url: str, start: int, length: int) -> tuple[bytes, int]:
    """Fetch ``length`` bytes from ``start`` and trim to the first block boundary."""
    raw = _http_range(url, start, length)
    off = find_block_boundary(raw)
    if off < 0:
        raise ValueError("no BGZF block boundary found; is this a bgzipped file?")
    return raw[off:], start + off


def stream_lines(url: str, start: int, length: int) -> Iterator[tuple[str, RangeSlice]]:
    """Yield decompressed text lines from a byte range of a remote BGZF file.

    The final member is usually cut off mid-way, so the last partial line is
    discarded rather than parsed. The accompanying :class:`RangeSlice` records
    what was really covered.
    """
    data, boundary = fetch_range(url, start, length)
    info = RangeSlice(url=url, requested_start=start, boundary_start=boundary,
                      compressed_bytes=len(data), uncompressed_bytes=0,
                      lines=0, truncated_tail=False)
    pos = 0
    carry = b""
    # One member at a time, yielding as we go. Accumulating the whole range
    # first would be fatal: a few hundred megabytes of bgzipped VCF expands to
    # several gigabytes of text, and the caller only wants a handful of columns
    # from each line.
    #
    # Handing the whole range to GzipFile does not work either: it fails on the
    # cut-off final member and discards everything decoded before it.
    while pos < len(data):
        bsize = block_size(data, pos)
        if bsize <= 0 or pos + bsize > len(data):
            info.truncated_tail = True
            break
        try:
            chunk = gzip.decompress(data[pos:pos + bsize])
        except (OSError, EOFError, zlib.error):
            info.truncated_tail = True
            break
        info.uncompressed_bytes += len(chunk)
        pos += bsize
        parts = (carry + chunk).split(b"\n")
        carry = parts.pop()               # may be a partial line; hold it over
        for ln in parts:
            info.lines += 1
            yield ln.decode("utf-8", "replace"), info
    # The last line is only complete if the range happened to end on a newline.
    if carry and not info.truncated_tail:
        info.lines += 1
        yield carry.decode("utf-8", "replace"), info
