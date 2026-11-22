"""Block boundary detection for BGZF, tested without touching the network.

A bgzipped VCF is a sequence of self-contained gzip members, each declaring its
own compressed length in a ``BC`` extra subfield. Finding a member boundary
inside an arbitrary byte range is what lets us measure a 7 GB remote callset by
fetching a few hundred megabytes of it, so it is worth testing against bytes we
build ourselves rather than whatever a server happens to return.
"""

from __future__ import annotations

import gzip
import struct
import zlib

from swapo.bgzf import BGZF_MAGIC, block_size, find_block_boundary


def make_member(payload: bytes) -> bytes:
    """Build one valid BGZF member, laid out as bgzip writes it.

    The header is 18 bytes: the magic already covers ID1, ID2, CM and FLG, so it
    is followed by MTIME (4), XFL, OS, XLEN (2) and the 6-byte BC extra field.
    The trailer is CRC32 and ISIZE.
    """
    body = zlib.compress(payload, 6)[2:-4]                   # raw deflate
    bsize = 18 + len(body) + 8
    extra = struct.pack("<BBHH", 0x42, 0x43, 2, bsize - 1)   # 'B','C', len 2, BSIZE-1
    header = (BGZF_MAGIC
              + b"\x00\x00\x00\x00"                          # MTIME
              + b"\x00"                                      # XFL
              + b"\xff"                                      # OS = unknown
              + struct.pack("<H", len(extra))                # XLEN
              + extra)
    trailer = struct.pack("<II", zlib.crc32(payload) & 0xFFFFFFFF, len(payload))
    return header + body + trailer


def test_the_fixture_is_a_real_gzip_member() -> None:
    """If the fixture were malformed, the rest of this file would prove nothing."""
    payload = b"chr21\t1\t.\tA\tG\n" * 40
    assert gzip.decompress(make_member(payload)) == payload


def test_block_size_matches_the_member_it_describes() -> None:
    m = make_member(b"chr21\t1\t.\tA\tG\n" * 50)
    assert block_size(m, 0) == len(m)


def test_boundary_found_at_the_start_of_a_clean_stream() -> None:
    stream = make_member(b"a" * 400) + make_member(b"b" * 400)
    assert find_block_boundary(stream) == 0


def test_boundary_found_after_a_truncated_leading_member() -> None:
    """The realistic case: a byte range begins in the middle of a member."""
    # Payloads that do not compress to almost nothing, so the members are long
    # enough for a cut to land inside the first one.
    a, b = bytes(range(256)) * 8, bytes(range(255, -1, -1)) * 8
    first, second = make_member(a), make_member(b)
    cut_at = len(first) // 2
    cut = (first + second)[cut_at:]
    off = find_block_boundary(cut)
    assert off == len(first) - cut_at
    assert cut[off:off + 4] == BGZF_MAGIC
    assert block_size(cut, off) == len(second)
    assert gzip.decompress(cut[off:off + len(second)]) == b


def test_stray_magic_inside_data_is_not_mistaken_for_a_boundary() -> None:
    """The magic can occur by chance in compressed bytes; the size field rules it out."""
    stream = make_member(BGZF_MAGIC * 40 + b"not a real header")
    assert find_block_boundary(stream) == 0
    assert block_size(stream, 0) == len(stream)


def test_no_boundary_in_arbitrary_bytes() -> None:
    assert find_block_boundary(b"\x00" * 300) == -1
