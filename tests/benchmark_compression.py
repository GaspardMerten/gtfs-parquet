"""Benchmark: Parquet (zstd) size vs original GTFS zip size."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import httpx

from gtfs_parquet import parse_gtfs_zip, write_parquet, read_parquet, write_gtfs

FEEDS = {
    "STIB": {
        "url": "https://gtfs.irail.be/mivb/gtfs/gtfs-mivb-2026-03-31.zip",
        "path": Path("/tmp/stib_gtfs.zip"),
    },
    "TEC": {
        "url": "https://gtfs.irail.be/tec/tec-gtfs.zip",
        "path": Path("/tmp/tec_gtfs.zip"),
    },
    "De Lijn": {
        "url": "https://gtfs.irail.be/de-lijn/de_lijn-gtfs.zip",
        "path": Path("/tmp/delijn_gtfs.zip"),
    },
}


def download(name: str, url: str, path: Path) -> None:
    if not path.exists():
        print(f"  Downloading {name}...")
        r = httpx.get(url, follow_redirects=True, timeout=600)
        path.write_bytes(r.content)


def benchmark_feed(name: str, zip_path: Path) -> dict:
    zip_size = zip_path.stat().st_size
    feed = parse_gtfs_zip(str(zip_path))

    with tempfile.TemporaryDirectory() as tmp:
        pq_dir = Path(tmp) / "parquet"
        t0 = time.perf_counter()
        write_parquet(feed, pq_dir)
        pq_write_time = time.perf_counter() - t0

        pq_files = sorted(pq_dir.glob("*.parquet"))
        pq_sizes = {f.stem: f.stat().st_size for f in pq_files}
        pq_total = sum(pq_sizes.values())

        t0 = time.perf_counter()
        _ = read_parquet(pq_dir)
        pq_read_time = time.perf_counter() - t0

        rt_path = Path(tmp) / "roundtrip.zip"
        t0 = time.perf_counter()
        write_gtfs(feed, rt_path)
        gtfs_write_time = time.perf_counter() - t0
        rt_size = rt_path.stat().st_size

    return {
        "name": name,
        "zip_size": zip_size,
        "pq_total": pq_total,
        "rt_size": rt_size,
        "pq_sizes": pq_sizes,
        "pq_write_time": pq_write_time,
        "pq_read_time": pq_read_time,
        "gtfs_write_time": gtfs_write_time,
    }


# ---------------------------------------------------------------------------
# Download all feeds
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f" Downloading feeds")
print(f"{'='*70}")
for name, info in FEEDS.items():
    download(name, info["url"], info["path"])
    sz = info["path"].stat().st_size
    print(f"  {name:<12} {sz/1e6:>8.1f} MB  ({info['path']})")

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
results = []
for name, info in FEEDS.items():
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")
    r = benchmark_feed(name, info["path"])
    results.append(r)

    ratio = r["pq_total"] / r["zip_size"]
    print(f"\n  GTFS zip:      {r['zip_size']:>12,} bytes  ({r['zip_size']/1e6:.2f} MB)")
    print(f"  Parquet:       {r['pq_total']:>12,} bytes  ({r['pq_total']/1e6:.2f} MB)")
    print(f"  Ratio:         {ratio:.3f}x  ({(1-ratio)*100:+.1f}%)")

    print(f"\n  {'Timing':<30} {'seconds':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Write Parquet (zstd-9)':<30} {r['pq_write_time']:>10.3f}")
    print(f"  {'Read Parquet':<30} {r['pq_read_time']:>10.3f}")
    print(f"  {'Write GTFS zip':<30} {r['gtfs_write_time']:>10.3f}")

    print(f"\n  {'Table':<30} {'Parquet':>12} {'% of total':>12}")
    print(f"  {'-'*56}")
    for tbl in sorted(r["pq_sizes"], key=r["pq_sizes"].get, reverse=True):
        sz = r["pq_sizes"][tbl]
        pct = sz / r["pq_total"] * 100
        print(f"  {tbl:<30} {sz:>10,} B  {pct:>10.1f}%")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f" SUMMARY")
print(f"{'='*70}")
print(f"  {'Feed':<12} {'GTFS zip':>12} {'Parquet':>12} {'Ratio':>8} {'Savings':>10}  {'Write(s)':>9} {'Read(s)':>9}")
print(f"  {'-'*76}")
for r in results:
    ratio = r["pq_total"] / r["zip_size"]
    savings = (1 - ratio) * 100
    print(
        f"  {r['name']:<12} {r['zip_size']/1e6:>10.1f}MB"
        f" {r['pq_total']/1e6:>10.1f}MB"
        f" {ratio:>7.3f}x"
        f" {savings:>+9.1f}%"
        f"  {r['pq_write_time']:>9.3f} {r['pq_read_time']:>9.3f}"
    )

total_zip = sum(r["zip_size"] for r in results)
total_pq = sum(r["pq_total"] for r in results)
total_ratio = total_pq / total_zip
print(f"  {'-'*76}")
print(
    f"  {'TOTAL':<12} {total_zip/1e6:>10.1f}MB"
    f" {total_pq/1e6:>10.1f}MB"
    f" {total_ratio:>7.3f}x"
    f" {(1-total_ratio)*100:>+9.1f}%"
)
