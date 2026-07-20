# Full-51 two-part extraction runbook

## Accepted contract

The all-category Stage 0 extraction uses both versioned configurations:

- `configs/stage0/full51-part-a.yaml`: 26 categories;
- `configs/stage0/full51-part-b.yaml`: 25 categories.

Their category lists are disjoint and cover exactly all 51 CO3Dv2 categories.
Both use official sequence splits, seed `20260718`, 30/5/5 maximum
train/validation/test sequences per category, up to four disjoint six-frame
windows per sequence, and the same frozen CUT3R/checkpoint/preprocessing
contract. Part A and Part B are storage shards, not scientific splits.

Measured Debug storage was 492 MiB for 41 windows. Upper projections are about
49 GiB for Part A (4,160 windows) and 47 GiB for Part B (4,000 windows) using
the Debug token-grid mix, before invalid targets and unavailable sequences
reduce the totals. Full-51 aspect ratios can require more image tokens, so the
run computes an exact post-manifest projection before extraction. Keep at least
70 GiB free on the VM before starting either part and at least 120 GiB free on
the receiving computer before retaining both caches. Never place these caches
inside OneDrive or on the VM's temporary `/mnt` resource disk.

## Part A overnight command

Run this inside `tmux`. It stops immediately if any acquisition, manifest,
validation, extraction, or cache-integrity gate fails.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cut3r-stage0

export PROJECT_ROOT="$HOME/cut3r-stage0/repos/cut3r-semantic-probing"
export CUT3R_ROOT="$HOME/cut3r-stage0/repos/CUT3R"
export CO3D_ROOT="$HOME/cut3r-stage0/datasets/co3dv2"
export CUT3R_CHECKPOINT="$HOME/cut3r-stage0/checkpoints/cut3r_512_dpt_4_64.pth"
export CUT3R_ARTIFACT_ROOT="$HOME/cut3r-stage0/artifacts"
export CUT3R_CACHE_ROOT="$HOME/cut3r-stage0/cache"

cd "$PROJECT_ROOT"
git switch main
git pull --ff-only
python -m pip install -e ".[dev]"

CONFIG="configs/stage0/full51-part-a.yaml"
CACHE="$CUT3R_CACHE_ROOT/full51-part-a-v1"
LOG_ROOT="$HOME/cut3r-stage0/logs/full51-part-a-v1"
RESULT_ROOT="$CUT3R_ARTIFACT_ROOT/runs/full51-part-a-v1"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
if test -e "$CACHE" && ! test -d "$CACHE"; then
  echo "Cache path exists but is not a directory: $CACHE"
  exit 1
fi
if ! test -d "$CACHE"; then
  available_kib=$(df --output=avail -k "$HOME" | tail -n 1)
  test "$available_kib" -ge $((70 * 1024 * 1024))
fi

set -euo pipefail

python -m pytest -q \
  2>&1 | tee "$LOG_ROOT/tests.log"

python -m scripts.download_co3d_selective \
  --config "$CONFIG" \
  > "$RESULT_ROOT/download-result.json" \
  2> >(tee "$LOG_ROOT/download.log" >&2)

python -m scripts.build_manifests \
  --config "$CONFIG" \
  2>&1 | tee "$LOG_ROOT/manifest-build.log"

python -m scripts.validate_manifests \
  --manifest-dir "$CUT3R_ARTIFACT_ROOT/manifests/full51-part-a-v1" \
  --dataset-root "$CO3D_ROOT" \
  --inspect-files \
  2>&1 | tee "$LOG_ROOT/manifest-validation.log"

python -m scripts.project_cache_storage \
  --manifest-dir "$CUT3R_ARTIFACT_ROOT/manifests/full51-part-a-v1" \
  --filesystem-path "$CUT3R_CACHE_ROOT" \
  --reserve-gib 10 \
  2>&1 | tee "$RESULT_ROOT/cache-storage-projection.json"

python -m scripts.extract_features \
  --config "$CONFIG" \
  --cache-dir "$CACHE" \
  2>&1 | tee "$LOG_ROOT/extraction.log"

python -m scripts.validate_cache \
  --cache-dir "$CACHE" \
  2>&1 | tee "$RESULT_ROOT/cache-validation.json"

du -sh "$CACHE" | tee "$RESULT_ROOT/cache-size.txt"
df -h "$HOME" | tee "$RESULT_ROOT/disk-after.txt"
```

Detach with `Ctrl+B`, then `D`. Reattach with
`tmux attach -d -t full51-a`. The download may take hours because selective ZIP
access trades whole-archive storage for many bounded range requests. If a safe
rerun is necessary, the downloader verifies existing RGB/mask files and the
cache writer verifies its contract and indexed shards before resuming.

## Produce transfer hashes

After Part A completes, validate once more and generate hashes outside the
cache directory:

```bash
export CACHE="$CUT3R_CACHE_ROOT/full51-part-a-v1"
export TRANSFER_ROOT="$CUT3R_ARTIFACT_ROOT/transfers"
mkdir -p "$TRANSFER_ROOT"

python -m scripts.validate_cache --cache-dir "$CACHE"

(
  cd "$CACHE"
  find . -maxdepth 1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$TRANSFER_ROOT/full51-part-a-v1.sha256"

wc -l "$TRANSFER_ROOT/full51-part-a-v1.sha256"
du -sh "$CACHE"
```

## Copy to Windows

Use a non-OneDrive destination with at least 120 GiB free. WinSCP is preferred
for a resumable graphical transfer. With Windows OpenSSH, run from PowerShell
using the VM's current public IP:

```powershell
New-Item -ItemType Directory -Force D:\cut3r-full51
scp -r vmadmin@VM_PUBLIC_IP:/home/vmadmin/cut3r-stage0/cache/full51-part-a-v1 D:\cut3r-full51\
scp vmadmin@VM_PUBLIC_IP:/home/vmadmin/cut3r-stage0/artifacts/transfers/full51-part-a-v1.sha256 D:\cut3r-full51\
```

Verify every copied byte in PowerShell:

```powershell
$root = 'D:\cut3r-full51\full51-part-a-v1'
$manifest = 'D:\cut3r-full51\full51-part-a-v1.sha256'
$failures = @()

Get-Content -LiteralPath $manifest | ForEach-Object {
    $expected, $relative = $_ -split '\s+', 2
    $relative = $relative.Trim() -replace '^\.\\|^\./', ''
    $path = Join-Path $root $relative
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        $failures += $relative
    }
}

if ($failures.Count -ne 0) {
    throw "Hash verification failed: $($failures -join ', ')"
}
"FULL51_PART_A_LOCAL_COPY_OK"
```

## Delete only after verified transfer

Do not delete anything on the VM until the PowerShell verifier prints
`FULL51_PART_A_LOCAL_COPY_OK`. Then run on the VM:

```bash
CACHE="$CUT3R_CACHE_ROOT/full51-part-a-v1"
CACHE_REAL=$(realpath "$CACHE")
CACHE_ROOT_REAL=$(realpath "$CUT3R_CACHE_ROOT")

case "$CACHE_REAL" in
  "$CACHE_ROOT_REAL"/full51-part-a-v1) ;;
  *) echo "Refusing unexpected deletion target: $CACHE_REAL"; exit 1 ;;
esac

python -m scripts.validate_cache --cache-dir "$CACHE_REAL"
rm -rf -- "$CACHE_REAL"
test ! -e "$CACHE_REAL"
echo "FULL51_PART_A_VM_CACHE_REMOVED"
```

Keep the Part A manifests, logs, transfer hash list, download provenance, and
dataset files. They are small relative to the cache and required for audit.

## Part B

Repeat the same workflow with these substitutions:

```text
full51-part-a.yaml       -> full51-part-b.yaml
full51-part-a-v1        -> full51-part-b-v1
full51-part-a-v1.sha256 -> full51-part-b-v1.sha256
full51-a tmux session   -> full51-b tmux session
```

Do not merge the two cache directories by copying files together: shard names
overlap and each cache has a distinct extraction contract. Later Stage 1/2 code
must load the two verified cache roots as a logical union and preserve their
separate metadata/manifests.

## Publish for the team

After **both** local PowerShell checksum gates pass, publish the two cache roots,
checksum files, manifests, run records, and audit artifacts using the staging
and promotion procedure in the
[Stage 0 Full-51 cache handoff](../data/stage0-full51-cache-handoff.md). Part A
may upload while Part B transfers, but the shared folder must retain its
`stage0-full51-v1-staging` name and Part B must not appear in the canonical
cache path until its local bytes are verified.

Do not train directly from a Google Drive streaming mount. Each teammate copies
both immutable cache roots to local compute storage, verifies the published
SHA-256 lists, runs `scripts.validate_cache` on each root, and only then treats
the two indexes as a logical union.
