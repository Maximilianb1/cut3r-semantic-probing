# Google Drive ↔ Technion VM data transfer

This runbook covers moving large artifacts (probe caches, checkpoints, logs)
between a teammate's Google Drive folder and the Technion GPU VM without
staging them on a personal laptop first. The laptop hop is what
`rclone` exists to remove.

The [`TECHNION_VM_RUNBOOK.md`](TECHNION_VM_RUNBOOK.md) covers preflight and
Stage 0 extraction on the VM; this document only covers file transfer.

## When to use this path

Prefer Drive → VM directly (this runbook) when the artifact is:

- Larger than ~1 GB, or
- Named by a shared Drive folder id, or
- Meant to land under `${CUT3R_CACHE_ROOT}` / `${CUT3R_ARTIFACT_ROOT}` for a
  training or eval run on the VM.

Only use laptop-mediated transfer (browser download → `scp`) when the artifact
is small (< 100 MB), one-off, and would not benefit from Drive being kept as
the canonical copy.

## The invariant

**Run `rclone` on the VM, not on the laptop.** Reasons:

- Drive-to-VM stays inside cloud backbones and typically saturates the VM's
  network link; Drive → laptop → VM is limited by the laptop's uplink and
  writes the artifact to a home directory that should not hold research data.
- The `gdrive` remote is configured *once*, on the VM, using a browser
  redirect completed on the laptop. After that any teammate with SSH access
  reuses the same configured remote.
- Windows PowerShell reports `rclone lsd gdrive: …` with exit code 1 when the
  remote is not configured locally, which is the standard failure mode when
  someone runs the command in the wrong shell. If you see that on your
  laptop, you are in the wrong terminal — SSH into the VM instead.

## One-time setup on the VM

Perform once per VM lifetime, and re-run only if `~/.config/rclone/rclone.conf`
is wiped.

```bash
# 1. Install rclone (v1.75+ observed working; v1.60+ should suffice).
curl -fsSL https://rclone.org/install.sh | sudo bash
rclone version

# 2. Configure the gdrive remote. Choose:
#      n) new remote           name: gdrive
#      Storage:                drive
#      client_id / secret:     leave blank (uses rclone defaults)
#      scope:                  1 (full)
#      root_folder_id:         leave blank
#      service_account_file:   leave blank
#      Edit advanced config:   n
#      Use auto config:        n   # VM has no browser
#      Then paste the auth URL rclone prints into a laptop browser,
#      approve access, and paste the returned verification code back
#      into the VM prompt.
#      Configure this as a Shared Drive:  n
rclone config

# 3. Smoke test — this must succeed on the VM, not on the laptop.
rclone lsd gdrive: | head
```

The remote is stored at `~/.config/rclone/rclone.conf`; the OAuth token in
that file refreshes automatically and does not need to be committed anywhere.
Do not copy that file off the VM.

## Common tasks

The examples below use `${DRIVE_FOLDER_ID}` as a placeholder — replace it with
the actual folder id (the string after `/folders/` in the Drive URL, or the
value the teammate pasted into chat).

### Inspect a Drive folder before pulling

```bash
DRIVE_FOLDER_ID=1wyEljQmtTxmC4mfqcc8T0hXYmaI_yhxr    # example only

rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" lsd    gdrive:
rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" size   gdrive:
rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" lsjson gdrive: --recursive \
  | jq 'length, [.[].Name] | .[:5]'
```

`lsd` shows top-level subfolders, `size` reports total bytes and file count,
`lsjson` is useful when you need to check a specific file exists before
committing to the transfer.

### Fetch a probe cache into `${CUT3R_CACHE_ROOT}/probe/<backbone>/`

List before copying — the subfolder under `cache/` does not reliably match
the bare backbone name. Observed real examples: CUT3R-random matched
(`cache/cut3r-random`), but CUT3R-trained was `cache/cut3r-trained-target-labeled`
and DINOv2 was `cache/dinov2-vitb14`, not `cache/cut3r-trained` or `cache/dinov2`.
A copy from an assumed path that doesn't exist silently transfers nothing.

```bash
DRIVE_FOLDER_ID=1wyEljQmtTxmC4mfqcc8T0hXYmaI_yhxr    # example only
BACKBONE=cut3r-random
DEST="${CUT3R_CACHE_ROOT}/probe/${BACKBONE}"

rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" lsd gdrive:cache   # confirm the real subfolder name first

mkdir -p "${DEST}"
rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" \
  copy gdrive:cache/<real-subfolder-name> "${DEST}" \
  --progress --transfers 8 --checkers 16 \
  --exclude "*.tmp" --exclude ".ipynb_checkpoints/**"
```

If the Drive folder has an unnested layout (files directly under the folder
id, no `cache/<backbone>/` prefix), drop the `cache/${BACKBONE}` path segment
from the source and copy `gdrive:` directly into `${DEST}`.

Stash the collaborator's extraction config alongside so we retain provenance
for who produced the cache and with what settings:

```bash
mkdir -p "${CUT3R_ARTIFACT_ROOT}/from-drive/${BACKBONE}"
rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" \
  copy gdrive:config "${CUT3R_ARTIFACT_ROOT}/from-drive/${BACKBONE}" \
  --progress
```

### Verify the transfer against the on-disk manifests

Trust the cache only after both checks pass:

```bash
# Manifest anchors must match summary.json inside the copied cache.
python -c "
import json, pathlib
root = pathlib.Path('${DEST}')
meta = json.loads((root / 'metadata.json').read_text())
summ = json.loads((root / 'summary.json').read_text())
for k in ('frames', 'sequences', 'windows'):
    assert meta['manifest_sha256'][k] == summ['manifest_sha256'][k], k
print('manifest sha256:', 'ok')
print('probe_cache_schema_version:', meta.get('probe_cache_schema_version'))
print('layout:', meta.get('layout'))
"

# Full-tree hash check (slower; run if a run misbehaves later).
rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" \
  check gdrive:cache/${BACKBONE} "${DEST}" --one-way
```

### Push VM results back to Drive

For sharing artifacts (`segmentation-*/`, `metrics.json`, `train.log`) with
teammates. Never push probe caches themselves — Drive is a copy of the last
labelled build, not a working directory. Use a dedicated results folder id.

```bash
DRIVE_RESULTS_ID=<paste from the results folder URL>
SRC="${CUT3R_ARTIFACT_ROOT}/segmentation/segmentation-cut3r-random"

rclone --drive-root-folder-id="${DRIVE_RESULTS_ID}" \
  copy "${SRC}" gdrive:$(basename "${SRC}") \
  --progress --exclude "*.pt.tmp" --exclude "*/checkpoints/**"
```

Prefer `copy` (idempotent) over `sync` (deletes on the destination) unless
you specifically want the destination cleaned to match local.

## Space and quota checks

Before starting a large transfer:

```bash
df -h ${CUT3R_CACHE_ROOT%/*}          # free space on the target disk
rclone --drive-root-folder-id="${DRIVE_FOLDER_ID}" size gdrive:
```

Keep at least 20 % headroom on the target filesystem. The Technion VM's OS
disk is ~124 GB total; do not fill it past ~100 GB.

## Troubleshooting

- **`rclone lsd gdrive: …` returns exit 1 on the laptop.** The `gdrive` remote
  is only configured on the VM. SSH in and re-run there.
- **`Failed to configure token: could not refresh token`.** OAuth token has
  expired or been revoked. Re-run `rclone config`, choose `Edit existing`,
  then `Reconfigure` on the `gdrive` remote, and complete the browser flow
  again.
- **`googleapi: Error 403: userRateLimitExceeded`.** Retry with
  `--tpslimit 8 --tpslimit-burst 8` and reduce `--transfers` to `4`.
- **Copy completed but `metadata.json` schema key is
  `cache_schema_version: stage0-target-cache-v1`.** The folder is a Stage 0
  target feature cache, not a probe cache; the trainer will not consume it.
  See ADR 0002 for the schema distinction.
- **Free space runs out mid-copy.** `rclone` leaves partial files behind. Run
  `rclone --drive-root-folder-id=… check gdrive:… "${DEST}" --one-way` to
  identify missing files after freeing space, then resume with a second
  `rclone copy` (idempotent).

## Related

- VM preflight and Stage 0 order: [`TECHNION_VM_RUNBOOK.md`](TECHNION_VM_RUNBOOK.md).
- Cache schemas (`probe-features-v2` vs `stage0-target-cache-v1`):
  [ADR 0002](../decisions/0002-co3dv2-stage0-data-protocol.md).
- CUT3R checkpoint / commit trust anchors:
  [ADR 0003](../decisions/0003-cut3r-trajectory-and-cache-contract.md).
