# Stage 0 Full-51 cache handoff

## Purpose and release status

This document defines how the team publishes, downloads, verifies, and consumes
the frozen CUT3R Stage 0 caches. The release name is `stage0-full51-v1`.

Current status on 2026-07-20:

| Part | Categories | Windows | Shards | Tensors | Size | Status |
|---|---:|---:|---:|---:|---:|---|
| `full51-part-a-v1` | 26 | 3,667 | 115 | 7,334 | about 43 GiB | Locally SHA-verified and GPU-audited |
| `full51-part-b-v1` | 25 | 3,458 | 109 | 6,916 | 40.285 GiB | Locally SHA-verified; VM cache validation passed |
| Combined logical dataset | 51 | 7,125 | 224 | 14,250 | 83.053 GiB | Both parts verified and ready for staged Drive publication |

Part A and Part B are storage/category shards, **not** train/test splits. The
official CO3D sequence split recorded in the manifests remains the scientific
train/validation/test assignment. Never merge the two cache directories:
their Safetensors shard names overlap.

## Canonical team folder

The project owner chose a private Google **My Drive** folder shared with the
team. Max owns the folder and its storage usage; teammates receive Viewer
access. This is simpler than requesting a Workspace Shared drive, but ownership
does not belong to the team automatically. Keep the independently verified
local copies and require at least one teammate to retain a second verified copy.

Canonical restricted folder:
[Stage 0 Full-51 Google Drive](https://drive.google.com/drive/folders/1UttTnkxRlcz3H3K-Puv1VhVfcjrAZTzN).

During upload, use the temporary name `stage0-full51-v1-staging`; rename it to
`stage0-full51-v1` only after both parts and all provenance files are present.

```text
cut3r-semantic-probing-data/
└── stage0-full51-v1-staging/
    ├── caches/
    │   ├── full51-part-a-v1/
    │   │   ├── metadata.json
    │   │   ├── index.parquet
    │   │   └── shard-00000.safetensors ... shard-00114.safetensors
    │   └── full51-part-b-v1/
    │       ├── metadata.json
    │       ├── index.parquet
    │       └── shard-00000.safetensors ... shard-00108.safetensors
    ├── checksums/
    │   ├── full51-part-a-v1.sha256
    │   └── full51-part-b-v1.sha256
    ├── manifests/
    │   ├── full51-part-a-v1/
    │   └── full51-part-b-v1/
    ├── run-records/
    │   ├── full51-part-a-v1/
    │   └── full51-part-b-v1/
    └── audits/
        └── part-a-window-000/
```

The `manifests` directories must contain `sequences.parquet`, `frames.parquet`,
`windows.parquet`, and `summary.json`. The `run-records` directories should
contain download provenance, storage projection, cache validation, cache size,
and extraction logs. Do not upload the CUT3R checkpoint or raw CO3D data to this
team folder. Teammates obtain those from their official sources and verify the
checkpoint hash recorded in `metadata.json`.

The private folder URL must be distributed in the team's private channel or
private repository settings; do not place an access token, OAuth credential,
or publicly accessible write link in Git.

### Exact files to upload

The final cache payload is exactly:

```text
caches/full51-part-a-v1/
  metadata.json
  index.parquet
  shard-00000.safetensors ... shard-00114.safetensors  # 117 files total
caches/full51-part-b-v1/
  metadata.json
  index.parquet
  shard-00000.safetensors ... shard-00108.safetensors  # 111 files total
checksums/full51-part-a-v1.sha256
checksums/full51-part-b-v1.sha256
```

Do not omit either `metadata.json` or `index.parquet`; the Safetensors files are
not usable or auditable without them. Do not rename any cache file.

Also upload these small provenance directories from the VM:

```text
manifests/full51-part-a-v1/{sequences,frames,windows}.parquet
manifests/full51-part-a-v1/summary.json
manifests/full51-part-b-v1/{sequences,frames,windows}.parquet
manifests/full51-part-b-v1/summary.json
run-records/full51-part-a-v1/
run-records/full51-part-b-v1/
logs/full51-part-a-v1/
logs/full51-part-b-v1/
audits/part-a-window-000/
```

The cache directories and two checksum lists are mandatory. The manifests are
mandatory for later scientific split/category/target lookup. Run records, logs,
and the audit are mandatory provenance for the final team release but are not
read on every training batch.

## Upload from the verified Windows copy

Google Drive for desktop is preferred over a browser upload because it retries
interrupted transfers. Configure My Drive to **Stream files**; do not use
Mirror files and do not mark these caches "Available offline". While another
large transfer is active, Drive for desktop can limit its upload rate under
`Settings -> Preferences -> Advanced settings`.

After Drive for desktop mounts My Drive, set the actual drive/folder in
PowerShell. This example uses the usual `G:` mount:

```powershell
$SourceRoot = 'C:\cut3r-full51'
$DriveRoot = 'G:\My Drive\cut3r-semantic-probing-data\stage0-full51-v1-staging'

New-Item -ItemType Directory -Force "$DriveRoot\caches" | Out-Null
New-Item -ItemType Directory -Force "$DriveRoot\checksums" | Out-Null

robocopy `
  "$SourceRoot\full51-part-a-v1" `
  "$DriveRoot\caches\full51-part-a-v1" `
  /E /Z /R:5 /W:10 /MT:2 /COPY:DAT /DCOPY:DAT `
  /LOG:"$SourceRoot\full51-part-a-v1-drive-upload.log"

if ($LASTEXITCODE -ge 8) {
    throw "Part A Drive copy failed with robocopy exit code $LASTEXITCODE"
}

Copy-Item -LiteralPath "$SourceRoot\full51-part-a-v1.sha256" `
  -Destination "$DriveRoot\checksums\full51-part-a-v1.sha256" -Force
```

Robocopy exit codes `0` through `7` are nonfatal; `8` or higher is failure.
Rerunning the same command is safe and skips files that already match its copy
criteria. Keep the verified source directory until another team member has
downloaded and SHA-verified the Drive copy. A Drive "sync complete" indicator
does not replace end-to-end hash verification.

Part B printed `FULL51_PART_B_LOCAL_COPY_OK` on 2026-07-20. Upload it with the
same command after replacing `part-a` with `part-b`. Do not upload any future
partially transferred cache into a canonical cache path.

Download the small VM provenance before promoting the folder. From PowerShell:

```powershell
$Extra = 'C:\cut3r-full51\stage0-full51-v1-extra'
New-Item -ItemType Directory -Force `
  "$Extra\manifests", "$Extra\run-records", "$Extra\logs", "$Extra\audits"
sftp vmadmin@VM_PUBLIC_IP
```

At the `sftp>` prompt:

```text
lcd C:/cut3r-full51/stage0-full51-v1-extra/manifests
get -R /home/vmadmin/cut3r-stage0/artifacts/manifests/full51-part-a-v1
get -R /home/vmadmin/cut3r-stage0/artifacts/manifests/full51-part-b-v1
lcd C:/cut3r-full51/stage0-full51-v1-extra/run-records
get -R /home/vmadmin/cut3r-stage0/artifacts/runs/full51-part-a-v1
get -R /home/vmadmin/cut3r-stage0/artifacts/runs/full51-part-b-v1
lcd C:/cut3r-full51/stage0-full51-v1-extra/logs
get -R /home/vmadmin/cut3r-stage0/logs/full51-part-a-v1
get -R /home/vmadmin/cut3r-stage0/logs/full51-part-b-v1
lcd C:/cut3r-full51/stage0-full51-v1-extra/audits
get -R /home/vmadmin/cut3r-stage0/artifacts/audits/part-a-window-000
bye
```

Copy the four resulting local directories into the matching Drive paths after
both large caches finish uploading.

## Download for a teammate

The owner shares the completed `stage0-full51-v1` folder with each teammate as
Viewer. A teammate opens the link in Drive on the web and uses
`Organize -> Add shortcut to My Drive`, then installs Google Drive for desktop
and uses streaming. Copy both cache directories to a local non-synced SSD with
at least 100 GiB free. Do not train directly against the streamed `G:` paths
because random Safetensors access would repeatedly depend on the network/cache
layer.

```powershell
$DriveRoot = 'G:\My Drive\stage0-full51-v1'
$LocalRoot = 'D:\cut3r-stage0-full51-v1'

New-Item -ItemType Directory -Force "$LocalRoot\caches" | Out-Null
New-Item -ItemType Directory -Force "$LocalRoot\checksums" | Out-Null

foreach ($Part in 'full51-part-a-v1', 'full51-part-b-v1') {
    robocopy "$DriveRoot\caches\$Part" "$LocalRoot\caches\$Part" `
      /E /Z /R:5 /W:10 /MT:2 /COPY:DAT /DCOPY:DAT
    if ($LASTEXITCODE -ge 8) {
        throw "$Part download failed with robocopy exit code $LASTEXITCODE"
    }
    Copy-Item "$DriveRoot\checksums\$Part.sha256" `
      "$LocalRoot\checksums\$Part.sha256" -Force
}
```

## Verify every downloaded file

Run this in PowerShell for each part. Verification is successful only when it
prints the corresponding `*_LOCAL_COPY_OK` line.

```powershell
$LocalRoot = 'D:\cut3r-stage0-full51-v1'

foreach ($Part in 'full51-part-a-v1', 'full51-part-b-v1') {
    $Cache = "$LocalRoot\caches\$Part"
    $Manifest = "$LocalRoot\checksums\$Part.sha256"
    $Failures = @()

    Get-Content -LiteralPath $Manifest | ForEach-Object {
        $Expected, $Relative = $_ -split '\s+', 2
        $Relative = $Relative.Trim() -replace '^\.\\|^\./', ''
        $Path = Join-Path $Cache $Relative
        if (-not (Test-Path -LiteralPath $Path)) {
            $Failures += "MISSING: $Relative"
        }
        else {
            $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
            if ($Actual -ne $Expected) {
                $Failures += "HASH MISMATCH: $Relative"
            }
        }
    }

    if ($Failures.Count -ne 0) {
        $Failures
        throw "$Part failed verification"
    }
    "${Part}_LOCAL_COPY_OK"
}
```

Then validate cache-level structure from the project environment:

```bash
python -m pip install -e ".[dev]"
python -m scripts.validate_cache --cache-dir /data/stage0-full51-v1/caches/full51-part-a-v1
python -m scripts.validate_cache --cache-dir /data/stage0-full51-v1/caches/full51-part-b-v1
```

Expected immutable identities:

| Part | Files | Index SHA-256 | Metadata SHA-256 |
|---|---:|---|---|
| Part A | 117 | `9430ea48bfe5be6b3dc6d854027a581380031e1cc2b08ea72f7e0cab8e26535c` | `e0717ffb35908946a94fdbbe27566eb6cbfda48ea8086176c665c9fa87893d5e` |
| Part B | 111 | `d17beff12ec10a7522f57d073d82acec0fc2b8f5b840a8c4a136aa0c5162ee7a` | `b50adbb8e4194538f3dfe3b79b68c73287dcb90a22428c1b1eed6bb286a1701e` |

## Use the two caches

Treat the roots as a logical union while retaining their identities:

Do not confuse the two **cache roots** with the two **representation fields**.
Part A and Part B divide the 51 categories for storage; every window in both
roots contains both `image_tokens` and `state_tokens`.

```python
from pathlib import Path

from src.common.tables import read_parquet
from src.embeddings.cache import load_trajectory, verify_cache

cache_roots = [
    Path("/data/stage0-full51-v1/caches/full51-part-a-v1"),
    Path("/data/stage0-full51-v1/caches/full51-part-b-v1"),
]

for cache_root in cache_roots:
    print(verify_cache(cache_root))
    rows = read_parquet(cache_root / "index.parquet")
    for row in rows:
        trajectory = load_trajectory(cache_root, row["window_id"])
        # [6, 1, spatial_tokens, 768]
        image_tokens = trajectory.image_tokens
        # [6, 1, 768 persistent-state tokens, 768]
        state_tokens = trajectory.state_tokens
        frame_ids = trajectory.frame_ids
        grid_height, grid_width = trajectory.token_grid
```

Important semantics:

- Timestep `t` in both tensors corresponds to `frame_ids[t]`.
- The planned supervised target is timestep `5` (the sixth frame).
- `image_tokens[t, 0]` is spatial and belongs to the current frame after
  interaction with the preceding recurrent state. Its token count is
  `grid_height * grid_width` and therefore may vary with transformed aspect
  ratio.
- `state_tokens[t, 0]` is the fixed-size persistent memory after the current
  frame is committed. It is not pixel-aligned and must not be reshaped as a
  segmentation grid.
- For segmentation, reshape `image_tokens[5, 0]` from
  `[grid_height * grid_width, 768]` to `[grid_height, grid_width, 768]`.
- For image classification, compare spatially pooled
  `image_tokens[5, 0]` with token-pooled `state_tokens[5, 0]`.
- The same physical frame may appear with different features in another window
  because its representation depends on the preceding recurrent context.
- Category, official split, RGB path, mask path, and transform metadata come
  from the corresponding manifests, not from the tensor shard itself.
- Keep train/validation/test separation at the sequence level; never split
  neighboring frames from one sequence across scientific splits.

## Permissions and retention

- Give teammates Viewer access to the immutable release folder. Max remains the
  sole owner/editor unless a second maintainer is explicitly required.
- The folder consumes the owner's My Drive quota even when shared.
- Never modify a published cache in place. Corrections require a new versioned
  folder and new checksums.
- Retain at least one independently verified local copy until the final course
  submission and one teammate has reproduced validation.
- Record the final private My Drive folder location in the team's private
  channel.

Official Google references:

- [Install Drive for desktop](https://support.google.com/a/users/answer/13022292)
- [Stream versus mirror](https://support.google.com/drive/answer/13401938)
- [Drive for desktop bandwidth limits](https://support.google.com/drive/answer/13470231)
- [Share folders and choose Viewer access](https://support.google.com/drive/answer/7166529)
- [Find shared folders and add a My Drive shortcut](https://support.google.com/drive/answer/2375057)
