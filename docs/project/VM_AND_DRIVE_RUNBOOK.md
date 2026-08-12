# GPU VM and Drive Runbook

This runbook describes the repeatable workflow for running extraction on a
course GPU VM and transferring large caches directly to Google Drive. It is
written for teammates using different VM accounts and public IP addresses.

Do not put passwords, SSH private keys, OAuth tokens, VM IPs, or private Drive
folder IDs in Git. Replace every placeholder below with machine-local values.

## 1. Set Variables

Run these commands on the VM after connecting over SSH:

```bash
export VM_USER="<vm-user>"
export PROJECT_ROOT="$HOME/cut3r-stage0/repos/cut3r-semantic-probing"
export CUT3R_ROOT="$HOME/cut3r-stage0/repos/CUT3R"
export CO3D_ROOT="$HOME/cut3r-stage0/datasets/co3dv2"
export CUT3R_ARTIFACT_ROOT="$HOME/cut3r-stage0/artifacts"
export CUT3R_CACHE_ROOT="$HOME/cut3r-stage0/cache"
export CUT3R_CHECKPOINT="$HOME/cut3r-stage0/checkpoints/cut3r_512_dpt_4_64.pth"

export PYTHONPATH="$PROJECT_ROOT"
```

Use persistent storage for repositories, raw data, manifests, caches, and
results. Do not use `/mnt` on Azure course VMs; it is a temporary resource disk.
Use the OS disk only when it has enough space, or use an approved persistent
share such as `/datashare` after checking its performance and quota.

Check the machine before large work:

```bash
nvidia-smi
df -h /
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
findmnt -T /mnt
time_left
```

## 2. Prepare the Environment

```bash
cd "$PROJECT_ROOT"
source "$HOME/miniconda3/bin/activate" cut3r-stage0
python --version
python -m pytest -q
```

Verify the CUT3R source and checkpoint before extraction:

```bash
git -C "$CUT3R_ROOT" rev-parse HEAD
sha256sum "$CUT3R_CHECKPOINT"
```

The project expects CUT3R commit
`8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`, checkpoint SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`, and the
versioned `curope-scalar-type-v1` compatibility patch.

## 3. Keep the VM Alive

The repository contains `scripts/keep_vm_alive.sh`. It calls the course
`cancel_shutdown` command every five minutes. Install it as a systemd service
so it starts again after a normal reboot:

```bash
sudo install -m 0755 scripts/keep_vm_alive.sh \
  /usr/local/bin/cut3r-keep-vm-alive.sh

sudo tee /etc/systemd/system/cut3r-keepalive.service >/dev/null <<EOF
[Unit]
Description=CUT3R VM shutdown keep-alive
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$VM_USER
ExecStart=/usr/local/bin/cut3r-keep-vm-alive.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now cut3r-keepalive.service
systemctl is-enabled cut3r-keepalive.service
systemctl is-active cut3r-keepalive.service
```

This prevents the course inactivity shutdown when `cancel_shutdown` is
available. It cannot prevent an Azure host failure, quota exhaustion, or a
manual stop. Long jobs must still be resumable.

## 4. Run Long Jobs Safely

Use `tmux`, `screen`, or `nohup`. The cache writers are atomic and resumable;
rerunning a command should reuse completed windows.

Example:

```bash
tmux new -s cut3r-job
```

Detach with `Ctrl-b` then `d`. Reconnect with:

```bash
tmux attach -t cut3r-job
```

If the VM reboots, the process disappears but completed cache shards and
downloaded files remain. Check the cache and rerun the same command.

## 5. Build and Validate an Incremental Manifest

Never re-extract old windows when expanding the dataset. Build a manifest for
the larger cap, then create an incremental manifest containing only sequence IDs
absent from the original manifest.

The incremental manifest must pass all of these checks before extraction:

```bash
python -m scripts.validate_manifests \
  --manifest-dir "$CUT3R_ARTIFACT_ROOT/manifests/<incremental-manifest>" \
  --dataset-root "$CO3D_ROOT" \
  --inspect-files
```

The overlap guard must report:

```text
sequence_intersection: 0
window_ids_unique: true
```

Use a new cache directory for every new backbone/data expansion. Do not point a
new run at an old cache directory.

## 6. Extract Features

Run one backbone at a time. The probe-feature extraction command consumes the
manifest and writes a target-only cache when configured with
`extraction.layout: target_only`:

```bash
python -m scripts.extract_probe_features \
  --config /path/to/<backbone>-incremental.yaml
```

Run a one-window smoke first:

```bash
python -m scripts.extract_probe_features \
  --config /path/to/<backbone>-incremental.yaml \
  --limit-windows 1
```

The final cache should report `layout: target_only` and contain only the new
manifest's windows. For CUT3R, target-only means the target frame's
`image_tokens` and target `state_tokens`; it does not contain all six recurrent
timesteps.

## 7. Install Modern rclone

Old rclone versions may generate Google's deprecated `redirect_uri=oob` OAuth
request and fail with `Error 400: invalid_request`. Use a current release:

```bash
curl -fsSL https://rclone.org/install.sh | sudo bash
rclone version
```

The VM needs a `gdrive` remote only once. Start configuration on the VM:

```bash
rclone config
```

Choose:

- New remote named `gdrive`
- Storage type `drive`
- Blank client ID and secret unless the team owns a custom OAuth client
- Scope `1` (`drive`)
- Blank service account and root folder
- Advanced config: `n`
- Shared client ID warning: `y` if no custom client is available
- Browser authentication on the VM: `n`

The VM then prints a command similar to:

```bash
rclone authorize "drive" "<config-token-argument>"
```

Run that command on a computer with a browser using the same modern rclone
version, complete Google authorization, and paste the resulting one-time config
token into the VM prompt. Never commit or publish the token.

Verify the remote:

```bash
rclone listremotes
rclone lsd gdrive:
```

## 8. Upload Directly to Drive

Upload from the VM, not through a laptop. Use a unique folder name for each
backbone and incremental cache:

```bash
export DRIVE_ROOT="gdrive:<private-team-folder>/stage0-full51-v1/caches/<cache-name>"
export LOCAL_CACHE="$CUT3R_CACHE_ROOT/probe/<cache-name>"

rclone copy "$LOCAL_CACHE" "$DRIVE_ROOT/cache" \
  --transfers 8 \
  --checkers 16 \
  --drive-chunk-size 64M \
  --stats 30s \
  --stats-one-line
```

Upload the manifest and extraction config beside the cache:

```bash
rclone copy "$CUT3R_ARTIFACT_ROOT/manifests/<incremental-manifest>" \
  "$DRIVE_ROOT/manifests" \
  --transfers 4 --checkers 8

rclone copyto /path/to/<backbone>-incremental.yaml \
  "$DRIVE_ROOT/config/<backbone>-incremental.yaml"
```

Upload the cache-layout README to the parent `caches` folder:

```bash
rclone copyto docs/data/part-a-cache-layout.md \
  "gdrive:<private-team-folder>/stage0-full51-v1/caches/PART_A_CACHE_README.md"
```

## 9. Verify Before Cleanup

The cache itself must match exactly:

```bash
rclone check "$LOCAL_CACHE" "$DRIVE_ROOT/cache" --one-way
```

Expected result:

```text
0 differences found
<N> matching files
```

Verify the manifest separately:

```bash
rclone check "$CUT3R_ARTIFACT_ROOT/manifests/<incremental-manifest>" \
  "$DRIVE_ROOT/manifests" --one-way
```

Do not delete the local cache before both checks succeed. Keep the raw CO3D data
if another backbone still needs the same windows.

After all required copies are independently verified, remove only the specific
uploaded cache:

```bash
rm -rf "$LOCAL_CACHE"
```

Never delete the original caches, the original manifests, or shared raw data
while another model still needs them.
