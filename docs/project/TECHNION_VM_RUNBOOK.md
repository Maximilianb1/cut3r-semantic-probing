# Technion GPU preflight runbook

The course guide describes an Azure Linux VM in the DDS course subscription,
using an NVIDIA A10 v5-class configuration. Do not copy the guide's credentials,
example address, or other secrets into this repository.

For moving large artifacts between a teammate's Google Drive and the VM
without staging on a personal laptop, see
[`DRIVE_TO_VM_RUNBOOK.md`](DRIVE_TO_VM_RUNBOOK.md).

## Before work

Start the assigned VM in Azure, obtain its current public address, and connect
over SSH. The address may change after shutdown or reboot.

Inside the VM, record:

```bash
nvidia-smi
df -h
free -h
time_left
git --version
python --version
```

Before choosing any artifact path, identify its backing device:

```bash
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
findmnt -T /mnt -o TARGET,SOURCE,FSTYPE,OPTIONS
readlink -f /dev/disk/azure/resource 2>/dev/null || true
```

Azure's resource disk is temporary. On this course VM it appears as
`/dev/disk/azure/resource`, is mounted at `/mnt`, and was observed being
reinitialized after a VM stop/recreation. Never place repositories,
checkpoints, datasets, caches, manifests, logs, or results there. Use the
persistent OS disk only for the small control-plane artifacts that fit, and
obtain an approved managed data disk or student-writable persistent share for
CO3Dv2 and large caches.

Use `tmux` or `screen` for extraction. The VM can initiate an automatic shutdown
after an inactivity period and broadcasts a warning first; the course-provided
`cancel_shutdown` command cancels a pending shutdown.

## Stage 0 order

1. Clone this repository and check upstream CUT3R out at
   `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`.
2. Create a clean CUDA environment following upstream CUT3R instructions.
3. Install this project, apply its versioned `curope-scalar-type-v1` patch, and
   compile cuRoPE. Extraction accepts only that exact one-line compatibility
   change and rejects every other upstream modification.

   ```bash
   python -m scripts.apply_cut3r_compatibility_patch \
     --cut3r-root "$CUT3R_ROOT" \
     --expected-commit 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
   (cd "$CUT3R_ROOT/src/croco/models/curope" && \
     python setup.py build_ext --inplace)
   ```
4. Set the five external path variables documented in the root README.
5. Download the released 512 checkpoint and confirm its SHA-256 equals
   `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
   The hash is a project trust anchor recorded from the official download on
   2026-07-18; a changed upstream file requires review rather than bypassing the
   check.
6. Run the test suite and the checkpoint-load smoke test:

   ```bash
   python -m pytest -q
   python -m scripts.validate_checkpoint \
     --config configs/stage0/debug.yaml \
     --load-model
   ```
7. In `tmux`, selectively acquire the debug data in three reviewable gates:

   ```bash
   mkdir -p "$CUT3R_ARTIFACT_ROOT/downloads" "$HOME/cut3r-stage0/logs"
   set -o pipefail
   python -m scripts.download_co3d_selective \
     --config configs/stage0/debug.yaml --plan-only \
     > "$CUT3R_ARTIFACT_ROOT/downloads/debug-plan.json" \
     2> >(tee "$HOME/cut3r-stage0/logs/debug-download-plan.log" >&2)
   python -m scripts.download_co3d_selective \
     --config configs/stage0/debug.yaml --index-only \
     > "$CUT3R_ARTIFACT_ROOT/downloads/debug-index.json" \
     2> >(tee "$HOME/cut3r-stage0/logs/debug-download-index.log" >&2)
   python -m scripts.download_co3d_selective \
     --config configs/stage0/debug.yaml \
     > "$CUT3R_ARTIFACT_ROOT/downloads/debug-result.json" \
     2> >(tee "$HOME/cut3r-stage0/logs/debug-download.log" >&2)
   ```

   Review each JSON count/byte projection before continuing. A rerun resumes
   only from files whose ZIP size and CRC still match; remove a reported corrupt
   file rather than bypassing validation.
8. Build and validate real debug manifests.
9. Extract one window into two distinct cache directories, compare them with
   `scripts.compare_caches`, and validate both caches.
10. Run and record a 100-window performance pilot.
11. Approve or revise the pilot/full caps based on measured projections.

Do not begin the full extraction until all Stage 0 preflight gates pass.

The gates passed on the complete valid Debug run. The approved all-category
execution now follows the dedicated
[Full-51 two-part runbook](FULL51_TWO_PART_RUNBOOK.md): 30/5/5 sequences per
category, Part A and Part B run sequentially, and each cache is copied and
hash-verified off-VM before deletion. Never start both parts concurrently.

## Storage incident response

If `/mnt` contains only `DATALOSS_WARNING_README.txt` and `lost+found`, the
resource disk has been reinitialized; do not attempt to reconstruct unique data
there. Re-clone source and re-download verified public artifacts onto approved
persistent storage. If diagnostic output accidentally includes a credential,
do not copy it into the repository or another message: notify the owning course
administrator and have it rotated immediately.

## After work

Flush cache shards, save logs and the preflight report outside Git, exit the
session, and stop/deallocate the VM through Azure. The course guide assigns a
finite VM-minute quota; leaving the VM running consumes shared resources.
