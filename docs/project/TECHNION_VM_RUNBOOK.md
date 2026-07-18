# Technion GPU preflight runbook

The course guide describes an Azure Linux VM in the DDS course subscription,
using an NVIDIA A10 v5-class configuration. Do not copy the guide's credentials,
example address, or other secrets into this repository.

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
5. Run the test suite.
6. Build and validate real debug manifests.
7. Extract one window into two distinct cache directories, compare them with
   `scripts.compare_caches`, and validate both caches.
8. Run and record a 100-window performance pilot.
9. Approve or revise the pilot/full caps based on measured projections.

Do not begin the full extraction until all Stage 0 preflight gates pass.

## After work

Flush cache shards, save logs and the preflight report outside Git, exit the
session, and stop/deallocate the VM through Azure. The course guide assigns a
finite VM-minute quota; leaving the VM running consumes shared resources.
