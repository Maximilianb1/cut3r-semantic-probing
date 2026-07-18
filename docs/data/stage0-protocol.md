# Stage 0 data and feature protocol

Stage 0 prepares data and extracts frozen CUT3R representations. It does not
train or fine-tune CUT3R, a segmentation head, or a classification head.

## Dataset tiers

| Tier | Categories | Sequence cap per category (train/val/test) | Windows per sequence | Purpose |
|---|---:|---:|---:|---|
| Debug | `ball`, `chair` | 3/3/3 | up to 3 | Local correctness and visualization |
| Pilot | 10 named categories | 50/10/10 | up to 4 | GPU runtime, memory, storage, and initial pipeline validation |
| Full | Exactly all 51 CO3Dv2 categories | 50/10/10 | up to 4 | Later probe datasets after the pilot gate |

Categories with fewer sequences use all available official sequences. The caps
are explicit configuration values and may change only through a new versioned
configuration and documented rationale.

## Selective acquisition

The bounded Debug and Pilot tiers do not require complete CO3Dv2 category
archives. `scripts.download_co3d_selective` pins the upstream CO3D link and
checksum indexes to tooling commit
`eb51d7583c56ff23dc918d9deafee50f4d8178c3`. It fully downloads and verifies
each small `<category>_000.zip` metadata archive, applies the exact sampler
below, and uses HTTP byte ranges to read only selected RGB/mask members from the
large official ZIPs.

Acquisition has three explicit gates:

1. `--plan-only` verifies metadata and reports selected sequences, windows, and
   files without inspecting the large data ZIPs.
2. `--index-only` locates every selected member and reports its source archive
   plus projected compressed/uncompressed bytes without reading payloads.
3. Full mode validates member size and ZIP CRC, atomically writes the payload,
   and records a per-file SHA-256 in
   `$CO3D_ROOT/.co3d-selective/selection.json`.

The official SHA-256 of every containing ZIP is recorded as provenance. It is
not claimed as locally verified because proving that hash would require
downloading the entire archive, which defeats selective acquisition. The small
metadata ZIP hashes are verified exactly; selected members are verified by ZIP
CRC and per-file SHA-256, and later cache records bind to the exact local bytes.
Only named category lists and finite caps are accepted to prevent an accidental
full-dataset download. Unsafe paths, links, missing members, changed remote
metadata, oversized files, and corrupt existing files fail closed.

## Split rule

The unit of splitting is a CO3Dv2 sequence: one video of one physical object.
The official few-view train/dev/test set lists determine the project
train/validation/test sequence assignments. All frames, windows, and cached
features from one sequence stay in that split. Validation fails on overlap.

## Deterministic six-frame windows

For a sequence with `N` usable frames, requested window count `R`, and window
length six:

1. Sort frames by the annotated `frame_number`, not the filename suffix.
2. Set `W = min(R, floor(N / 6))`.
3. Select `6W` evenly spaced indices from the full sorted sequence, including
   its first and last available positions.
4. Divide the selected frames into `W` consecutive ordered groups of six.
5. Use frame 6 of each group as the primary supervised target in later stages.

The selected windows do not share raw frames within a sequence. Selection has
no dependence on directory enumeration order. Sequence caps use a stable
SHA-256 ranking over seed, category, split, and sequence ID.

Example for 30 frames and four requested windows: select 24 evenly distributed
frames and group them as four ordered six-frame windows. The remaining six raw
frames are not used by this dataset version.

## Manifests

`sequences.parquet` records category, sequence ID, official split, annotation
quality, counts, selection status, and exclusion reason.

`frames.parquet` records stable frame ID, annotated number and timestamp, RGB
and mask paths, original dimensions, soft mask mass/fraction, viewpoint JSON,
the exact deterministic resize/crop plan as JSON, category, sequence, and split.

`windows.parquet` records the stable window ID, ordered frame IDs and numbers,
target frame, target timestep, sampler version, and target-RGB visibility.

`summary.json` binds these files to the configuration hash and records counts,
rejections, and manifest SHA-256 values.

## Shared spatial transform

CUT3R's released 512 preprocessing preserves aspect ratio, resizes the longest
edge to 512, and center-crops to patch-aligned dimensions. Square inputs are
center-cropped to a 4:3 output when `square_ok=false`. RGB uses CUT3R's
bicubic/Lanczos rule; masks use nearest-neighbor interpolation. Both receive
identical resize dimensions and crop coordinates.

Manifest validation recomputes every recorded plan. With `--inspect-files`, it
also decodes every selected RGB and mask and verifies that both dimensions match
the CO3D annotation before any GPU work begins. It applies the configured
transform and threshold to every frame-6 target mask and rejects empty targets.

## Cached trajectory

Every six-frame window stores two float16 tensors:

- `image_tokens`: `[6, batch, spatial_tokens, channels]`;
- `state_tokens`: `[6, batch, persistent_state_tokens, channels]`.

Image tokens are the final normalized per-view decoder tokens after interaction
with the prior persistent state; the pose token is removed. State tokens are
the committed persistent state after each timestep. Cache identity is the
window plus timestep because the same image can produce different features
under different preceding context.

Each cache-index row also stores SHA-256 values for the six exact RGB files and
their six masks. Cache metadata records the pinned upstream CUT3R commit, exact
compatibility-patch source hash, compiled cuRoPE shared-object hash, checkpoint
hash and restricted-load policy, manifest hashes, preprocessing, and
Python/PyTorch/CUDA device runtime. The checkpoint is rejected before
deserialization unless its SHA-256 matches the versioned configuration and its
static pickle globals equal the audited seven-type OmegaConf allowlist. This
binds a representation to its actual input bytes and environment.

## Preflight gates

Before pilot/full extraction:

1. CPU tests and CLI help checks pass.
2. Real debug manifests pass file, split, order, and hash validation.
3. Human review confirms RGB/mask overlays for varied aspect ratios.
4. The Technion VM reports GPU, disk, memory, and remaining-time capacity.
5. One window is independently extracted into two caches with expected shapes;
   `scripts.compare_caches` confirms exact equality (or a separately documented
   nonzero tolerance if the pinned CUDA stack cannot be bitwise deterministic).
6. Cache verification passes after a one-window extraction.
7. A 100-window run records seconds/window, peak VRAM, cache bytes/window, and
   projected full runtime/storage.

Probe overfitting and scientific metrics begin in Stages 1 and 2, not Stage 0.
