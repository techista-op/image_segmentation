# Architecture Audit + LTL / PTL Integration Plan

Integrating the LTL (Locations-to-Location) and PTL (Patches-to-Location) modules from
*"Image Harmonization by Matching Regional References"* (Zhu et al.) into the current
Harmonizer + decoder-branch codebase.

**This document is a plan only — no code has been written.** All file paths, class names,
resolutions, and channel counts below were read directly from the source on the `decoder`
branch, not assumed.

---

## Part A — Current architecture (as found in the code)

### A.1 File map (training path)

The model that `train.py` actually runs lives under `src/train/harmonizer/`:

| File | Key contents |
|---|---|
| `src/train/harmonizer/module/harmonizer.py` | `Harmonizer` (top module): `predict_arguments`, `restore_image`, `adjust_image` |
| `src/train/harmonizer/module/module.py` | `Decoder`, `DecoderStage`, `CascadeArgumentRegressor`, `SpatialCascadeArgumentRegressor`, `SpatialIndependentArgumentRegressor`, `FilterPerformer` |
| `src/train/harmonizer/module/filter.py` | 6 white-box filters + `local_box_mean` |
| `src/train/harmonizer/module/backbone/efficientnet/__init__.py` | `EfficientBackbone` (dual-stream), `EfficientBackboneCommon` |
| `src/train/harmonizer/model.py` | proxy `Harmonizer(TaskModel)` — builds model, defines param groups, `forward` returns `resulter` |
| `src/train/harmonizer/criterion.py` | `HarmonizerLoss` (masked MSE) |
| `src/train/harmonizer/trainer.py` | `HarmonizerTrainer` — train/val loops, all TensorBoard logging |
| `src/train/harmonizer/func.py` | `HarmonizationFunc.metrics` — MSE/fMSE/PSNR/SSIM |
| `src/train/harmonizer/data.py` | `HarmonizerIHarmony4`, `OriginalIHarmony4` datasets |
| `src/train/harmonizer/script/train.py` | run config (per_pixel=True, terminal_size=256, etc.) |

> **Parallel copy:** `src/model/` mirrors the module code for inference/eval (`infer.py`,
> `batch_inference.py`, `eval_pretrained*.py`, `test_per_pixel.py`). It uses
> `SpatialCascadeArgumentRegressor` where the training copy uses
> `SpatialIndependentArgumentRegressor`. **Any architectural change must be mirrored there
> or checkpoints won't load for eval.**

### A.2 Dual encoder (foreground / background streams)

`EfficientBackbone` (`backbone/efficientnet/__init__.py:15-67`). Fusion happens at the
**stem, feature-map level (channel concat), before any EfficientNet block** — *not*
token-level cross-attention and *not* pooled:

- Two separate stride-2 stem convs, each `Conv2d(4 → 16, k=3, s=2)`:
  - `_conv_fg` / `_bn_fg` consumes `fg = cat(comp, mask)` → 16 ch (`harmonizer.py:71`, backbone `:44-45,52`)
  - `_conv_bg` / `_bn_bg` consumes `bg = cat(comp, 1-mask)` → 16 ch (`harmonizer.py:72`, backbone `:47-48,53`)
- `x = torch.cat((xfg, xbg), dim=1)` → **32 ch at 128×128** (backbone `:55`)
- From here a single joint tensor flows through all EfficientNet-B0 blocks. **Foreground and background are already fused; there is no separate `F_b` tensor at deeper stages.** Any fg/bg separation later must be recovered via the (downsampled) mask.

Encoder taps returned (`backbone:67`), for 256×256 input:

| Tap | Resolution | Channels | (`enc_channels` = `[16,24,40,112,1280]`) |
|---|---|---|---|
| `enc2x` = block_outputs[0] | 128×128 | 16 |
| `enc4x` = block_outputs[2] | 64×64 | 24 |
| `enc8x` = block_outputs[4] | 32×32 | 40 |
| `enc16x` = block_outputs[10] | 16×16 | 112 |
| `enc32x` = head | 8×8 | 1280 |

### A.3 U-Net decoder — every stage, resolution, channels

`Decoder` (`module.py:76-117`), `DecoderStage` (`module.py:50-73`). Each stage:
`F.interpolate(×2, bilinear)` → concat same-scale encoder skip → two `Conv2d(3×3)+BN+ReLU`.
With `terminal_size=256`, `TERMINAL_STAGES[256]=5` so **all 5 stages run**
(`module.py:86,94`).

Bottleneck reduce: `self.reduce = Conv2d(1280 → 160, 1×1)` (`module.py:97`), producing
**8×8×160**.

`stage_specs` (`module.py:101-107`), `(in, skip, out)` low-res → high-res:

| Stage | Op | Output resolution | Output channels | Skip concatenated |
|---|---|---|---|---|
| `reduce` | 1×1 conv on `enc32x` | 8×8 | 160 | — |
| `stages[0]` | 8→16 | **16×16** | **128** | `enc16x` (112) |
| `stages[1]` | 16→32 | **32×32** | **96** | `enc8x` (40) |
| `stages[2]` | 32→64 | **64×64** | **64** | `enc4x` (24) |
| `stages[3]` | 64→128 | **128×128** | **32** | `enc2x` (16) |
| `stages[4]` | 128→256 | **256×256** | **32** | none |

`Decoder.forward` (`module.py:112-117`) returns the final **256×256×32** feature map.
`self.out_channels = 32` (`module.py:110`).

> **Clean analogs for the paper:** the paper's **32×32** stage = **`stages[1]` output
> (32×32×96)** exactly; the paper's **128×128** stage = **`stages[3]` output
> (128×128×32)** exactly. No resolution mismatch — see Part B.

### A.4 Independent regressor heads → 6× 256×256 argument maps

`SpatialIndependentArgumentRegressor` (`module.py:158-194`), constructed as
`SpatialIndependentArgumentRegressor(in_channels=32, base_channels=160, out_channels=1,
head_num=6)` (`harmonizer.py:47-48`). It consumes the **256×256×32** decoder output:

- Shared `f = Conv2d(32 → 160, 1×1)`, `g = Conv2d(32 → 160, 1×1)`; `fg = cat(f, g)` → 320 ch (`module.py:172-173,187`).
- 6 independent heads (no cascade), each `Conv2d(320 → 160, 1×1)` then `Conv2d(160 → 1, 1×1)` (`module.py:175-182, 189-193`).
- Output: **list of 6 tensors, each `[N, 1, 256, 256]`** (one spatial argument map per filter: Temperature, Brightness, Contrast, Saturation, Highlight, Shadow — order at `harmonizer.py:22-29`).

Called from `Harmonizer.predict_arguments` (`harmonizer.py:67-80`): input `comp`/`mask` are
bilinearly resized to 256×256 (`:68-69`), backbone runs, decoder runs, regressor runs.

### A.5 Bicubic upsample 256 → original resolution

Two places, both in `harmonizer.py`:

- `restore_image` (`:82-98`): each 256×256 argument map is `F.interpolate(arg, size=comp
  H×W, mode='bicubic', align_corners=False)` then `clamp(-1,1)` (`:88-93`), then filters are
  applied element-wise by `FilterPerformer.restore` (`:97`). The mask is applied at full
  resolution inside the performer (`module.py:218`), so the crisp composite edge comes from
  the full-res mask, not the upsampled maps.
- `adjust_image` (`:100-117`): same bicubic upsample of scaled arguments (`:106-112`) for the
  synthetic-degradation path used in training.

### A.6 Current loss function(s)

`HarmonizerLoss` (`criterion.py:17-38`): **foreground-masked MSE on the final output image
only**. `loss = Σ(MSE(pred,gt)·mask)/Σmask`, per-image. No argument-space or perceptual term.

In the trainer (`trainer.py:95-123`) this single criterion is applied twice per step:
- `fine_loss` = criterion on the **labeled** sub-batch (`:96-105`)
- `coarse_loss` = criterion on the **additional-data** sub-batch × 10 (`:108-118`)
- optional `smooth_loss` = TV penalty on argument maps if `smooth_lambda>0` (`:128-133`, `_argument_smoothness` `:499-506`)
- `loss = fine_loss + coarse_loss (+ smooth)`

> Note: `fine`/`coarse` here are **data partitions** (labeled vs additional), *not* stage
> losses. Keep this in mind when adding LTL/PTL — do not overload these names.

### A.7 Current TensorBoard logging (exact tags)

All in `trainer.py`. Writer root: `runs/<log_tag>` (`:63`).

Train (`_train`, per epoch):
- `Loss/train` (total), `Loss/fine_loss`, `Loss/coarse_loss` (`:167-169`)
- `LR/lr` (`:170`)

Validation (`_validate`, per epoch):
- `Loss/val` (`:264`)
- `Metrics/MSE`, `Metrics/fMSE`, `Metrics/PSNR`, `Metrics/SSIM` (`:265-268`)
- Every 5 epochs (`log_args`, `:199`): `Args/mean_<filter>`, `Args/std_<filter>` (`:271-274`);
  `Images/comparison_grid` (`_log_comparison_grid` `:328-343`, 4 fixed samples one per subset
  = `[comp|gt|out|diff]`); `Args/map_<filter>` heatmaps (`_log_argument_maps` `:345-360`).
- JSON mirror of scalars to `checkpoints/<tag>/training_log.json` (`:281-300`).

Metrics computed in `func.py:16-41` (MSE/fMSE/PSNR/SSIM), all under **`id_str='IH'`**
(hardcoded in `trainer.py:227`).

> **Critical gap for this project:** metrics are **aggregated across all four subsets** under
> one `IH` tag. The val loader concatenates HCOCO/HFlickr/HAdobe5k/Hday2night (config
> `valset`, `train.py:71-77`) and the dataset never returns which subset a sample came from
> (`data.py:__getitem__` returns only `((adjusted,mask),(image,))`). So **there is currently
> no per-subset PSNR/MSE.** Part C.4 specifies the change needed to get it — this is the
> single most important logging addition given HAdobe5k is the weak point.

---

## Part B — LTL / PTL integration plan

### B.0 Resolution-ladder mapping (paper → our code)

| Paper module | Paper resolution | Our closest stage | Our resolution × channels | Mismatch? |
|---|---|---|---|---|
| **LTL** | 32×32 | **`stages[1]` output** | 32×32 × **96** | **None** — exact match |
| **PTL** | 128×128 | **`stages[3]` output** | 128×128 × **32** | **None** — exact match |

Both target resolutions exist exactly in our ladder, so no nearest-neighbor compromise is
needed. The only adaptation vs the paper is **channel count** (paper operates on its own
decoder widths; we use 96 and 32 respectively) and the **fg/bg separation** (paper has
explicit streams; we reconstruct fg/bg from the downsampled mask because our backbone fuses
them at the stem — see A.2).

---

### B.1 LTL — Locations-to-Location Translation (low-res, 32×32×96)

**Purpose.** Cross-attention giving each foreground location a background *appearance target*
matched by content similarity: `T_r = softmax(T_f · T_bᵀ) · T_b`, concat with `T_f`,
linear-fuse back to C.

**File(s) to modify.**
- New class in `src/train/harmonizer/module/module.py` (alongside `Decoder`).
- Wire-in: `Decoder.__init__` + `Decoder.forward` in the same file; mask plumbing in
  `harmonizer.py:predict_arguments` (`:76`).
- Mirror to `src/model/module.py` + `src/model/harmonizer.py`.

**Insertion point.** After `stages[1]` (32×32×96), before `stages[2]`. This is the paper's
32×32 stage exactly.

**New module (spec, not code).** `class LTL(nn.Module)`, C=96:
- Linear/1×1 projections `q,k,v: Conv2d(96→96,1×1)` (v optional; can reuse features).
- Downsample mask to 32×32; `m∈{0,1}`. Queries = all L=1024 tokens (or only fg tokens);
  keys/values = **background** tokens only (weight keys by `(1-m)`; set fg-key logits to
  −inf before softmax so attention can only draw from background).
- `A = softmax(q_f · k_bᵀ / √C)` over background keys; `T_r = A · v_b`.
- `out = fuse(cat[T_f, T_r])`, `fuse = Conv2d(192→96,1×1)`.
- Apply only at foreground: `x = x·(1−m) + out·m`.

**Forward wiring.** In `Decoder.forward` add `mask` arg; after `x = self.stages[1](x, skips[1])`
call `x = self.ltl(x, mask)` (guarded by `if mask is not None`). In
`Harmonizer.predict_arguments` pass the already-resized 256×256 `mask` into
`self.decoder(..., mask)`.

**Approx added parameters.** q,k,v = 3 × (96·96 + 96) ≈ 27,936; fuse = 192·96 + 96 = 18,528.
**≈ 0.046M params** (drop v-projection → ≈ 0.037M).

**Approx VRAM @ inference, batch 1.** Attention logits `[L_q, L_k]` at 32×32 dense =
1024×1024 × 4 B ≈ **4 MB** (fp32; ~2 MB bf16). Projections/activations at 32×32×96 are
~1.5 MB each. **Total on the order of ~10–15 MB.** Well within the tens-of-MB budget.
*Flag:* if you make queries all-tokens AND keep fp32 AND store attention for logging every
step, peak could ~2×; log attention only every N steps (Part C.1).

**Resolution-mismatch issues.** None. If you later run a smaller `terminal_size` ablation
(fewer stages), guard construction/use with `self.num_stages > 2` so `stages[1]` exists.

---

### B.2 PTL — Patches-to-Location Translation (high-res, 128×128×32)

**Purpose.** Match foreground *content* to background *patch* content, then apply the matched
**patch's** appearance stats (mean/std) — not a single global background mean/std — to the
foreground via AdaIN-style scale+shift. Using patches (not per-pixel tokens) on the key side
is what keeps high-res attention cheap.

**File(s) to modify.**
- New class in `src/train/harmonizer/module/module.py`.
- Wire-in: `Decoder.__init__` + `Decoder.forward`; mask already threaded from B.1.
- Mirror to `src/model/module.py`.

**Insertion point.** After `stages[3]` (128×128×32), before `stages[4]`. This is the paper's
128×128 stage exactly.

**New module (spec, not code).** `class PTL(nn.Module)`, C=32:
- Split **background** feature map into overlapping patches via `F.unfold` (e.g. patch=16,
  stride=8 → `N_p ≈ 15×15 ≈ 225` patches; tune). Weight by downsampled `(1−m)` so patches are
  background-dominated.
- Per patch compute **appearance** = (mean μ_p, std σ_p) per channel, and **content** =
  instance-normalized patch, then pooled to a content descriptor `c_p ∈ R^C`.
- Foreground **content** = instance-normalized foreground features, per-location descriptor
  `c_f(i) ∈ R^C` (128×128 locations).
- `A = softmax(c_f · c_pᵀ / √C)` over patches → per-location weights.
- Matched stats: `μ*(i) = Σ_p A(i,p) μ_p`, `σ*(i) = Σ_p A(i,p) σ_p`.
- AdaIN: `out(i) = σ*(i) · IN(x)(i) + μ*(i)`; apply only at foreground `x = x·(1−m)+out·m`.
- Learned params: small content projections `Conv2d(32→32,1×1)` for fg and patch descriptors
  (2×). AdaIN itself is parameter-free.

**Forward wiring.** In `Decoder.forward`, after `x = self.stages[3](x, skips[3])` call
`x = self.ptl(x, mask)` (guarded).

**Approx added parameters.** 2 × (32·32 + 32) ≈ 2,112, plus optional 1×1 fuse
32·32+32 = 1,056. **≈ 0.002–0.003M params.**

**Approx VRAM @ inference, batch 1.** Query side L_q = 128×128 = 16,384 locations; key side =
**N_p ≈ 225 patches** (not 16,384). Attention `[16384, 225]` × 4 B ≈ **14.7 MB** (fp32; ~7 MB
bf16). `unfold` at patch=16 produces `[1, 32·256, 225]` ≈ 1.8M floats ≈ 7 MB. Per-patch
μ/σ are tiny. **Total on the order of ~25–35 MB.** Within tens-of-MB budget.
*Flag — this is the one to watch:* if PTL is (mis)implemented with per-pixel background keys
(16,384 × 16,384) it explodes to **~1 GB** and blows the budget. The patch key side is
mandatory. Keep `N_p` in the low hundreds; smaller stride raises `N_p` and cost roughly
linearly. Also avoid materializing an instance-normalized copy of the full unfold tensor more
than once.

**Resolution-mismatch issues.** None at 128×128. Guard use with `self.num_stages > 4` (i.e.
`terminal_size == 256`); for smaller terminal sizes PTL's 128×128 stage may not exist — in
that case fall back to the highest available stage and note the change.

---

### B.3 Combined cost summary

| Module | Stage | Res × C | Params | VRAM @ bs1 |
|---|---|---|---|---|
| LTL | after `stages[1]` | 32×32×96 | ~0.046M | ~10–15 MB |
| PTL | after `stages[3]` | 128×128×32 | ~0.003M | ~25–35 MB |
| **Total added** | | | **~0.05M** | **~35–50 MB** |

New model total ≈ 5.4M + 0.05M ≈ **5.45M** (ceiling 10M). Parameter budget is not the
binding constraint; the VRAM stays in tens of MB as required, **provided PTL uses patch keys**.

---

## Part C — TensorBoard logging plan (verify LTL/PTL are doing real work)

All additions go in `src/train/harmonizer/trainer.py`. To expose attention maps and
per-module tensors, the modules should stash their last attention tensor on `self`
(e.g. `self.last_attn`) during forward, and the proxy `model.py:forward` can surface
references via `resulter` (or the trainer reaches them through
`self.model.module.model.decoder.ltl.last_attn`).

### C.1 LTL / PTL attention health (collapse detection)

For each module, log **attention entropy** normalized by `log(N_keys)` (1.0 = uniform,
→0 = collapsed to one location):
- `LTL/attn_entropy` — mean over foreground queries of `-Σ A log A / log(L_k)`.
- `PTL/attn_entropy` — same over patches, normalized by `log(N_p)`.
- Also log `LTL/attn_max` and `PTL/attn_max` (mean of per-query max weight) — a spike toward
  1.0 with entropy →0 means collapse to a single location; entropy ≈1.0 persistently means the
  module is degenerate/ignored.
- **Image tags** every N steps: `LTL/attn_example` and `PTL/attn_example` — for a fixed query
  location on a fixed val image, render the attention distribution back over the background
  spatial grid as a heatmap (reuse `_log_argument_maps` style, `trainer.py:345-360`).

### C.2 Dead-module detection (gradient flow)

Per epoch, log the gradient norm into each module's weights right after `loss.backward()`
(`trainer.py:136`), before `optimizer.step()`:
- `GradNorm/ltl` = Σ‖p.grad‖ over `decoder.ltl.parameters()`.
- `GradNorm/ptl` = Σ‖p.grad‖ over `decoder.ptl.parameters()`.
- `GradNorm/decoder`, `GradNorm/regressor` as baselines for scale.
A norm that is ~0 or orders of magnitude below the decoder means the module is dead weight.

### C.3 Before/after loss curves (regression visibility)

Keep the existing `Loss/train`, `Loss/fine_loss`, `Loss/coarse_loss`, `Loss/val` tags
unchanged so old and new runs overlay directly in TensorBoard. Add a distinguishing
`log_tag` per experiment (config `train.py`, consumed at `trainer.py:62`), e.g.
`baseline`, `+ltl`, `+ltl+ptl`. This makes any regression immediately visible against the
pre-module baseline curve.

### C.4 Per-subset PSNR/MSE (HCOCO / HFlickr / HAdobe5k / Hday2night) — REQUIRED CHANGE

Currently impossible without a code change (see A.7 gap). Minimal plan:
1. `data.py`: have `OriginalIHarmony4.__getitem__` also return its `subset` string (it's
   known at `:140`). Propagate the subset id through the batch (e.g. as a third tuple element
   or a parallel list).
2. `trainer.py:_validate` (`:227`): replace the hardcoded `id_str='IH'` with the per-sample
   subset, so `func.metrics` accumulates `HCOCO_*_psnr`, `HAdobe5k_*_mse`, etc.
3. `trainer.py` (`:264-268`): also emit `Metrics/PSNR/HAdobe5k`, `Metrics/MSE/HAdobe5k`
   (and the other three) as separate scalars, plus keep the aggregate `IH`.
This is the key signal: it shows whether LTL/PTL specifically move **HAdobe5k** and
lighting-heavy scenes rather than just nudging the average.

### C.5 Fixed hard-case validation images every N steps

Extend the existing fixed-sample machinery (`_ensure_fixed_samples` `:304-326`,
`_log_comparison_grid` `:328-343`, which already logs one sample per subset). Add:
- At least one **hand-picked hard case**: strong local background lighting variation (e.g. a
  bright window in an otherwise dark room) with a foreground that a global correction
  over/under-exposes. Drop the file id into the fixed-sample list.
- Log `Images/hardcase_grid` = `[comp | gt | output | amplified-diff]` every N steps (reuse
  the existing grid builder). Track visually whether the over/under-correction improves as
  LTL/PTL train.
- Optionally overlay the LTL attention heatmap for that image's foreground centroid so you can
  see *which* background region the model decided to match.

---

## Part D — Order of operations

**Add and validate one module at a time — LTL first, then PTL.** Reasons: LTL is the simpler
mechanism (single cross-attention, exact 32×32 match), it's the paper's primary contributor,
and isolating it makes the gradient-norm / attention-entropy diagnostics unambiguous.

Recommended sequence:
1. **Baseline run** on the current model with the new logging (C.3, C.4, C.5) *already in
   place* — you need the per-subset baseline curves before changing the model.
2. **LTL alone.** Verify in isolation before trusting metrics:
   - Shape/NaN smoke test (extend `test_per_pixel.py`): decoder still returns 256×256×32; args
     still 6×[N,1,256,256]; finite.
   - `GradNorm/ltl` > 0 and comparable order to the decoder (C.2) — proves it's training.
   - `LTL/attn_entropy` neither pinned at 1.0 (ignored) nor at 0 (collapsed) after a few
     hundred steps (C.1).
   - Overfit one batch → loss → ~0 (wiring/gradient check).
   - Then a full run; watch `Metrics/PSNR/HAdobe5k` vs baseline.
3. **PTL alone** (LTL removed) — same isolation checks, plus the VRAM guard: confirm inference
   peak is tens of MB (patch keys, not per-pixel). This tells you PTL's independent
   contribution.
4. **LTL + PTL together** — only after each passes in isolation. Compare against both single-
   module runs on the per-subset HAdobe5k metric to confirm the combination is additive, not
   redundant.

Fastest isolation verification for each: the **gradient-norm + attention-entropy pair**
catches a dead or collapsed module within a few hundred steps, long before full-run metrics
are meaningful — run those first, then commit to a full training run.

---

## Open items / assumptions flagged (not guessed)

- **Per-subset metrics require the C.4 code change** — the current val pipeline genuinely has
  no subset tagging. Confirm you want subset propagated via the dataset return tuple vs a
  separate index map (either works; dataset-return is simplest).
- **PTL patch geometry** (patch size / stride → `N_p`) is a tunable that directly sets VRAM
  and match granularity; the plan assumes patch=16, stride=8 (`N_p≈225`) as a starting point.
- **bf16 autocast** is already enabled in training (`amp_bf16=True`, `train.py`), which halves
  the attention-tensor VRAM figures above; the MB estimates quote fp32 to be conservative.
- LTL/PTL both reconstruct fg/bg from the **downsampled mask** because the backbone fuses the
  two streams at the stem (A.2) — there is no separate background feature tensor to attend to,
  unlike the paper's explicit streams. This is the one substantive architectural adaptation.
