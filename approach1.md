# Approach 1 — Bottleneck K-NN Local Dynamic (LD) module

> Implementation spec for the `decoder` branch. Goal: add explicit foreground↔background
> local correspondence at the encoder bottleneck. This is the highest-priority change.

---

## 0. Codebase reality check (read this before touching anything)

The research report made a few assumptions that do **not** match this repo. Correct facts:

| Report said | Actual in this repo | Where to verify |
|---|---|---|
| "8×8×256 bottleneck" | Bottleneck `enc32x` is **8×8×1280** (EfficientNet-B0 head). After `Decoder.reduce` it becomes **8×8×160**. There is no 256-ch bottleneck. | `src/train/harmonizer/module/backbone/efficientnet/__init__.py:19,65-67`; `module.py:97` |
| "separate F_f and F_b feature vectors at the bottleneck" | The backbone is **dual-stream but fused at the stem**: fg=`cat(comp,mask)` and bg=`cat(comp,1-mask)` each pass a stride-2 conv (16 ch each) and are **concatenated to 32 ch before the EfficientNet blocks**. By `enc32x` there is a **single joint feature map** — foreground and background are already mixed. | `backbone/efficientnet/__init__.py:44-55` |
| "1×1 fusion conv ≈ 0.13M params" | True only at 256 ch. At the real 1280-ch bottleneck a `1×1` fusion conv is **~3.3M**; at the reduced 160-ch point it is **~0.05M**. | param math in §4 |

**Consequence:** because fg/bg are fused, you cannot read off separate `F_f`/`F_b` tensors. Instead you compute similarity between all bottleneck tokens and **use the downsampled mask to restrict which tokens count as background candidates** (background = mask≈0) and which locations get updated (foreground = mask≈1). This is functionally equivalent to HDNet's LD and is the correct adaptation here.

**Two files must be kept in sync.** The model is duplicated:
- Training copy (what `train.py` runs): `src/train/harmonizer/module/`
- Inference/eval copy (used by `infer.py`, `batch_inference.py`, `eval_pretrained*.py`, `test_per_pixel.py`): `src/model/`

Any architectural change below must be applied to **both** copies or checkpoints won't load for eval. Do the training copy first, get it working, then mirror.

---

## 1. What we're adding

A `LocalDynamic` module inserted at the reduced bottleneck (8×8×160, right after `Decoder.reduce`). For each foreground token it finds the K=1 most cosine-similar **background** token, copies that background feature as a reference `φ_ref`, concatenates `[x ‖ φ_ref]`, and fuses back to 160 ch with a 1×1 conv. Background locations pass through unchanged. Output feeds the existing decoder stages untouched.

Inserting at the **reduced 160-ch** point (not raw 1280) is the cheap, recommended choice (~0.05M params vs ~3.3M).

---

## 2. New module — add to `module.py`

**File:** `src/train/harmonizer/module/module.py`
**Where:** after the `Decoder` class (after line 117), before `SpatialCascadeArgumentRegressor` (line 120).

```python
class LocalDynamic(nn.Module):
    """Bottleneck foreground<->background correspondence (HDNet-style LD).

    The backbone fuses fg/bg at the stem, so there is no separate F_b tensor.
    We instead use the downsampled mask to mark which tokens are valid background
    candidates and which (foreground) locations get their reference injected.

    x    : [N, C, h, w] joint bottleneck features (C=160 after Decoder.reduce)
    mask : [N, 1, H, W] foreground mask (any resolution; downsampled internally)
    K    : nearest-neighbour count (HDNet found 1 optimal; re-sweep here)
    """
    def __init__(self, channels, k=1):
        super(LocalDynamic, self).__init__()
        self.k = k
        self.fuse = nn.Conv2d(channels * 2, channels, 1)  # [x || phi_ref] -> x

    def forward(self, x, mask):
        n, c, h, w = x.shape
        m = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        m = (m > 0.5).float()                      # [n,1,h,w] 1=foreground
        fg = m.view(n, 1, h * w)                   # [n,1,L]
        bg = 1.0 - fg                              # [n,1,L]

        feat = x.view(n, c, h * w)                 # [n,c,L]
        featn = F.normalize(feat, dim=1)           # cosine -> unit vectors
        sim = torch.bmm(featn.transpose(1, 2), featn)   # [n,L,L] token-token cosine

        # only background tokens are valid neighbours (mask out fg & self-column)
        neg = torch.finfo(sim.dtype).min
        valid = bg.transpose(1, 2).expand(n, h * w, h * w)  # key must be background
        sim = sim.masked_fill(valid < 0.5, neg)

        if self.k == 1:
            idx = sim.argmax(dim=2)                # [n,L] best bg token per location
            phi = torch.gather(feat, 2,
                               idx.unsqueeze(1).expand(n, c, h * w))  # [n,c,L]
        else:
            topv, topi = sim.topk(self.k, dim=2)   # [n,L,k]
            alpha = torch.softmax(topv, dim=2).unsqueeze(1)          # [n,1,L,k]
            gathered = torch.gather(
                feat.unsqueeze(2).expand(n, c, h * w, h * w), 3,
                topi.unsqueeze(1).expand(n, c, h * w, self.k))       # [n,c,L,k]
            phi = (gathered * alpha).sum(dim=3)    # [n,c,L]

        phi = phi.view(n, c, h, w)
        out = self.fuse(torch.cat([x, phi], dim=1))
        # only inject the reference at foreground locations; bg passes through
        return x * (1 - m) + out * m
```

Notes: `torch.bmm` on `[n,64,64]` is trivial. If a background is empty for some image (all-foreground), `argmax` still returns an index; the `x*(1-m)+out*m` gate plus the `neg` fill keeps it stable, but add a guard `if bg.sum()==0: return x` if you see NaNs.

---

## 3. Wire it into the forward path

### 3a. `Decoder.__init__` — construct the module
**File:** `src/train/harmonizer/module/module.py`
**Anchor:** the `Decoder.__init__`, right after `self.reduce = nn.Conv2d(c32x, 160, 1)` (line 97).

```python
        self.reduce = nn.Conv2d(c32x, 160, 1)
        self.local_dynamic = LocalDynamic(160, k=1)   # <-- ADD
```

### 3b. `Decoder.forward` — thread the mask and call LD
**File:** `src/train/harmonizer/module/module.py`
**Anchor:** replace `forward` (lines 112-117):

```python
    def forward(self, enc2x, enc4x, enc8x, enc16x, enc32x, mask=None):   # add mask
        skips = [enc16x, enc8x, enc4x, enc2x, None]
        x = self.reduce(enc32x)
        if mask is not None:
            x = self.local_dynamic(x, mask)          # <-- ADD
        for i in range(self.num_stages):
            x = self.stages[i](x, skips[i])
        return x
```
Keeping `mask=None` default means the global-scalar path and any old caller still works.

### 3c. `Harmonizer.predict_arguments` — pass the mask
**File:** `src/train/harmonizer/module/harmonizer.py`
**Anchor:** line 76, inside the `if self.per_pixel:` branch:

```python
        if self.per_pixel:
            dec = self.decoder(enc2x, enc4x, enc8x, enc16x, enc32x, mask)  # add mask
            arguments = self.regressor(dec)
```
(`mask` here is already the 256×256 interpolated mask from line 69 — `LocalDynamic` re-downsamples it to 8×8, so passing the 256 version is fine.)

---

## 4. Parameter budget (real numbers)

Insertion at the reduced 160-ch bottleneck:
- `fuse` = `Conv2d(320, 160, 1)` → `320*160 + 160 = 51,360` ≈ **0.05M**.
- similarity + gather are **parameter-free**.

New total ≈ 5.4M + 0.05M ≈ **5.45M** — far under the 10M ceiling.

Alternative insertions if you want more capacity:
- On raw `enc32x` (1280 ch): `Conv2d(2560,1280,1)` = **3.28M** (still fits, but 60× costlier for little expected gain).
- 3×3 fuse at 160 ch: `320*160*9+160` ≈ **0.46M**.

Recommendation: start with the 1×1 @160-ch version (0.05M).

---

## 5. Mirror to the inference copy

Repeat §2, §3a, §3b, §3c in `src/model/module.py` and `src/model/harmonizer.py`.
Watch the difference: `src/model/harmonizer.py` uses `SpatialCascadeArgumentRegressor` while the training copy uses `SpatialIndependentArgumentRegressor` — do **not** "fix" that; just add the LD plumbing. Confirm with `diff src/model/module.py src/train/harmonizer/module/module.py` that only intended lines differ.

---

## 6. No config/CLI change needed

LD is on whenever `per_pixel=True` (already set in `src/train/harmonizer/script/train.py:84`). If you want it toggleable for a clean A/B, add a flag:
- `model.py:10-19` add `--local-dynamic` (`cmd.str2bool`, default False), mirror the pattern of `--local-stats`.
- `model.py:33-34` read it and pass `local_dynamic=...` into `_harmonizer.Harmonizer(...)`; thread a constructor arg through `Harmonizer.__init__` → `Decoder(...)`.
- add `('local_dynamic', True)` to the config list in `script/train.py` near line 84.
Optional but recommended so you can run the exact same code with LD off as the baseline.

---

## 7. How to verify it's correct

**7a. Shape / smoke test (no training).** Extend `test_per_pixel.py` (it already builds the per-pixel model). Add:
```python
m = Harmonizer(per_pixel=True, terminal_size=256).eval()
comp = torch.rand(2,3,256,256); mask = (torch.rand(2,1,256,256)>0.5).float()
args = m.predict_arguments(comp, mask)
assert len(args)==6 and all(a.shape[-2:]==(256,256) for a in args)
out = m.restore_image(comp, mask, args)
assert out.shape==(2,3,256,256) and torch.isfinite(out).all()
```
Run: `python test_per_pixel.py`. Confirms no shape/NaN regressions.

**7b. Param count / delta.**
```python
sum(p.numel() for p in m.parameters())          # expect base + ~51k
sum(p.numel() for p in m.decoder.local_dynamic.parameters())  # expect 51,360
```

**7c. LD actually does something (sanity of the mechanism).** With a fixed input, zero the `fuse` weight+bias and confirm output equals the no-LD model at foreground (since `out=x` when fuse≡0 only if you also skip concat — instead just assert that removing `local_dynamic` and adding it changes `predict_arguments` output). Simpler: assert `(args_with_LD != args_without_LD).any()`.

**7d. Correspondence smoke test.** Build a synthetic 8×8 feature map where one background token is an exact copy of a foreground token's vector and all others are orthogonal; check `argmax` selects it. This validates the masked-cosine logic in isolation.

**7e. Overfit one batch.** Train on a single batch for ~200 steps with `ignore_additional=True`; loss should approach ~0. Catches wiring/gradient bugs.

**7f. Real signal — the metric that matters.** Full train run; watch TensorBoard `Metrics/PSNR`, `Metrics/MSE`, and specifically the **HAdobe5k** subset. The trainer already logs per-epoch PSNR/MSE/fMSE/SSIM (`trainer.py:264-268`) and per-subset comparison grids (`_log_comparison_grid`). Compare against the pre-LD baseline checkpoint. Success = HAdobe5k PSNR closes most of the gap toward HDNet's 41.17 (256×256) and average clears ~40.46.

**7g. K sweep.** HDNet's K=1 optimum is for their setup; re-sweep `k ∈ {1,3,5}` on this model (change `LocalDynamic(160, k=...)`). Cheap since only the gather path changes.

---

## 8. Rollback / risk

Single localized change; revert by removing the `local_dynamic` construction + the two call sites. Main risk is empty-background images (all-foreground masks) — guard as noted in §2. Low risk overall; highest expected value of the five approaches.
