# Approach 3 — Explicit background-statistics reference channels (RAIN / AdaIN style)

> Cheapest of the five (near-zero params). Historically the weakest — HDNet's ablation
> shows matched embeddings (LD) beat matched statistics. **Use as a quick ablation
> baseline or near-free add-on, not as the primary fix.**

---

## 0. Reality check

- See `approach1.md §0`. Same dual-stream stem fusion applies: at the bottleneck there is no separate background tensor, so "background statistics" must be computed by **masking the bottleneck features with the downsampled (1−mask)** (or, for the simplest global variant, by masking the input image).
- The regressor consumes the decoder output (32 ch @256×256). We inject the reference either (a) as extra channels into the decoder input at the bottleneck, or (b) via AdaIN renormalization of the bottleneck features. Option (b) is closer to RAIN and adds essentially zero params; option (a) adds a few thousand.

We document **Option A (concat stat channels)** as primary — it's the most interpretable and the easiest to verify — and note Option B as the near-zero-param alternative.

---

## 1. Option A — background-mean reference channels concatenated at the bottleneck

Compute the background mean feature vector per image, broadcast it to the bottleneck grid, concatenate to the reduced bottleneck, and let a widened `reduce`/first-stage conv ingest it.

### 1a. Widen the bottleneck reduce conv
**File:** `src/train/harmonizer/module/module.py`, `Decoder.__init__`, line 97.

Change:
```python
        self.reduce = nn.Conv2d(c32x, 160, 1)
```
to keep the 160-ch output but add a small side path that appends background stats. Simplest: compute a `bg_ref` (C=1280 mean, projected to 16 ch) and concat before reduce:
```python
        self.bg_proj = nn.Conv2d(c32x, 16, 1)              # project bg mean -> 16 ch
        self.reduce = nn.Conv2d(c32x + 16, 160, 1)         # was c32x -> 160
```

### 1b. Compute and concat in `Decoder.forward`
**File:** `src/train/harmonizer/module/module.py`, replace `forward` head:
```python
    def forward(self, enc2x, enc4x, enc8x, enc16x, enc32x, mask=None):
        skips = [enc16x, enc8x, enc4x, enc2x, None]
        if mask is not None:
            h, w = enc32x.shape[-2:]
            m = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
            bgm = (1 - m)
            denom = bgm.sum(dim=(2, 3), keepdim=True) + 1e-6
            bg_mean = (enc32x * bgm).sum(dim=(2, 3), keepdim=True) / denom  # [n,c,1,1]
            bg_ref = self.bg_proj(bg_mean.expand(-1, -1, h, w))            # [n,16,h,w]
            x = self.reduce(torch.cat([enc32x, bg_ref], dim=1))
        else:
            x = self.reduce(enc32x)                       # baseline path unchanged? see note
        for i in range(self.num_stages):
            x = self.stages[i](x, skips[i])
        return x
```
**Note:** because you changed `reduce`'s in-channels to `c32x+16`, the `mask is None` branch would break. Either always pass mask (per-pixel always has it) or keep a separate `self.reduce_plain`. Cleanest: always pass mask in the per-pixel path (it is available at `harmonizer.py:76`), and drop the `else`.

### 1c. Pass mask from `predict_arguments`
Same edit as `approach1.md §3c` — add `mask` to the `self.decoder(...)` call at `harmonizer.py:76`.

---

## 2. Option B — AdaIN renormalization (near-zero params)

Instead of concatenation, renormalize foreground bottleneck features to background statistics (RAIN's core idea):
```python
# inside Decoder.forward, after computing m, bg_mean, and bg_std similarly:
bg_std = (((enc32x - bg_mean)**2 * bgm).sum((2,3),keepdim=True)/denom + 1e-6).sqrt()
fg_mean = (enc32x * m).sum((2,3),keepdim=True) / (m.sum((2,3),keepdim=True)+1e-6)
fg_std  = (((enc32x - fg_mean)**2 * m).sum((2,3),keepdim=True)/(m.sum((2,3),keepdim=True)+1e-6)+1e-6).sqrt()
normed = (enc32x - fg_mean) / fg_std * bg_std + bg_mean
enc32x = enc32x * (1 - m) + normed * m      # apply only in foreground
x = self.reduce(enc32x)
```
Params added: **0** (no new layers). This is the truest RAIN analogue but least controllable.

---

## 3. Parameter budget

- Option A: `bg_proj` = `Conv2d(1280,16,1)` = `1280*16+16 = 20,496`; widened `reduce` adds `16*160 = 2,560`. Total ≈ **0.023M**.
- Option B: **~0 params**.

Either way the parameter cost is negligible; new total ≈ **5.42M**.

---

## 4. Mirror to inference copy

Apply to `src/model/module.py` (+ `harmonizer.py` mask plumbing). Sync discipline per `approach1.md §5`.

---

## 5. Verification

**5a. Shape/NaN test.** `approach1.md §7a` harness. Watch for divide-by-zero on all-foreground or all-background masks — the `+1e-6` denominators guard this; add an explicit test with `mask=torch.ones(...)` and `mask=torch.zeros(...)`.

**5b. Interpretability check (the point of this approach).** Log the `bg_ref` / `bg_mean` map — it should look like a smooth per-image constant field. Because it's a statistic, you can visualize it directly (unlike a learned embedding). Add it to `_log_argument_maps` (`trainer.py:345`) or dump to disk in the smoke test.

**5c. Param delta.** Confirm `sum(p.numel() for p in m.decoder.bg_proj.parameters())==20496` (Option A) or that Option B adds zero.

**5d. Ablation vs LD.** This approach exists mainly to reproduce HDNet's finding. Run: baseline, +stats (this), +LD (Approach 1). Expectation from the literature: `+LD > +stats`. If stats matches or beats LD on your data, that's a genuinely interesting result worth keeping; otherwise drop it. RAIN's own reported lift was only +0.48 dB on the DIH baseline.

**5e. HAdobe5k subset metric.** As always, judge on the HAdobe5k per-subset PSNR/MSE in TensorBoard, not the average.

---

## 6. Recommendation

Implement Option A or B in an afternoon, run it as a controlled ablation against Approach 1, and use the result to confirm (or challenge) the "matched embeddings beat matched statistics" claim on your specific data. Do not ship it as the main fix.
