# Approach 2 — Multi-scale LD (add a second correspondence at 32×32)

> Builds on Approach 1. Only pursue after Approaches 1 (+5) plateau. Adds spatial
> granularity to correspondence, which 8×8 matching cannot provide.

---

## 0. Prerequisite & reality check

- **Do Approach 1 first.** This reuses the `LocalDynamic` module defined there.
- Read `approach1.md §0` for the architecture facts (dual-stream stem fusion, mask-based fg/bg partition, real channel counts). They all apply here.
- Decoder feature resolutions and channels (from `module.py:99-110`, terminal_size=256, `enc_channels=(16,24,40,112,1280)`):

| Decoder point | Resolution | Channels | Source |
|---|---|---|---|
| after `reduce` | 8×8 | 160 | `module.py:97` |
| after `stages[0]` | 16×16 | 128 | `stage_specs[0]=(160,c16x,128)` |
| **after `stages[1]`** | **32×32** | **96** | `stage_specs[1]=(128,c8x,96)` ← target |
| after `stages[2]` | 64×64 | 64 | `stage_specs[2]=(96,c4x,64)` |

We add a **second LD at the 32×32 / 96-ch point** (after `stages[1]`). At 32×32 there are 1024 tokens, so a dense 1024×1024 similarity is ~1M entries per image — tractable but heavier; use **windowed** matching to keep it cheap.

---

## 1. New module — windowed LD

A full dense LD at 32×32 works but is wasteful. Restrict each foreground token's background candidates to a local `W×W` window (e.g. 7×7). Add to `src/train/harmonizer/module/module.py` after `LocalDynamic`:

```python
class WindowedLocalDynamic(nn.Module):
    """LD with a local W x W background search window (cheap high-res match)."""
    def __init__(self, channels, k=1, window=7):
        super(WindowedLocalDynamic, self).__init__()
        self.k, self.window = k, window
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x, mask):
        n, c, h, w = x.shape
        m = (F.interpolate(mask, size=(h, w), mode='bilinear',
                           align_corners=False) > 0.5).float()
        pad = self.window // 2
        featn = F.normalize(x, dim=1)
        # unfold background candidates in each local window
        bgfeat = (x * (1 - m))
        cols = F.unfold(bgfeat, self.window, padding=pad)        # [n, c*W*W, L]
        colsn = F.unfold(F.normalize(bgfeat, dim=1), self.window, padding=pad)
        L = h * w
        cols = cols.view(n, c, self.window * self.window, L)
        colsn = colsn.view(n, c, self.window * self.window, L)
        q = featn.view(n, c, 1, L)
        sim = (q * colsn).sum(1)                                 # [n, W*W, L]
        # invalidate window slots that were foreground (their bg feat is ~0)
        idx = sim.argmax(dim=1, keepdim=True)                    # [n,1,L] K=1
        phi = torch.gather(cols, 2, idx.unsqueeze(1).expand(n, c, 1, L)).squeeze(2)
        phi = phi.view(n, c, h, w)
        out = self.fuse(torch.cat([x, phi], dim=1))
        return x * (1 - m) + out * m
```

This is O(L·W²) instead of O(L²): ~1024·49 vs ~1024² — a ~20× compute saving. (A dense version is fine functionally; swap in `LocalDynamic` from Approach 1 if you'd rather not window.)

---

## 2. Wire into the decoder

**File:** `src/train/harmonizer/module/module.py`, `Decoder.__init__` (after the Approach-1 `local_dynamic` line):

```python
        self.local_dynamic = LocalDynamic(160, k=1)          # Approach 1 (8x8)
        self.local_dynamic_hi = WindowedLocalDynamic(96, k=1, window=7)  # 32x32
```

**`Decoder.forward`** — call it right after `stages[1]`:

```python
    def forward(self, enc2x, enc4x, enc8x, enc16x, enc32x, mask=None):
        skips = [enc16x, enc8x, enc4x, enc2x, None]
        x = self.reduce(enc32x)
        if mask is not None:
            x = self.local_dynamic(x, mask)
        for i in range(self.num_stages):
            x = self.stages[i](x, skips[i])
            if mask is not None and i == 1:          # after stages[1] -> 32x32
                x = self.local_dynamic_hi(x, mask)   # <-- ADD
        return x
```

Guard the index: `local_dynamic_hi` only exists / makes sense when `num_stages > 2` (terminal_size ≥ 64). For terminal_size=256 (`num_stages=5`) this is always satisfied.

---

## 3. Parameter budget

- `fuse` at 96 ch = `Conv2d(192, 96, 1)` → `192*96 + 96 = 18,528` ≈ **0.02M**.
- Parameter-free similarity/gather.
- With an optional pre-reduction conv or 3×3 fuse, budget **~0.2–0.5M** as the report estimated; the minimal version is ~0.02M.

New total (Approach 1 + 2) ≈ 5.45M + 0.02M ≈ **5.47M**.

---

## 4. Mirror to inference copy

Apply §1–§2 to `src/model/module.py`. Same sync discipline as Approach 1 §5.

---

## 5. Verification

**5a. Shape/NaN test.** Same harness as `approach1.md §7a`. Additionally assert the decoder still returns 32-ch @256×256 output (unchanged) and that `local_dynamic_hi` fires (add a temporary `print(x.shape)` inside the `i==1` branch, expect `[N,96,32,32]`).

**5b. Compute check.** Time one forward with vs without the high-res LD (`python -m timeit`-style, or wrap `predict_arguments` in `time.time()`); windowed version should add only a few ms at 32×32. If it balloons, your window is too large or you accidentally left a dense L×L matrix.

**5c. Param delta.** `sum(p.numel() for p in m.decoder.local_dynamic_hi.parameters())` == 18,528.

**5d. Ablation that isolates the added value.** Three runs from the same seed: (i) no LD, (ii) LD@8×8 only (Approach 1), (iii) LD@8×8 + LD@32×32. Compare HAdobe5k PSNR/MSE. Multi-scale matching can inject noise — if (iii) < (ii) on HAdobe5k, reduce window size, add the smoothness regularizer (`smooth_lambda>0`, already in `trainer.py:128-133`), or drop this approach.

**5e. Window sweep.** `window ∈ {5,7,9}`. Larger windows cost more and can over-smooth; pick by HAdobe5k validation MSE.

**5f. Watch the right subset.** As in Approach 1, the target signal is the **HAdobe5k** per-subset metric in TensorBoard, not the iHarmony4 average. This approach specifically targets large/high-res foregrounds.

---

## 6. Risk

Highest tuning risk of the five (multi-scale correspondence is noisy). Keep it behind the same `i==1` guard so it's trivially removable, and only adopt if the ablation in §5d shows a clean HAdobe5k gain.
