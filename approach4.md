# Approach 4 — Mask-aware Dynamic convolution (MGD-style region kernels)

> Gives the decoder region-conditioned capacity (separate fg/bg filters). Ablation
> gain overlaps Approach 1's LD, so expect diminishing returns after LD. Moderate cost.
> Optional / lower priority.

---

## 0. Reality check

- See `approach1.md §0`. The mask must be downsampled to each decoder stage's resolution to split fg/bg.
- This does **not** add correspondence; it adds per-region filter capacity. It is complementary to (not a substitute for) Approach 1.
- Apply to **one or two decoder stages only**. Converting every conv to dynamic is where CondConv/DyConv blow up in params (per Chen et al. CVPR2020: CondConv K=8 → 27.5M on MobileNetV2 vs ~3.5M static). We predict just **2** kernel sets (fg, bg), which is far cheaper.
- Decoder stage channels (from `module.py:99-110`): stage0 out=128@16×16, stage1 out=96@32×32, stage2 out=64@64×64. Good target: **stage2 (64 ch @64×64)** — mid-resolution, moderate channel count.

---

## 1. New module — mask-aware dynamic conv

Add to `src/train/harmonizer/module/module.py` after `LocalDynamic` (or after `Decoder`). This wraps a conv whose weights are modulated per region by a lightweight generator (adaptive-pool → two 1×1 convs), then blends fg/bg results by the mask.

```python
class MaskAwareDynamicConv(nn.Module):
    """Region-specialised conv: distinct modulation for fg vs bg, blended by mask.

    Cheaper than DRConv/CondConv: instead of N candidate kernels we keep ONE base
    conv and predict two per-channel modulation vectors (fg, bg) from pooled context.
    F' = (conv(F) * gamma_fg) ⊗ M + (conv(F) * gamma_bg) ⊗ (1-M)
    """
    def __init__(self, channels):
        super(MaskAwareDynamicConv, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.gen = nn.Sequential(                      # filter/attention generator
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 2, 1),  # -> [gamma_fg ; gamma_bg]
        )

    def forward(self, x, mask):
        n, c, h, w = x.shape
        m = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        y = self.bn(self.conv(x))
        g = self.gen(x)                                # [n, 2c, 1, 1]
        g_fg, g_bg = torch.sigmoid(g[:, :c]), torch.sigmoid(g[:, c:])
        out = y * g_fg * m + y * g_bg * (1 - m)
        return F.relu(x + out, inplace=True)           # residual keeps it stable
```

Using per-channel modulation (not full dynamic kernels) keeps params ~O(c²/4) instead of O(N·c²·9). It captures "fg and bg get different treatment" — the MGD intent — at a fraction of the cost.

---

## 2. Wire into the decoder

**File:** `src/train/harmonizer/module/module.py`, `Decoder.__init__`:
```python
        self.mgd = MaskAwareDynamicConv(64)     # matches stage2 out_channels
```

**`Decoder.forward`** — apply after `stages[2]` (64×64, 64 ch):
```python
        for i in range(self.num_stages):
            x = self.stages[i](x, skips[i])
            if mask is not None and i == 2:      # after stages[2] -> 64x64, 64ch
                x = self.mgd(x, mask)
        return x
```
Guard: only valid when `num_stages > 3` (terminal_size ≥ 128). For terminal_size=256 (`num_stages=5`) fine. If you also run smaller `terminal_size` ablations, wrap construction/use in a `self.num_stages > 3` check.

Mask plumbing into `Decoder.forward` / `predict_arguments` is the same edit as `approach1.md §3b/§3c`.

---

## 3. Parameter budget

For `MaskAwareDynamicConv(64)`:
- `conv` 3×3: `64*64*9 = 36,864`
- `bn`: `128`
- `gen`: `64*16 + 16` + `16*128 + 128` = `1,024 + 16 + 2,048 + 128 = 3,216`
- Total ≈ **0.04M** at stage2.

At stage0 (128 ch) it'd be ~`128*128*9≈0.15M`. Report's "~0.3–1M" corresponds to applying it at higher channel widths / multiple stages. Staying at stage2 keeps it ~0.04M. New total ≈ **5.44M**.

If you want the literal MGD (two full 3×3 kernels rather than per-channel modulation), replace `gen` with a DRConv-style kernel generator; that pushes cost to ~0.3–1M — only do this if the cheap modulation version shows promise.

---

## 4. Mirror to inference copy

Apply §1–§2 to `src/model/module.py`. Sync per `approach1.md §5`.

---

## 5. Verification

**5a. Shape/NaN test.** `approach1.md §7a` harness; assert decoder output unchanged shape and finite. Test all-fg and all-bg masks (the `m`/`1-m` blend must not NaN — it can't here since there are no divisions).

**5b. Param delta.** `sum(p.numel() for p in m.decoder.mgd.parameters())` ≈ 40,208.

**5c. Region-specialization sanity.** Feed an input, log `g_fg` vs `g_bg` vectors — they should diverge during training (if they stay identical, the module is degenerate and adds nothing; consider increasing generator capacity or moving to an earlier stage).

**5d. Overlap ablation (important).** HDNet's own numbers show MGD's gain largely overlaps LD's. Run: (i) LD only (Approach 1), (ii) LD + MGD. If (ii) doesn't beat (i) on HAdobe5k, drop MGD — you're paying params for redundant capacity. This is the explicit decision gate for this approach.

**5e. Overfit one batch + HAdobe5k subset metric.** As in prior approaches.

---

## 6. Recommendation

Only add this if you have budget headroom after Approaches 1 and 5 and want to mirror full HDNet. Gate adoption strictly on the §5d overlap ablation — its expected marginal value over LD alone is small.
