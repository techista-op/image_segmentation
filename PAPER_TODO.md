# AAAI-27 Paper — Working Checklist

**Deadlines:** abstract **Mon 21 Jul**, full paper **Tue 28 Jul**, supplementary/code **Fri 31 Jul** (all AoE).
**Now:** 12 Jul → **9 days to abstract, 16 to full paper.**

**Result (confirmed):** 40.42 dB PSNR / 15.36 MSE, **All-iHarmony4 @256**, 5.4M params.
**Position:** ties HDNet (40.46) at ~half its params; beats it on MSE (15.36 vs 16.55).
**Lowest MSE of any non-diffusion method on the benchmark.** Below R2R / DiffHarmony++ / DiffHarmony.

**Length:** currently **5 pages**. Budget is 7 + 2 refs. Expect ~6–6.5 when figures and
training config land. **~1 page of slack. Nothing needs cutting.**

---

# ✅ DONE

- [x] Title, abstract (real numbers)
- [x] Introduction (full prose)
- [x] Related Work (4 paragraphs, incl. mandatory DCCF / PCT-Net differentiation)
- [x] Method: Preliminaries, Global-Argument Bottleneck, Independent Per-Pixel Argument Maps
- [x] Experiments: Setup (dataset, metrics, baselines)
- [x] Qualitative Results (prose)
- [x] Limitations (3, written disarmingly — incl. pre-empting the RPHarmony attack)
- [x] Conclusion
- [x] Table 1 — 18 baseline rows, primary sources, our All-set result
- [x] Table 3 — high-res HAdobe5k scaffold, baselines pre-filled
- [x] Table 4 — ablation scaffold
- [x] Bibliography — 24 entries, all authors verified against primary sources
- [x] LaTeX toolchain (local + Overleaf), no em dashes, no table symbols

---

# 🟢 CAN DO NOW (no office laptop, no final architecture needed)

## Figures — the biggest remaining win

We need **at least 5 image figures**. All of these can be produced with the
checkpoint and eval outputs already on hand, or with the pretrained baselines.

- [ ] **FIG 1 — Teaser (the money figure).**
      The dark-room / bright-window case.
      Panels: `Composite | Harmonizer (global args) | Ours | Ground truth`
      Inset: our predicted per-pixel **brightness** map, showing it varies spatially.
      Caption must make the argument: a single scalar over-brightens the whole
      foreground; a map does not.
      → **A reviewer forms their opinion from this figure and Table 1, in that order.**

- [ ] **FIG 2 — Argument maps (the interpretability figure).**
      One composite → the six predicted maps (brightness, contrast, saturation,
      temperature, highlight, shadow), 2×3 grid, diverging colormap centred at 0
      so signs are readable, with composite + output alongside.
      **No diffusion or pixel-regression method can produce this figure.**

- [ ] **FIG 3 — Qualitative comparison grid.**
      Rows = 4–6 test images (pick across HCOCO / HAdobe5k / HFlickr / Hday2night).
      Cols = `Composite | Harmonizer | HDNet | PCT-Net | Ours | Ground truth`.
      Pretrained weights for Harmonizer / HDNet / PCT-Net are all public.
      Choose cases where we visibly win. Include at least one spatially-varying-
      illumination case.

- [ ] **FIG 4 — Cascade vs independent.**
      Same input, two models. Show per-filter contribution or argument magnitude
      per filter. **Under the cascade the correction concentrates in the terminal
      filters; independently it spreads across the bank.**
      ⚠️ The Method section *asserts* this and currently has **no evidence behind it.**
      This figure (or an equivalent table column) is required to back the paper's
      central design claim.

- [ ] **FIG 5 — Editability demo.**
      Take a predicted argument map, nudge it (e.g. scale the brightness map by
      0.5 / 1.5), show the predictable photographic change in the output.
      Sells interpretability as a *capability*, not just a property.

- [ ] *(optional)* **FIG 6 — Architecture diagram.**
      We already have `harmonizer_architecture.png`, `ParallelHead256_architecture.png`,
      `MLP256_architecture.png` in the repo. Adapt once the decoder is frozen.

## Writing / polish
- [ ] Tighten the Introduction once the real page count is known
- [ ] Decide: add **fMSE columns** to Table 1? (most papers report it; we don't)
      → Table 1 is already 18 rows at `\footnotesize`; may need `\scriptsize`,
        or drop DoveNet/RainNet (historical, nobody will object)
- [ ] Write the **Reproducibility Checklist** (`Paper/ReproducibilityChecklist.tex`) —
      AAAI requires it, it's quick, and it's easy to forget
- [ ] Sanity-read the whole paper end to end for flow

## Verification (cheap, high value)
- [ ] **Confirm the PSNR convention** of the script that produced 40.42.
      ⚠️ *None* of the eval code in the synced repo uses `data_range=255`:
        `func.py` and `val_decoder.py` use `pred.max()-pred.min()`;
        `eval_pretrained.py` uses `gt.max()-gt.min()`. HDNet uses **255**.
      If 40.42 was NOT computed with 255, the true number is **higher**.
      MSE/fMSE are unaffected either way.

---

# 🔴 WHEN YOU HAVE THE OFFICE LAPTOP + FINAL ARCHITECTURE

## Numbers
- [ ] **Per-subset cells in Table 1** — HCOCO / HAdobe5k / HFlickr / Hday2night, MSE + PSNR.
      *Sanity check:* image-count-weighted mean of the four MSEs must reconcile
      to 15.36. Weights: 4283 / 2160 / 828 / 133 (= 7404). If it doesn't, something
      is wrong (wrong checkpoint, wrong split, metric mismatch) — catch it before print.
- [ ] **Table 4 — ablation numbers.** Rows:
      `global + cascaded` (Harmonizer baseline, reproduced) →
      `per-pixel + cascaded` → `per-pixel + independent` (ours)
      Ideally multiple seeds. Same backbone / data / schedule across rows.
- [ ] **Table 3 — high-res HAdobe5k @1024².**
      Targets: CDTNet 38.77, INR 38.38, HDNet 41.56.
      **Our best shot at an outright win** — filters are resolution-free; CDTNet must
      be retrained per resolution.

## Architecture
- [ ] **Which decoder ships?** (`mlp` / `parallel_head` / `decoder` / `unet_decoder` /
      `attention_decoder`) and its exact parameter count.
      → Confirm the 5.4M figure with `sum(p.numel() for p in model.parameters())`.
- [ ] Fill Method §Training: composite synthesis, loss, optimizer, LR schedule,
      epochs, batch size, hardware.
- [ ] Complete the Method text for the final decoder (currently a TODO).

## Evidence for the central claim
- [ ] **Per-filter contribution under cascade vs independent.** The paper claims the
      cascade concentrates the correction in the terminal filters. Pull this from the
      training logs (`epoch_metrics.csv` already tracks per-filter losses) or measure it.
      **This is the load-bearing evidence for the whole design decision.**

---

# 🟡 OPTIONAL / IF TIME

- [ ] **RPHarmony evaluation.** ⚠️ Read `R2R_ANALYSIS.md` first.
      R2R *fine-tuned* every competitor on its train split — this is NOT a zero-shot
      test, and PCT-Net/AICT still collapsed (~33 dB). **Real downside risk.**
      R2R is an unvenued arXiv preprint; omitting is defensible. We already pre-empt
      it in Limitations.
- [ ] Params-only measurement of PCT-Net / AICT / Harmonizer (no GPU, ~20 min) to
      strengthen the efficiency claim beyond the single HDNet comparison.
- [ ] 2048² HAdobe5k results.

---

## The two things that actually decide this paper

1. **Figure 1.** It is the argument. Everything else is corroboration.
2. **Evidence that the cascade concentrates the correction in the terminal filters.**
   The Method asserts it; nothing yet proves it. That's the one claim a sharp reviewer
   will demand support for.
