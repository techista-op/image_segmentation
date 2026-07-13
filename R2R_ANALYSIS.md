# R2R — Current SOTA Analysis & Threat Assessment

**Paper:** Region-to-Region: Enhancing Generative Image Harmonization with Adaptive Regional Injection
**arXiv:** 2508.09746 (Aug 2025) — preprint, AAAI format, no confirmed venue
**Status:** ⭐ **#1 on iHarmony4. The leaderboard has not moved since Aug 2025.**
**Code, dataset, and weights: publicly released.**

---

## 1. Its numbers (iHarmony4, 256×256 — verified from Table 1)

| Subset | PSNR | MSE | fMSE |
|---|---|---|---|
| HCOCO | 42.62 | 7.93 | 146.81 |
| HAdobe5k | 42.22 | 15.70 | 102.14 |
| HFlickr | 38.04 | 26.28 | 185.38 |
| Hday2night | 39.42 | 22.64 | 497.19 |
| **All** | **41.94** | **12.51** | **144.38** |

Beats us by **~1.6 dB**. We do not beat SOTA. Full stop.

**Leaderboard context:**
R2R 41.94 → DiffHarmony++ 41.66 → DiffHarmony 40.97 → HDNet 40.46 → **us ~40.32** → AICT 39.99 → PCT-Net 39.85

---

## 2. What it costs — ✅ NOW MEASURED, NOT INFERRED

R2R **reports no parameter count anywhere.** The only figure in the paper is
Clear-VAE = SD-VAE (83.65M) + Adaptive Filter (+1.90M).

I measured the rest directly from their released HuggingFace checkpoints
(`1243asdad/region2region`), reading **file sizes from the HF API metadata** —
no download, no GPU required.

| Component | File size | Params |
|---|---|---|
| UNet (SD-1.5-inpainting) | 3.44 GB | **859.59 M** |
| Harmony Controller (ControlNet branch) | 1.66 GB | **414.37 M** |
| Clear-VAE | 0.17 GB | **83.70 M** |
| Text encoder (CLIP ViT-L/14) | 0.49 GB | 123.08 M |
| Projector | 0.01 GB | 1.77 M |
| **Core generative (UNet + Controller + VAE)** | | **1,357.7 M = 1.36 B** |
| **Full inference stack** | | **1,482.5 M = 1.48 B** |

**Dtype validation (three independent cross-checks — this is what makes the
number defensible):**
1. UNet → 859.59M at fp32, matching the known SD-1.5 UNet (859.5M). ✓
2. Text encoder → 123.08M at fp32, matching CLIP ViT-L/14 (123.06M). ✓
3. Clear-VAE → 83.70M at fp16, matching **the paper's own stated 83.65M**. ✓

**Us: 5.4M params, single forward pass.**
→ **R2R is ~251× larger** (core) / **~275×** (full stack).
→ Our earlier "~1B" estimate was an **underestimate**.

Also: **10 diffusion sampling steps** (iterative, not one pass); trains at 512²,
infers at 1024²; trained on 3× RTX 4090. **No FLOPs, no latency reported** —
same for DiffHarmony/++.

**Citation method for the paper (footnote it):** *"Parameter counts for
diffusion-based methods are not reported by their authors; we measure them from
the officially released checkpoints."* Same method gives Harmonizer 4.77M and
PCT-Net 4.81M.

---

## 3. 🎯 THE QUOTE — our paper's opening line

From R2R's own Conclusion:

> *"Considering the current limitation of model size, future work will explore lightweight diffusion models to improve efficiency."*

**The SOTA paper names its own size as the open problem. We are the answer to that sentence.** Lead the intro with this.

---

## 4. ⚠️ THE THREAT — RPHarmony (must be answered)

R2R built a new benchmark, **RPHarmony**, using **Random Poisson Blending**: instead of global color transfer (how iHarmony4 was made), it Poisson-blends a foreground into a *random region of a random reference image*, producing **local, spatially-varying disharmony**.

Result — **the entire color-transform / parameter-prediction family collapses:**

| Method | RPHarmony PSNR | MSE | fMSE |
|---|---|---|---|
| **R2R** | **36.32** | 40.25 | 192.66 |
| DiffHarmony++ | 36.03 | 42.16 | 203.45 |
| HDNet | 34.46 | 47.52 | 252.54 |
| **AICT** | **33.28** | 60.38 | 333.15 |
| **PCT-Net** | **33.26** | 60.39 | 332.61 |

They use this explicitly to argue color-transform methods generalize poorly to local disharmony. **A reviewer will cite this at us.** It is the sharpest attack on our family that exists.

### ⚠️ CRITICAL: their protocol is FINE-TUNE, not zero-shot

Verbatim, from R2R §Experiments (Datasets and Metrics):

> *"For the experiments on RPHarmony, we first initialized the models with weights pre-trained on iHarmony4 and then **finetune the model on RPHarmony**. Training and testing configurations were identical to those used for iHarmony4."*

So every competitor was **pre-trained on iHarmony4, then fine-tuned on RPHarmony's 12,787-image train split**, then tested. This is *not* a zero-shot generalization test.

**This matters enormously.** It means PCT-Net and AICT were *given the training data* and still landed at ~33 dB. The comfortable explanation ("they simply never saw local disharmony") is **wrong**. It is a claim about the function class, and we are in that function class.

**Eval protocol:** test at **1024×1024, results downsampled to 256×256** for metrics (following DiffHarmony). This is a *third* protocol, distinct from both the standard 256 eval and full-resolution eval. Match it exactly or the numbers are not comparable.

### Do NOT assume we are safe

An earlier version of this document claimed our per-pixel arguments were "precisely the defence." **That was wrong on two counts:**

1. **PCT-Net is already per-pixel** (a pixel-wise 3×3 affine colour transform). AICT predicts position-dependent LUTs. Both are spatially varying, and both still collapsed. "We are spatial, they are global" is **not** a valid distinction.
2. They were **fine-tuned**, so it is not mere distribution shift.

### The honest hypothesis (and our real rebuttal)

Our basis is **photographic** (brightness, contrast, saturation, temperature, highlight, shadow), not a generic affine colour map. Poisson-blended disharmony is largely a local *illumination* change, which our basis may express natively where an affine RGB map cannot. **That is a hypothesis, not a result.**

**The methodological criticism we CAN make:** RPHarmony's train split is only 12,787 images versus iHarmony4's 65,742 — a ~5× smaller fine-tuning budget. R2R and DiffHarmony++ carry a **Stable Diffusion prior trained on billions of images**, so adapting to 12.8k examples is trivial for them. PCT-Net and AICT have no such prior and must learn the new distribution nearly from scratch, from a fifth of the data. **The experiment conflates "our function class is better" with "we have a massive pretrained prior and you do not."** R2R does not disentangle these. That is a fair point to make in Discussion.

### Downloads (all public)

- **RPHarmony dataset (zip):** https://huggingface.co/1243asdad/region2region/blob/main/RPHarmony.zip
- **Code / README with all links:** https://github.com/anonymity-111/Region_to_Region
- **LDM checkpoint:** https://huggingface.co/1243asdad/region2region/tree/main/stable-diffusion-inpainting
- **Clear-VAE checkpoint:** https://huggingface.co/1243asdad/region2region/tree/main/clear_vae

Unzipped structure:
```
data/RPHarmony
  |- R-ADE20K/   (composite_images, masks, real_images)
  |- R-DUTS/
  |- train.jsonl / test.jsonl
  |- train.txt / test.txt
```
12,787 train / 1,422 test, built from DUTS + ADE20K, aesthetic-scorer filtered.

### → ACTION

**Optional, with real downside risk.** Run it *privately* first, then decide whether it goes in the paper. R2R is an **unvenued arXiv preprint (Aug 2025)** and RPHarmony is not an established benchmark, so omitting it is entirely defensible. We already pre-empt the criticism in our Limitations section.

Priority: **below** the per-subset table, the ablation, and the high-res HAdobe5k results.

---

## 5. Its soft spots (exploitable)

1. **Hday2night**: R2R (39.42) actually **loses** to DiffHarmony (39.45) and DiffHarmony++ (39.49), and has the worst fMSE there (497 vs 464/470). They blame the subset's small size (133 test images).
2. **MACA — their named contribution — is worth +0.02 dB.** Their own ablation: w/o Controller 41.76 → w/o MACA 41.92 → full 41.94. The Controller and Clear-VAE do all the work. On ccHarmony the full model is *worse* in PSNR than w/o-MACA (41.57 vs 41.66).
3. **No efficiency numbers whatsoever.** Cannot rebut an efficiency comparison with published data.
4. **Preprint, unvenued** (as of the last check). Cite as arXiv.

---

## 6. How we position against it

- ❌ **Never claim state of the art.**
- ✅ Concede R2R in the intro, immediately and cleanly. Do not let a reviewer do it for us. *(Done: Related Work and Limitations both concede it explicitly.)*
- ✅ Reframe on the **Pareto frontier**: *the last 1.5 dB currently costs ~250× the parameters and 10 sampling steps.*
- ✅ Quote their limitation sentence (§3). *(Done: it opens our Introduction.)*
- ✅ **Pre-empt RPHarmony in Limitations rather than ignore it.** *(Done — we raise it ourselves and leave the question open.)* Answer with results only if we run it.
- ⏸️ Params/FLOPs/latency table — **no harmonization paper has one**, and R2R cannot rebut it. Currently **out of scope** (we are not measuring FLOPs/latency/memory). Params measured from checkpoints are in `PARAM_MEASUREMENT.md` if we want to revive it.

---

**Related docs:** `LITERATURE_SURVEY.md` (full field, all numbers traced to primary sources), `PAPER_OUTLINE.md` (AAAI-27 structure + build order).
