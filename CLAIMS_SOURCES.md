# Introduction — Claim-by-Claim Source Trace

Every factual assertion in the two paragraphs below, with its source, what that
source actually says, and how the number was obtained.

**Three tiers, and the distinction matters:**

| Tier | Meaning |
|---|---|
| 🟢 **REPORTED** | Stated by the cited paper's own authors. Safe. |
| 🟡 **THIRD-PARTY** | Reported by a *later* paper about an *earlier* one. Cite accordingly. |
| 🔴 **OURS** | We derived or measured it. **Must be defensible under challenge.** |

---

# PARAGRAPH 1 — the two expensive lines

### Claim: "Since the introduction of the iHarmony4 benchmark (Cong et al. 2020)"
🟢 **REPORTED.**
**Source:** DoveNet, CVPR 2020 — https://arxiv.org/abs/1911.13239
**What it says:** Introduces iHarmony4, the first large-scale harmonization benchmark. Four sub-datasets (HCOCO, HAdobe5k, HFlickr, Hday2night), 65,742 train / 7,404 test pairs. Establishes the 256×256 evaluation protocol the field still uses.

---

### Claim: "region-aware normalization (Ling et al. 2021)"
🟢 **REPORTED.**
**Source:** RainNet, CVPR 2021 — https://arxiv.org/abs/2106.02853
**What it says:** Introduces RAIN, which normalizes foreground activations using foreground statistics and then re-affines them with scale/bias learned from the *background* region. Frames harmonization as background-to-foreground style transfer.

### Claim: "domain codes (Cong et al. 2021)"
🟢 **REPORTED.**
**Source:** BargainNet, ICME 2021 — https://arxiv.org/abs/2009.09169
**What it says:** A domain-code extractor maps a masked region to a vector; triplet losses pull the harmonized foreground's code toward the background's code.

### Claim: "intrinsic decomposition (Guo et al. 2021b)"
🟢 **REPORTED.**
**Source:** Intrinsic Image Harmonization, CVPR 2021 — https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Intrinsic_Image_Harmonization_CVPR_2021_paper.pdf
**What it says:** Decomposes the composite into reflectance and illumination (Retinex-style), harmonizes each separately, recomposes.

### Claim: "attention (Guo et al. 2021a; Hang et al. 2022)"
🟢 **REPORTED.**
**Sources:**
- HT / D-HT, ICCV 2021 — https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Image_Harmonization_With_Transformer_ICCV_2021_paper.pdf
- SCS-Co, CVPR 2022 — https://arxiv.org/abs/2204.13962
**What they say:** Transformer-based harmonization; and BAIN, an attention-weighted background-statistics normalization plus a style-contrastive objective.

### Claim: "explicit foreground-to-background correspondence (Chen et al. 2023; Zhu et al. 2022; Shen et al. 2023)"
🟢 **REPORTED.**
**Sources:**
- HDNet, ACM MM 2023 — https://arxiv.org/abs/2211.08639
- Zhu et al., arXiv 2022 — https://arxiv.org/abs/2204.04715
- GKNet, ICCV 2023 — https://arxiv.org/abs/2305.11676
**What they say:** HDNet: cosine-similarity K-nearest-neighbour matching between foreground and background features at the bottleneck. Zhu et al.: soft attention matching foreground locations to background locations (LTL) and patches (PTL). GKNet: long-distance reference selection to modulate local kernels.

---

### ⚠️ Claim: "These models are large, typically 40 to 67M parameters"
🟡 **THIRD-PARTY. Handle with care.**

**Where the numbers come from:** HDNet (ACM MM 2023), **Table 1**. Also reproduced in S²CRNet's table.

| Method | Params |
|---|---|
| IntrinsicIH | 40.86M |
| DIH | 41.76M |
| RainNet | 54.75M |
| DoveNet | 54.76M |
| BargainNet | 58.74M |
| S²AM | 66.70M |

Range: **40.86M – 66.70M** → "40 to 67M". ✅ The claim is accurate.

**⚠️ The catch:** **None of those six papers reports its own parameter count.** Every figure traces to HDNet's third-party table. If challenged, the honest answer is *"as reported by Chen et al. (2023)."*

**⚠️ A known error in that same table:** HDNet lists **Harmonizer as 21.70M**. That is Harmonizer's **21.7 MB** model-file size mislabelled as parameters. The true count is **~4.77M** (measured from the released checkpoint). We do **not** repeat this error — we cite the range only for the *large* U-Net methods, all of which genuinely are 40–67M.

---

### Claim: "the best of them now reach roughly 40 dB PSNR"
🟢 **REPORTED.**
**Source:** HDNet, arXiv 2211.08639, Table 1. All-iHarmony4 @256: **40.46 dB / 16.55 MSE**. Highest non-diffusion result published.

---

### Claim: "fine-tunes a latent diffusion model (Zhou, Feng, and Wang 2024; Zhou et al. 2024; Zhang et al. 2025)"
🟢 **REPORTED.**
**Sources:**
- DiffHarmony, ICMR 2024 — https://arxiv.org/abs/2404.06139
- DiffHarmony++, ACM MM 2024 — https://doi.org/10.1145/3664647.3681466
- R2R, arXiv 2025 — https://arxiv.org/abs/2508.09746
**What they say:** All three adapt Stable Diffusion to harmonization. DiffHarmony++ and R2R are principally concerned with repairing the high-frequency detail destroyed by the SD VAE (Harmony-VAE and Clear-VAE respectively).

---

### Claim: "R2R attaining 41.94 dB"
🟢 **REPORTED.**
**Source:** R2R, Table 1 (All-iHarmony4 @256): **PSNR 41.94 / MSE 12.51 / fMSE 144.38**. Current state of the art; the leaderboard has not moved since Aug 2025.

---

### 🔴 Claim: "a Stable-Diffusion backbone of roughly 1.4 billion parameters"
🔴 **OURS. R2R reports NO parameter count anywhere.**

**How we obtained it.** We read file sizes from the HuggingFace API for R2R's released checkpoints (`1243asdad/region2region`) and divided by bytes-per-parameter. **No download, no GPU.**

| Component | File size | dtype | Params |
|---|---|---|---|
| UNet (SD-1.5-inpainting) | 3.44 GB | fp32 | 859.59 M |
| Harmony Controller (ControlNet) | 1.66 GB | fp32 | 414.37 M |
| Clear-VAE | 0.17 GB | fp16 | 83.70 M |
| CLIP text encoder | 0.49 GB | fp32 | 123.08 M |
| Projector | 0.01 GB | fp32 | 1.77 M |
| **Core generative (UNet + Controller + VAE)** | | | **1,357.7 M** |
| **Full inference stack** | | | **1,482.5 M** |

**Why the dtypes are trustworthy — three independent cross-checks:**
1. UNet → 859.59M at fp32 = the known SD-1.5 UNet (859.5M) ✓
2. Text encoder → 123.08M at fp32 = CLIP ViT-L/14 (123.06M) ✓
3. Clear-VAE → 83.70M at fp16 = **R2R's own stated 83.65M** ✓

The third is decisive: their own reported figure validates our method.

**⚠️ WORDING RISK — fix before submission.** "Roughly 1.4 billion" sits *between* our core figure (1.36B) and our full-stack figure (1.48B). Worse, the word **"backbone"** could be read as the UNet alone (859M), which would make the sentence wrong.

**Recommended fixes (pick one):**
- *"a Stable-Diffusion stack of roughly 1.36 billion parameters"* (core: UNet + Controller + VAE — conservative and precise)
- *"a Stable-Diffusion stack exceeding 1.3 billion parameters"* (safe under either reading)

Full derivation and error bounds: **`PARAM_MEASUREMENT.md`**.

---

### Claim: "evaluated over ten iterative sampling steps"
🟢 **REPORTED.**
**Source:** R2R, Implementation Details. Euler-ancestral sampler, **10 sampling steps**. Trains at 512², infers at 1024².

---

### Claim: R2R's conclusion — *"considering the current limitation of model size, future work will explore lightweight diffusion models to improve efficiency"*
🟢 **REPORTED — direct quote.**
**Source:** R2R, Conclusion. https://arxiv.org/html/2508.09746v1
**Why it matters:** The state-of-the-art paper names its own model size as the open problem. This is the strongest sentence in our introduction and it is *theirs*.

---

# PARAGRAPH 2 — the parameter budget

### Claim: "HDNet comprises 10.41M parameters"
🟢 **REPORTED.**
**Source:** HDNet, Tables 1 and 4. (Table 1 mislabels the unit as "MB"; Table 4 uses "M" alongside DoveNet 54.76M and S²AM 66.70M, which are standard parameter counts. It is parameters, not megabytes.)

---

### 🔴 Claim: "of which 9.76M, some 94%, form a conventional U-Net backbone; the two modules it introduces account for the remaining 0.65M"
🔴 **OURS.** HDNet reports only the 10.41M total. **The breakdown is our analytic recount from their released code.**

**Source code:** https://github.com/chenhaoxing/HDNet (`models/networks.py`, `models/att.py`, `models/drconv.py`; `ngf=32`, `input_nc=4`, `output_nc=3`)

| Component | Params |
|---|---|
| U-Net backbone (16 conv/deconv layers) | **9.7596 M** |
| Local Dynamic module (one `Conv1d(512→256, k=1)`) | **0.1313 M** |
| Mask-aware Global Dynamic ×3 (`DRConv2d` at C=256/128/64) | **0.5188 M** |
| **Total** | **10.4097 M** ✅ |

Matches their reported 10.41M to four significant figures — which is what makes the breakdown credible.

**Percentage:** 9.7596 / 10.4097 = **93.75%** → "some 94%". ✅
**Contributed modules:** 0.1313 + 0.5188 = **0.65M**. ✅

**Why this is worth the risk of being "ours":** it shows HDNet's headline "80% parameter reduction" comes almost entirely from halving the channel width (ngf 64→32), not from its architectural contribution. **Their own Base ablation row corroborates it:** the 9.76M U-Net alone reaches 38.53 dB.

**If challenged:** the recount is reproducible from their public code in minutes. Consider stating it as *"a parameter count of the released implementation shows..."* to make the provenance explicit.

---

### Claim: "Zhu et al. enlarge their correspondence architecture from 9.48M to 117.42M parameters, a factor of twelve, and gain 0.60 dB"
🟢 **REPORTED — their own table.**
**Source:** Zhu et al., arXiv 2204.04715, **Table 5** (parameter-scaling study).

| Channels | Params | PSNR | MSE | fMSE |
|---|---|---|---|---|
| C | **9.48M** | **37.97** | 25.16 | 282.69 |
| 2C | 37.81M | 38.35 | 23.24 | 261.34 |
| 3C | 72.39M | 38.45 | 22.83 | 256.69 |
| 4C | **117.42M** | **38.57** | 21.43 | 248.12 |

**Arithmetic:** 117.42 / 9.48 = **12.4×** → "a factor of twelve" ✅
38.57 − 37.97 = **0.60 dB** ✅

This is the single best citation in the paragraph: **the authors published it themselves**, and it makes our argument for us.

---

## Summary — what would survive a hostile reviewer

| Claim | Tier | Risk |
|---|---|---|
| iHarmony4, all method attributions | 🟢 | none |
| R2R 41.94 dB, 10 sampling steps, the quote | 🟢 | none |
| HDNet 40.46 dB, 10.41M total | 🟢 | none |
| Zhu et al. 9.48M→117.42M, +0.60 dB | 🟢 | none — **their own table** |
| "40 to 67M parameters" | 🟡 | low — accurate, but sourced from HDNet's table, not the originals |
| **"roughly 1.4 billion parameters"** | 🔴 | **⚠️ fix the wording — see above** |
| **HDNet's 94% / 0.65M split** | 🔴 | low — reproducible from public code, matches their reported total exactly |

**One action item:** change *"backbone of roughly 1.4 billion parameters"* to *"stack of roughly 1.36 billion parameters"* (or *"exceeding 1.3 billion"*). Everything else stands.
