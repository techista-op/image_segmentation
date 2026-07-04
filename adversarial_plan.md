# Adversarial Discriminator for Image Harmonization — Research Summary & Implementation Plan

## 1. Context

**Current architecture**: Harmonizer-based decoder branch predicting a 256×256×6
spatial filter-argument map, applied via white-box filters, bicubic-upsampled to
original resolution.

**Current best result**: 40.32 PSNR / 15.5 MSE on iHarmony4 @ 256×256 eval.

**Baseline (GiftNet's reproduction of Harmonizer)**: 37.84 PSNR / 24.26 MSE.

**Target (R2R, LDM-based SOTA)**: ~41.94 PSNR / ~12 MSE.

**Diagnosed failure mode**: global/pooled foreground-background conditioning breaks
on spatially non-uniform backgrounds — e.g. a subject standing in a dark room next
to a bright window gets incorrectly brightened, because the background feature
pool is dragged upward by the window even though the locally-relevant background
(the dark room) should push the correction the other way. HAdobe5k (large,
complex foregrounds) is the weakest subset relative to HDNet.

**Primary fix in progress**: LTL/PTL (Locations-to-Location / Patches-to-Location
Translation, from Zhu et al., "Image Harmonization by Matching Regional
References") — running now on a separate GPU. This is the higher-confidence fix
for the diagnosed problem.

**This document**: scopes adversarial training as a secondary, complementary
refinement, per mentor request, to be layered on top of the LTL/PTL result.

---

## 2. What the research shows — summary of prior art

### 2.1 Two discriminator philosophies in this subfield

1. **Realism discriminators** — judge "does this look like a real photo" (whole
   image or patches), independent of fg/bg relationship. This is the generic
   GAN/PatchGAN approach borrowed from image-to-image translation.
2. **Relational / domain-verification discriminators** — judge whether the
   foreground and background *belong to the same domain* (lighting, color,
   tone), i.e. explicitly compare fg vs bg rather than judging either region's
   realism in isolation. This is the harmonization-specific innovation,
   pioneered by DoveNet.

The second is the better-motivated fit for our diagnosed failure mode, since a
region-independent realism discriminator can be fooled by a case where the fg
looks plausible on its own, the bg looks plausible on its own, but the two are
mismatched relative to each other (exactly our window/dark-room case).

### 2.2 Method-by-method findings

- **DoveNet (Cong et al., CVPR 2020)** — introduced the domain-verification
  discriminator: judges whether foreground and background patches come from the
  same "domain." This is the closest precedent to what we want. Their own
  ablation shows the adversarial/domain-verification component gives a **real
  but modest** gain over the non-adversarial baseline — a refinement, not a
  primary driver of their result.
- **BargainNet** — background-guided domain code extractor; uses a domain
  similarity loss (not a classic discriminator) to pull fg domain code toward
  bg domain code. Relational in spirit, but not implemented as an adversarial
  game — a useful alternate mechanism to keep in mind if GAN training proves
  unstable.
- **PHDNet / S²AM / RainNet / AIC-Net / S²CRNet / SCS-Co** — architectural or
  contrastive approaches to fg/bg consistency; several avoid adversarial
  training entirely and instead use attention, normalization matching (AdaIN/
  RAIN-style), or contrastive losses to enforce fg/bg domain alignment without
  the instability of a GAN.
- **Huang et al. (video harmonization)** — pixel-wise disharmony discriminator,
  notable for removing the need for an explicit fg mask by having the
  discriminator itself localize disharmony. Relevant if we ever want the
  discriminator to also produce a spatial disharmony map for diagnostics.
- **Diffusion-based methods (DiffHarmony, R2R)** — don't use classic GAN
  discriminators; they get realism from the generative prior of the diffusion
  process itself. Not directly transferable to our filter-regression
  architecture.
- **Non-adversarial SOTA (Harmonizer, PCT-Net, DCCF, CDTNet)** — notably, most
  of the strongest recent results on iHarmony4 do **not** use adversarial loss
  at all. They win through better feature-transfer/regression design. This is
  the field's main signal that adversarial training is a secondary refinement,
  not the primary lever for closing a PSNR/MSE gap.

### 2.3 Challenges reported in the literature, and mitigations

- **Fidelity vs. realism tension**: adversarial loss optimizes for perceptual
  plausibility, not ground-truth fidelity, and can directly conflict with
  PSNR/MSE — well documented across restoration literature generally (not
  harmonization-specific, but directly applicable). Mitigation: low adversarial
  loss weight (relative to pixel loss), and treating pixel/reconstruction loss
  as the dominant term throughout training.
- **Global discriminators miss local fg/bg mismatch**: a discriminator judging
  the whole image, or fg/bg regions independently, can be fooled by cases where
  each region is locally realistic but mismatched relative to each other.
  Mitigation used in the field: mask-conditioning (feed the discriminator the
  mask so it has access to the fg/bg boundary), patch-level discrimination
  (judge many local regions rather than one global verdict, increasing the
  chance a boundary-straddling patch catches the mismatch), or explicit
  relational/dual-branch comparison (DoveNet's actual approach — compare fg and
  bg domain codes/features directly rather than relying on an implicit signal).
- **Training instability** (mode collapse, discriminator overpowering
  generator): standard GAN-training issues, generally mitigated by using
  LSGAN (least-squares) loss instead of vanilla BCE GAN loss for training
  stability, keeping the discriminator from training too far ahead of the
  generator (e.g. by loss weighting or update-ratio control), and starting
  from a pretrained, already-good generator (which we have) rather than
  training the GAN from scratch.
- **Artifacts / color shift / texture hallucination**: reported failure mode
  when adversarial weight is too high — the generator starts producing
  "realistic-looking" but factually wrong corrections to satisfy the
  discriminator. Mitigation: kill-switch monitoring of PSNR/MSE every eval, and
  keeping adversarial weight low enough that this class of failure doesn't
  dominate.

### 2.4 Best-validated design, synthesized

Given the above, the best-supported design for our specific problem (catching
fg/bg lighting/color inconsistency, not general realism) is:

- **DoveNet-style domain-verification framing**, adapted to a **single-stream,
  mask-conditioned PatchGAN** for implementation speed (full two-branch fg/bg
  comparison is more faithful to DoveNet but costs more implementation time
  than our 2-week window supports).
- **LSGAN loss**, not vanilla BCE, for stability.
- **Low starting adversarial weight (~0.01× main loss)**, escalate only if
  stable and PSNR/MSE hold.
- **Attach point: final harmonized image** (post-filter, post-bicubic-upsample)
  — the only point where a direct comparison to real GT images is meaningful.
- **Treat as a refinement on top of the LTL/PTL checkpoint**, not a standalone
  competing experiment — matches how DoveNet itself used it (as one component
  of an already-reasonable generator), and matches the field's overall
  preference for architectural fixes over adversarial ones as the primary
  lever.

---

## 3. Key references

- Cong, W. et al. "DoveNet: Deep Image Harmonization via Domain Verification."
  CVPR 2020. https://arxiv.org/abs/1911.13239
- Cong, W. et al. "BargainNet: Background-Guided Domain Translation for Image
  Harmonization." ICME 2021.
- Huang, H. et al. "Temporally Coherent Video Harmonization Using Adversarial
  Networks."
- Cun, X. & Pun, C.M. "Improving the Harmony of the Composite Image by
  Spatial-Separated Attention Module (S²AM)." IEEE TIP 2020.
  https://arxiv.org/abs/1907.06406
- Ling, J. et al. "Region-aware Adaptive Instance Normalization for Image
  Harmonization (RainNet)." CVPR 2021.
- Zhu, Z. et al. "Image Harmonization by Matching Regional References"
  (LTL/PTL — our primary fix, running in parallel).
- Ke, Z. et al. "Harmonizer: Learning to Perform White-Box Image and Video
  Harmonization." ECCV 2022. https://arxiv.org/abs/2207.01322
- Guerreiro, J. et al. "PCT-Net: Full Resolution Image Harmonization Using
  Pixel-Wise Color Transformations." CVPR 2023.
- Xue, B. et al. "DCCF: Deep Comprehensible Color Filter Learning Framework
  for High-Resolution Image Harmonization." ECCV 2022.
  https://arxiv.org/abs/2207.04788

---

starting checkpoint to use.
```
