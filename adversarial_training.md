# Adversarial Discriminator — Integration Plan (grounded in the actual code)

Companion to `ADVERSARIAL_DISCRIMINATOR_PLAN.md` (the research summary + design rationale)
and structured like `ARCHITECTURE_LTL_PTL_INTEGRATION.md`. This maps the Section-2.4 design
(mask-conditioned, single-stream PatchGAN, LSGAN loss, low weight, attached to the final
image, layered on the LTL/PTL checkpoint) onto the real files on the `decoder` branch.

**Plan only — no code written.** All paths/line refs were read from source. This is a
**separate, flagged, second-GPU experiment**; the existing non-GAN pipeline must remain
runnable unchanged.

---

## Part A — Current architecture (relevant parts, as found)

### A.1 Where the final harmonized image is produced (the adversarial attach point)

- `Harmonizer.restore_image` (`src/train/harmonizer/module/harmonizer.py:82-98`): upsamples the
  6 argument maps to the composite resolution (bicubic, `:88-93`) and applies the white-box
  filters via `FilterPerformer.restore`, returning `outputs[-1]` — the **final harmonized
  image, post-filter, post-upsample** (`:97-98`).
- Proxy `model.py:forward` (`src/train/harmonizer/model.py:46-55`) calls
  `predict_arguments` then `restore_image` and stores `resulter['outputs'] = pred`.
- In the trainer this surfaces as `pred_image = tool.dict_value(resulter, 'outputs')`
  (`trainer.py:93`), shape `[N, 3, H, W]`. During training everything runs at **256×256**
  (composite is resized to `input_size=(256,256)` at `harmonizer.py:68`).

**→ The adversarial loss attaches to `pred_image` (fake) vs the GT real image, both 256×256,
with the mask as conditioning.** This is the only point where a direct real-vs-generated
comparison is meaningful, matching Section 2.4.

### A.2 Training loop / optimizer / loss combination

- Loop: `HarmonizerTrainer._train` (`trainer.py:74-191`). Per step: `_batch_prehandle`
  (`:84`) → forward under bf16 autocast (`:89-91`) → losses → `loss.backward()` (`:136`) →
  `self.optimizer.step()` (`:137`).
- **Single optimizer** built in `_build` (`trainer.py:50`) from `self.model.module.param_groups`
  (defined in `model.py:35-44`: backbone / regressor / performer / decoder groups). One Adam,
  one `zero_grad`/`step` per iteration.
- Loss: `HarmonizerLoss` = foreground-masked MSE on the final image (`criterion.py:17-38`).
  Combined as `loss = fine_loss + coarse_loss (+ smooth_loss)` (`trainer.py:104,117,123,132`),
  where `fine`/`coarse` are **labeled vs additional-data** partitions, not stages.

**→ Adding a GAN means a second optimizer (discriminator) + an alternating update, and adding
`λ_adv · L_adv^G` into the generator loss at `trainer.py:123`. Must be gated by a flag so the
default path is untouched.**

### A.3 Current per-subset eval + TensorBoard (the known gap)

Same finding as the LTL doc: `func.py:16-41` computes MSE/fMSE/PSNR/SSIM, but the trainer
calls it with a hardcoded `id_str='IH'` (`trainer.py:227`), so metrics are **aggregated over
all four subsets** — HCOCO/HFlickr/HAdobe5k/Hday2night are not separated, because the dataset
never returns its `subset` (known at `data.py:140`) from `__getitem__`. TensorBoard tags today:
`Loss/train|fine_loss|coarse_loss`, `Loss/val`, `Metrics/MSE|fMSE|PSNR|SSIM`, `LR/lr`,
`Args/*`, `Images/comparison_grid` (`trainer.py:167-170,264-277`).

**→ The kill-switch and the "does it move HAdobe5k?" question both require per-subset PSNR/MSE
first. That code change (Part C.3) is a prerequisite, not optional.**

### A.4 Main-loss magnitude (needed to scale λ_adv — do not hardcode)

The generator's reconstruction loss is masked MSE in `[0,1]` space. At the current
~40.32 PSNR the implied per-image MSE ≈ `10^(-40.3/10) ≈ 9.3e-5`; `fine_loss` typically sits
in the **~1e-4** range at this stage. **Read the real number** from
`checkpoints/<log_tag>/training_log.json` (`train_fine_loss`, written at `trainer.py:286`) or
the `Loss/fine_loss` scalar before setting the weight — see Part B.4.

### A.5 Existing discriminator scaffolding (reuse it)

There is prior work compiled on this branch (source lives on the `discriminator` git branch;
only `.pyc` are present here):
- `src/discriminator/patchgan.py` → `class PatchGAN(in_channels, base, n_layers,
  use_spectral_norm)` with a `_sn(...)` spectral-norm helper. Standard PatchGAN geometry
  (conv `k=4,s=2,p=1`, `LeakyReLU(0.2)`), optional spectral norm.
- `src/discriminator/train_discriminator.py` → `RealVsCompositeDataset`, `build_loader`,
  **`lsgan_d_loss(score, target_is_real)`**, **`patch_accuracy(score, target_is_real)`**,
  `main`. CONFIG shows a **Phase-0 pretrain**: real-vs-composite classification, `im_size=256`,
  `batch=32`, `epochs=30`, `lr=2e-4`, `betas=(0.5,0.999)`, `use_spectral_norm=True`,
  `exp_id='disc_phase0_real_vs_composite'`.

**→ Do not rewrite from scratch. Port `PatchGAN`, `lsgan_d_loss`, `patch_accuracy` from the
`discriminator` branch into the `decoder` branch, adapt `PatchGAN` in_channels for
mask-conditioning, and reuse the Phase-0 pretrain as an optional warm start for D.**

---

## Part B — Integration plan

### B.1 Discriminator module (new / ported file)

**File:** `src/discriminator/patchgan.py` (port from the `discriminator` branch; keep it
isolated from generator code, per the prompt).

- **Single-stream, fully-convolutional PatchGAN**, mask-conditioned:
  `in_channels = 3 (RGB) + 1 (mask) = 4`. Fake input = `cat(pred_image, mask)`; real input =
  `cat(gt_image, mask)` — identical mask both sides so D judges the **fg/bg relationship**, not
  just realism (Section 2.3 mask-conditioning mitigation).
- Geometry: default `base=64, n_layers=3` → ~70×70 receptive field, output score map
  `[N, 1, ~30, ~30]`. Keep `use_spectral_norm=True` (stabilizes D; already the Phase-0 default).
- No sigmoid on the output (LSGAN uses raw scores).

**Approx params:** standard 70×70 PatchGAN at base=64 ≈ **2.7M**. This is a **training-only**
network — it does **not** count against the 10M generator budget and is discarded at inference.

**Approx VRAM @ train, per sample 256×256:** activations for a 4→64→128→256→512→1 conv stack
at halving resolutions ≈ **tens of MB**; negligible vs the generator + autograd graph.

### B.2 Where it attaches in the forward/loss path

Attach at `pred_image` (A.1). No change to the generator's architecture or forward — the GAN
only adds loss terms and a second network. Concretely, in `_train`:
- **D step:** `d_real = D(cat(gt, mask))`, `d_fake = D(cat(pred_image.detach(), mask))`;
  `L_D = 0.5·(lsgan_d_loss(d_real, True) + lsgan_d_loss(d_fake, False))`; backward on D
  optimizer only.
- **G step:** add `L_adv^G = mse(D(cat(pred_image, mask)), 1)` (LSGAN generator target = real)
  to the existing generator loss with weight `λ_adv` (B.4). `pred_image` here is **not**
  detached so gradients flow into the generator.

### B.3 Optimizers & update schedule

- **Second optimizer** `Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))` — standard GAN betas,
  matching the Phase-0 config. Build it alongside the generator optimizer in
  `trainer.py:_build` (`:50`), store as `self.d_optimizer`.
- **Alternating update each step** (1:1): D step then G step (or G then D — keep consistent).
  Because the generator starts from a strong checkpoint, consider **D warm-up off** and a 1:1
  ratio; if D overpowers G (Part C.2 accuracy pins ~100%), throttle to update D every other
  step.
- Keep bf16 autocast consistent for both networks; LSGAN (squared error) is bf16-safe.

### B.4 Adversarial weight (compute, don't guess)

Target: `λ_adv · E[L_adv^G] ≈ 0.01 · L_main`. With `L_main ≈ 1e-4` (A.4, **read the real
value**) and LSGAN `E[L_adv^G] ≈ O(0.25–1)` early on, that gives **`λ_adv ≈ 1e-6 … 1e-5`**.
Procedure: read `train_fine_loss` from `training_log.json`, run ~100 warm-up steps logging the
**raw** (unweighted) `L_adv^G`, then set `λ_adv = 0.01 · L_main / mean(raw L_adv^G)`. Escalate
only if PSNR/MSE hold (Section 2.4). Reconstruction loss stays the dominant term throughout.

### B.5 Separate flagged entry point (default path untouched)

- Add `--use-adversarial` (`cmd.str2bool`, default **False**) + `--adv-weight`,
  `--adv-warmup-steps`, `--d-lr` in `trainer.py:add_parser_arguments` (`:18-22`), mirroring the
  `--amp-bf16` pattern.
- Gate **all** GAN code (`D` construction, `d_optimizer`, D/G adversarial terms, GAN logging)
  behind `if getattr(self.args, 'use_adversarial', False):`. When the flag is off, `_train`
  behaves exactly as today.
- Add a config entry (e.g. `src/train/harmonizer/script/train_adv.py`, or a
  `('use_adversarial', True)` block in a copied config) so the two experiments are distinct run
  scripts. Use a distinct `log_tag` (e.g. `ltlptl+adv`) so curves overlay against the baseline
  (`trainer.py:62-63`).

### B.6 Generator initialization (checkpoint) — FLAGGED AMBIGUITY

The prompt says init G from "current best." **`pretrained/harmonizer.pth` is the WRONG file** —
it's the original global-scalar Harmonizer, not the per-pixel decoder-branch model, and its
`state_dict` won't match (no `decoder.*`, and the regressor differs). The correct start is the
per-pixel best checkpoint saved by this trainer: `checkpoints/<log_tag>/best_model.pth`
(bare weights, `trainer.py:381`) or `checkpoint_latest.pth`. Since this GAN run should layer on
**the LTL/PTL result**, the intended file is the LTL/PTL run's `best_model.pth`.
**Confirm the exact path before running** (see Open items). Load via
`self.model.module.model.load_state_dict(...)` (mirrors `_load_checkpoint`, `trainer.py:419`).

---

## Part C — TensorBoard logging plan (prove D is doing real work, not damage)

All in `trainer.py`, gated by the adversarial flag.

### C.1 Loss components (separate, never merged into the recon curve)
- `Adv/D_loss` — total discriminator LSGAN loss.
- `Adv/G_adv_loss` — the **weighted** `λ_adv·L_adv^G` actually added to G.
- `Adv/G_adv_loss_raw` — the **unweighted** term (needed to recompute λ_adv, B.4).
- Keep `Loss/fine_loss`, `Loss/coarse_loss`, `Loss/train`, `Loss/val` **unchanged** so the
  reconstruction curve is directly comparable to the non-GAN baseline (regression visibility).

### C.2 Discriminator health (collapse detection)
- `Adv/D_acc_real`, `Adv/D_acc_fake` via the ported `patch_accuracy` (threshold on the LSGAN
  score). Both pinned near 1.0 → D too strong (throttle updates / lower `d_lr`); both near 0.5 →
  D uninformative. Healthy is somewhere in between and moving.
- `Adv/D_score_real_mean`, `Adv/D_score_fake_mean` — the raw score gap; a collapsing game shows
  the gap saturating.
- `GradNorm/generator_from_adv` — grad norm into the **generator** from `L_adv^G` only
  (compute on a separate backward or via hooks). ~0 means the adversarial term is dead weight.

### C.3 PSNR/MSE overall AND per-subset (prerequisite change, then log every eval)
Implement the per-subset propagation exactly as in `ARCHITECTURE_LTL_PTL_INTEGRATION.md §C.4`
(dataset returns `subset`; `trainer.py:227` uses it instead of `'IH'`; emit
`Metrics/PSNR/<subset>` and `Metrics/MSE/<subset>`). Then, **every eval**, compare against the
`40.32 PSNR / 15.5 MSE` baseline — with special attention to `Metrics/PSNR/HAdobe5k` and other
lighting-heavy cases, since that's where a relational discriminator should help (or where
adversarial artifacts would first show up).

### C.4 Fixed validation images, including a hard local-lighting case
Extend `_ensure_fixed_samples`/`_log_comparison_grid` (`trainer.py:304-343`, already one sample
per subset). Add at least one **strong-local-lighting** case (bright window in a dark room) to
the fixed list and log `Images/adv_hardcase_grid = [comp | gt | output | amplified-diff]` every
N steps, to watch whether fg/bg lighting correction improves or whether the GAN introduces
color shift / texture hallucination (Section 2.3 failure mode).

### C.5 Kill-switch (flag, don't auto-stop)
Track eval PSNR vs the baseline (40.32, or the LTL/PTL checkpoint's PSNR if that's the start).
If PSNR is **>0.3 dB below baseline for 3 consecutive evals**, write a loud warning to the log
and a TensorBoard text/scalar tag `Adv/killswitch_triggered=1`. Do **not** stop training — leave
the decision to the user (per the prompt).

---

## Part D — Order of operations & smoke test

1. **Prerequisite:** land the per-subset logging change (C.3) and run the current
   non-adversarial model once to capture **baseline per-subset curves** and the real
   `fine_loss` magnitude (B.4). Confirm the **starting checkpoint path** (B.6).
2. **Port D + Phase-0 warm start (optional):** bring `PatchGAN`/`lsgan_d_loss`/`patch_accuracy`
   onto this branch; optionally run the existing real-vs-composite Phase-0 pretrain so D starts
   competent (faster, more stable coupling).
3. **Smoke test (~a few hundred steps), report before any full run:** shapes correct
   (`D(cat(img,mask)) → [N,1,~30,~30]`); no NaNs; **both** `Adv/D_loss` and `Adv/G_adv_loss`
   are moving (not constant, not exploding); D real/fake accuracy not already pinned at 100%;
   PSNR/MSE on a small val subset ≈ the starting checkpoint (pipeline not broken). Verify the
   `--use-adversarial off` path is byte-for-byte the old behavior.
4. **Full run** only after the smoke test passes and the user confirms the checkpoint. Watch
   the kill-switch and `Metrics/PSNR/HAdobe5k` closely; escalate `λ_adv` only if stable.
5. **Attribution:** because this layers on LTL/PTL, keep three comparable runs —
   `ltlptl` (baseline), `ltlptl+adv` — overlaid via `log_tag`, so the discriminator's marginal
   effect on HAdobe5k is isolated from the LTL/PTL gain.

**Fastest isolation check:** the `Adv/D_acc_*` pair + `GradNorm/generator_from_adv` reveal a
dead or overpowering discriminator within a few hundred steps — run those in the smoke test
before committing GPU time.

---

## Open items / flagged ambiguities (not guessed)

- **Starting checkpoint (blocking):** `pretrained/harmonizer.pth` is the original global model
  and will **not** load into the per-pixel decoder model. Provide the path to the LTL/PTL (or
  current-best per-pixel) `best_model.pth`. — *Need your confirmation.*
- **Real main-loss magnitude:** set `λ_adv` from the actual logged `fine_loss` (B.4), not the
  ~1e-4 estimate.
- **Per-subset logging is a prerequisite** (C.3) and currently missing (A.3) — the kill-switch
  and the HAdobe5k question depend on it.
- **Discriminator source** currently exists only as `.pyc` on this branch (real source on the
  `discriminator` branch); confirm you want it ported into `decoder` vs merged from that branch.
- **Update ratio / spectral norm / weight escalation** are the three stability knobs; defaults
  proposed (1:1, SN on, λ start ~0.01×) but expect to tune if `Adv/D_acc_*` shows imbalance.
