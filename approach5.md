# Approach 5 — Two-stage lightweight residual refinement (CDTNet-style)

> Cheap (<0.1M params) and highly complementary to Approach 1. Adds a local residual
> correction on top of the global/parametric filter output — the CDTNet/DCCF/PCT-Net
> consensus fix for high-resolution HAdobe5k. Recommended Stage-2 change after LD.

---

## 0. Reality check & where the loss conflict actually lives

- Stage 1 = your existing pipeline: `predict_arguments` → filter arguments → `FilterPerformer` → harmonized image (`harmonizer.py:82-98`). This is the global/parametric transform. Keep it.
- Stage 2 = a tiny conv refiner that predicts a residual Δ added to the stage-1 image inside the foreground.
- **Loss naming collision — read carefully.** The trainer already has `fine_loss` and `coarse_loss` (`trainer.py:104,117`), but those are **labeled-batch vs additional-data-batch** losses, *not* stage losses. Do not overload them. The new stage-1/stage-2 supervision is a separate axis. CDTNet's fix (`L = L_pix + L_rgb + L_ref`) means: **supervise the stage-1 output AND the refined output against GT separately**, so the parametric head is never asked to minimize a dense pixel objective it structurally can't. We implement that as an added `L_refine` term.
- The refiner works in image space on `[comp, stage1_out, mask]` — the simplest grounded version. (CDTNet also feeds upsampled decoder features; you can add those later, but image-space is enough to start and avoids replumbing decoder features through `restore_image`.)

---

## 1. New module — residual refiner

Add to `src/train/harmonizer/module/module.py` (after `FilterPerformer` or near the decoder classes):

```python
class ResidualRefiner(nn.Module):
    """CDTNet-style local refinement: predict a residual on the parametric output.

    in_ch = 3 (comp) + 3 (stage1 out) + 1 (mask) = 7  ->  two conv3x3+BN+ELU  -> 3
    """
    def __init__(self, in_ch=7, hidden=16):
        super(ResidualRefiner, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ELU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ELU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, comp, coarse, mask):
        delta = self.net(torch.cat([comp, coarse, mask], dim=1))
        refined = torch.clamp(coarse + delta * mask, 0.0, 1.0)   # only refine fg
        return refined
```

---

## 2. Wire into `Harmonizer`

**File:** `src/train/harmonizer/module/harmonizer.py`

### 2a. Import & construct
Add `ResidualRefiner` to the import block (lines 8-14), then in `__init__` after the regressor is built (inside `if per_pixel:`, around line 48):
```python
            self.refiner = ResidualRefiner(in_ch=7, hidden=16)   # <-- ADD
        else:
            self.decoder = None
            self.regressor = CascadeArgumentRegressor(1280, 160, 1, len(self.filter_types))
            self.refiner = None                                  # <-- ADD
```

### 2b. Expose both stage outputs
Add a method (do not break `restore_image`, which callers/eval use):
```python
    def restore_image_stages(self, comp, mask, arguments):
        """Return (coarse, refined). coarse = parametric stage-1 output."""
        coarse = self.restore_image(comp, mask, arguments)       # existing stage 1
        if self.per_pixel and self.refiner is not None:
            refined = self.refiner(comp, coarse, mask)
        else:
            refined = coarse
        return coarse, refined
```

---

## 3. Expose stage outputs to the trainer (proxy)

**File:** `src/train/harmonizer/model.py`, `Harmonizer.forward` (lines 46-55). Replace body:
```python
    def forward(self, inp):
        resulter, debugger = {}, {}
        x, mask = inp
        arguments = self.model.predict_arguments(x, mask)
        coarse, refined = self.model.restore_image_stages(x, mask, arguments)
        resulter['outputs'] = refined          # final output = refined
        resulter['coarse'] = coarse            # <-- ADD for stage-1 supervision
        resulter['arguments'] = arguments
        return resulter, debugger
```
Also register the refiner params so it trains. In `__init__` (after the decoder param-group insert, lines 40-44):
```python
        if getattr(self.model, 'refiner', None) is not None:
            self.param_groups.append({
                'params': filter(lambda p: p.requires_grad, self.model.refiner.parameters()),
                'lr': self.args.lr,
            })
```

---

## 4. Add the stage-1 (coarse) supervision term

**File:** `src/train/harmonizer/trainer.py`, `_train`, after `pred_image` is fetched (line 93) and the fine/coarse (data-batch) losses are computed. The cleanest spot is right before `loss = fine_loss + coarse_loss` (line 123). Add a term that supervises the **coarse (stage-1)** output against GT so the parametric head keeps a clean objective:

```python
            # --- Stage supervision (CDTNet L_rgb): keep a loss on the parametric
            #     (pre-refiner) output so the filter head isn't forced to serve the
            #     dense refinement objective. 'outputs' is already the refined image.
            coarse_image = tool.dict_value(resulter, 'coarse')
            if coarse_image is not None:
                l_coarse = self.criterion(
                    (coarse_image[:lbs],), (l_gt_image,), (x[:lbs], mask[:lbs]))
                stage1_loss = torch.mean(l_coarse[0])
                self.meters.update('stage1_loss', stage1_loss.item())
            else:
                stage1_loss = 0.0

            loss = fine_loss + coarse_loss + 0.5 * stage1_loss   # weight is tunable
```
`fine_loss`/`coarse_loss` already supervise the **refined** output (since `outputs`=refined) = CDTNet's `L_ref`. The new `stage1_loss` = CDTNet's `L_rgb`. Start the weight at 0.5 and tune (see §7).

**Validation path** (`trainer.py:208-223`) uses `outputs` (=refined) — no change needed; it will report metrics on the final refined image, which is what you want.

---

## 5. Parameter budget

`ResidualRefiner(7,16)`:
- conv1 3×3: `7*16*9 + 16 = 1,024`
- conv2 3×3: `16*16*9 + 16 = 2,320`
- conv3 3×3: `16*3*9 + 3 = 435`
- 2× BN(16): `64`
- Total ≈ **0.004M** (well under the report's <0.1M).

New total ≈ **5.4M + 0.004M ≈ 5.4M**. Negligible.

---

## 6. Mirror to inference copy

Apply §1–§2 to `src/model/module.py` and `src/model/harmonizer.py`. For inference you want the **refined** output, so update `infer.py` / `batch_inference.py` / `eval_pretrained*.py` to call `restore_image_stages(...)[1]` (or make `restore_image` return refined when a refiner exists — but that changes existing semantics, so prefer the explicit call). Verify checkpoints load: the refiner adds new keys, so old baseline checkpoints load with `strict=False` or you retrain.

---

## 7. Verification

**7a. Shape/NaN test.** `approach1.md §7a` harness plus:
```python
coarse, refined = m.restore_image_stages(comp, mask, args)
assert coarse.shape == refined.shape == (2,3,256,256)
assert torch.isfinite(refined).all()
assert (refined[ (mask.expand_as(refined))<0.5 ] == coarse[ (mask.expand_as(coarse))<0.5 ]).all()  # bg untouched
```
The last assert confirms the refiner only edits the foreground.

**7b. Residual starts near zero.** At init, `delta` should be small; log `delta.abs().mean()` — early training it should be ≪ the coarse error, i.e. the refiner starts as near-identity and *adds* correction. If it explodes, lower its LR or add a `tanh` on `delta`.

**7c. Both losses decrease.** Watch TensorBoard: add `Loss/stage1_loss` logging next to the existing scalars (`trainer.py:167-169`). Both `stage1_loss` (parametric) and the refined-output loss should fall. If `stage1_loss` rises while total falls, the refiner is masking a degrading parametric stage — raise the stage-1 weight.

**7d. Refiner actually helps (ablation).** Compare refined-output PSNR vs coarse-output PSNR on the val set (log both). The gap is the refiner's contribution; it should be positive and largest on **HAdobe5k**. If refined ≈ coarse everywhere, the refiner is dead weight — increase `hidden` or feed decoder features.

**7e. Loss-weight sweep.** You flagged that stricter per-argument supervision (λ=16) hurt. The analogue here is the stage-1 weight in §4. Sweep `{0.25, 0.5, 1.0}`; too high over-constrains the parametric head, too low lets it drift. Pick by HAdobe5k val MSE.

**7f. Overfit one batch + HAdobe5k subset metric**, as in prior approaches.

---

## 8. Why this pairs best with Approach 1

Approach 1 adds correspondence (semantic "what to match"); Approach 5 adds local high-res correction ("clean up the residual the parametric pass left"). CDTNet/DCCF/PCT-Net all show the HAdobe5k win comes from exactly this local-correction stage. Combined cost ≈ 5.45M — roughly half of HDNet's 10.41M — leaving large budget headroom.
