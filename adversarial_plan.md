# Diagnosis — (1) Alpha-filter dead collapse, (2) Arg-loss regresses with the decoder

Scope: analysed from the committed repo (`more_filters` / `extended_filters` filter
math, `decoder` trainer, untracked `inverse_targets.py` + `inverse_torch.py`, and the
`INVERSE_*` design docs). The decoder+argloss integration itself lives on your
workstation, so items marked **VERIFY** are things to confirm against that code.

The two bugs are **coupled**: a dead brightness/tint head (Bug 1) emits `y_pred≈0`
while its arg target is `y_true=-x_degrade≠0`, so it also inflates the arg-loss share
and injects noise into Bug 2. Fix Bug 1 first.

---

## BUG 1 — Brightness & Tint (the two "alpha" filters) explode, then go dead

### The exact culprit
Both `BrightnessFilter` and `TintFilter` share this gain:

```python
amask = (x >= 0).float()
alpha = (1 / ((1 - x) + eps)) * amask + (x + 1) * (1 - amask)
```

- `x ≥ 0`:  `alpha = 1/(1-x)`  → **pole at x=1** (α=10 at x=0.9, α=100 at x=0.99).
- `x < 0` :  `alpha = 1+x`      → **→0 at x=-1**.

Brightness does `V *= alpha` (in HSV). Tint does `G /= alpha`, `R,B *= sqrt(alpha)`.

### Why it explodes
1. **Derivative blows up faster than the value.** `d/dx [1/(1-x)] = 1/(1-x)²` → 100 at
   x=0.9, 1e4 at x=0.99. Gradients w.r.t. the predicted arg diverge near x→1.
2. **Tint has a *second* pole.** `G/alpha` with `alpha=1+x → 0` as `x→-1` sends
   `G/alpha → ∞`. So Tint is unstable at **both** ends; Brightness only at x→1.
3. **`sqrt(alpha)` in Tint** also diverges as x→1 (R,B channels).

### Why clamping α to [0.1, 10] then made them go dead (predict <0.005)
4. **Hard clamp = zero gradient outside the band.** α=10 at x=0.9 and α=0.1 at x=-0.9,
   so for |x|>0.9 the gradient is exactly 0. The only "live" region is x∈(-0.9, 0.9),
   and inside it the gradient is **wildly asymmetric**: steep `1/(1-x)²` on the positive
   side, flat `=1` on the negative side. An optimizer facing that asymmetry retreats to
   the one safe fixed point — **x=0, where α=1 (identity / no-op)**. That is exactly your
   "99.5% dead, predicting ≈0."
5. **Post-multiply clamp to [0,1] destroys gradient.** Large α saturates most pixels to
   1; `clamp` has zero gradient on saturated pixels, so the arg receives signal from
   almost no pixels → it drifts to the no-op.
6. **The degrade itself is information-destroying.** A degrade of x=0.8 means α=5
   (V×5 → blown to white). The composite is unrecoverable, so the minimum-expected-loss
   prediction over the symmetric degrade distribution is **the mean, ≈0** → collapse.
7. **Your global grad-norm clip to 1 explains "nothing learned."** These 2 heads produce
   huge per-sample gradients that dominate the global norm; clipping the *global* vector
   scales **all 11 heads + backbone** down to near-zero. That is why global clipping
   killed learning everywhere, not just for the two bad filters.

### Diagnosis checklist — Bug 1
- [ ] Confirm the two are **Brightness + Tint** (both use `alpha=1/(1-x)`). (Confirmed in code.)
- [ ] Log **per-filter arg mean AND std** each epoch. Dead = std≈0 & mean≈0. (Trainer already
      accumulates arg *mean* at ~line 215 — add std.)
- [ ] Log **per-filter gradient norm** (arg/head output). Confirm Brightness/Tint spike.
- [ ] Log **% pixels clamped to 0/1** after each filter during the degrade. If Bright/Tint
      saturate a large fraction, the degrade range is too hot.
- [ ] Confirm the **degrade arg range** fed to these filters (are you sending full [-1,1]?).
- [ ] Confirm grad clipping is **per-filter/per-parameter**, not global.
- [ ] **Round-trip test:** degrade a batch with known x, apply the inverse, measure recovery
      excluding clamped pixels. Huge error for Bright/Tint ⇒ parameterisation is the problem.

### Fixes — Bug 1 (ranked)
1. **Reparameterise the gain in log-space (the real fix).**
   ```python
   K = math.log(A_MAX)          # A_MAX ≈ 3–4  → K ≈ 1.10–1.39
   alpha = torch.exp(K * x)     # x ∈ [-1, 1]
   ```
   - Bounded: `α ∈ [1/A_MAX, A_MAX]`, **no poles, no clamp needed**.
   - Symmetric (keeps your design intent): `α(-x) = 1/α(x)` exactly, and now the
     **gradient is symmetric** too (`dα/dx = K·α`, max `K·A_MAX ≈ 3–5`).
   - Tint's `G/α` can no longer blow up (α ≥ 1/A_MAX).
   - Bonus: the closed-form inverse collapses to **`y = -x`** for *both* Brightness and
     Tint (exact up to clamp) — clean arg-loss target, and it fixes Bug 2's coverage too.
2. **If you must keep the rational form,** move the pole outside the domain:
   `alpha = 1/(1 - s*x)` with `s≈0.9` (pole at x=1.11, α_max=10 at x=1). Still steeper than
   exp; second-best.
3. **Shrink the degrade range for these two filters** (e.g. x∈[-0.6,0.6]) so you never
   approach the pole and stop saturating the image. Combine with fix 1.
4. **Per-filter / clip-by-value gradients**, never global norm, so two heads can't gag the rest.
5. **Un-stick a collapsed head:** small non-zero bias init on those heads; temporarily
   up-weight their degrade magnitude so they get signal again; watch arg-std recover.
6. **Mask clamp-saturated pixels out of the image loss** so blown pixels don't zero the
   gradient (helps, but fix 1 largely removes the saturation in the first place).

---

## BUG 2 — Arg-loss helps the 6-filter global model but *regresses* the decoder model

Your own `INVERSE_LOSS_INVESTIGATION.md` (§"Two ways to add it", non-commutativity note)
already flags the mechanism: the added term
`loss += w·Σ_i |ŷ_i − y_true_i|` *"can pull each filter toward its individually-ideal
inverse, which (because filters do not commute and clamp is lossy) may not be jointly
optimal."* That is exactly why capacity matters — read on.

### Root causes (most→least likely to be the primary)
1. **Stat-convention mismatch (whole-image target vs local-stat filter).** The closed-form
   inverses for **temperature / contrast / saturation** are derived under the
   **whole-image mean** convention (docs say so explicitly). With the decoder you almost
   certainly run `local_stats=True` (per-pixel local means in `FilterPerformer`). Then the
   *actual* filter inverse is no longer `-x` / `-x/(1+x)` — so for **3 of the 6** filters
   the arg target is systematically **wrong**, and the arg loss actively fights the image
   loss. In the global model (whole-image stats) the targets match, so it helps. This one
   single-handedly explains "helps global, hurts decoder." **VERIFY `local_stats`.**
2. **Coverage gap / index mismatch across 11 filters.** `inverse_torch.py` defines targets
   for only the **6 base filters**. If you run 11 with the arg loss, the 5 new heads either
   get no target or — worse — are paired **by index**. `filter_types` order **differs**
   between `more_filters` and `extended_filters` (HUE_ROTATION moves), so any index-based
   `y_true[i] ↔ y_pred[i]` pairing supervises the **wrong filter**. Must pair **by name**.
   *(Good news: all 5 new filters are exactly invertible — see fix 2 — so close the gap.)*
3. **Capacity × over-constraint (the core conceptual issue).** The synthetic degrade is a
   *smooth low-freq field*. The arg loss pins **every pixel** of **every** filter map to the
   closed-form inverse of that synthetic field. A low-capacity 6-scalar model can't overfit
   that, so the term acts as a healthy regulariser. The high-capacity **per-pixel decoder**
   *can* satisfy it — and doing so makes it reproduce the *synthetic-degrade inverse* instead
   of learning to harmonise *real* iHarmony composites. Net: helps low capacity, hurts high.
4. **Teacher-forcing error is larger per-pixel.** `y_true_i` is the inverse computed against
   the true degrade chain (`comp_{i+1}`), but the model's running image at stage i equals
   `comp_{i+1}` only at convergence. For the 3 **stateful** filters this makes the target
   wrong *en route*, and per-pixel prediction diverges from the degrade-time stats faster.
5. **Shape/aggregation mismatch.** `compute_y_true` expects `x_scaled [N,1]`; the decoder
   degrade is a **field `[N,1,H,W]`**. If y_true is built from a scalar/mean but compared to
   the per-pixel map (or vice-versa), you get a broadcast/aggregation error that inflates the
   raw arg loss — consistent with the term hitting **53% of total at λ=0.6**.
6. **The λ symptom is a tell, not just weighting.** You dropped λ 0.6→0.2, share 53%→30%,
   still bad. If the term were merely too strong, lowering λ would help. It didn't → the arg
   loss points at a **wrong optimum** (mismatched target / wrong filter / local-vs-global),
   not just too heavy. A 30–53% share also means the **raw magnitude is large** → target
   mismatch, because a *correct* arg target should sit well below the image loss.

### Diagnosis checklist — Bug 2
- [ ] **VERIFY** y_true↔y_pred are paired **by filter name**, not index (orders differ!).
- [ ] **VERIFY** every active filter has a target; base file covers only 6 (add the other 5).
- [ ] **VERIFY `local_stats`.** If ON with the decoder, temp/contrast/sat targets are wrong.
- [ ] **VERIFY** y_true is computed **per-pixel from the degrade field** and its shape equals
      the predicted map; both in **native [-1,1] units** (not one scaled by a per-filter range).
- [ ] Log `l_fine`, `l_arg`, `l_coarse` **separately** + per-filter `arg_loss_i`. Healthy
      share is <~20%.
- [ ] **Ablation A (is it net-harmful?):** decoder with λ=0 vs λ>0. If λ=0 wins, the term is
      currently net-negative → schedule/decay it (fix 4).
- [ ] **Ablation B (stat mismatch?):** decoder + argloss with `local_stats=OFF`. If it
      recovers, root cause #1 confirmed.
- [ ] Measure the **teacher-forcing gap** `|running_image_i − comp_{i+1}|` per stage for the
      stateful filters.
- [ ] Watch **args mean/std on *real* composites**: should stay nonzero-and-correct. If the
      arg loss collapses them toward 0, it's over-constraining (root cause #3).

### Fixes — Bug 2 (ranked)
1. **Match the stat convention.** Simplest: keep **whole-image means** for temperature/
   contrast/saturation even in per-pixel mode, so `-x` / `-x/(1+x)` stay exact targets.
   (Or derive local-stat-aware targets — more work; not worth it first pass.)
2. **Close the coverage gap — pair by name and add the 5 missing inverses (all exact):**
   - `hue_rotation`: forward `h += x·π (mod 2π)` → **`y = -x`** (pure additive rotation).
   - `r/g/b_curve`: forward gamma exponent `(1-x)` on a channel → **`y = 1 - 1/(1-x)`**
     (identical to `shadow`).
   - `tint` (after the exp reparam in Bug 1): **`y = -x`** (like brightness).
   With these you can supervise all 11 — or deliberately restrict to the well-posed ones.
3. **Feed the per-pixel degrade field into the (pointwise) inverses** to build a `[N,1,H,W]`
   `y_true` map; compare to the predicted map. Assert shapes + units.
4. **Use the arg loss as a warm-up curriculum, not a permanent term.** It is most useful
   early (wakes up each head, prevents the Bug-1 dead-collapse) and most harmful late
   (over-constrains the decoder). Schedule λ: hold ~0.2–0.3 for the first N epochs, then
   **decay to 0** (or a tiny floor). This directly resolves "helps low-capacity, hurts high."
5. **Supervise only the global (pooled) component of each arg with the decoder.** Since the
   synthetic degrade is smooth, apply the arg loss to the **spatially-pooled mean** of the
   predicted map vs pooled `y_true`, leaving the decoder free to add spatial detail that
   harmonises real composites. Keeps the per-filter wake-up without pinning every pixel.
6. **Keep the image loss as master** (your doc's own recommendation). After fixes 1–3 shrink
   the raw arg magnitude, re-tune λ toward a <20% share; don't tune λ before fixing the
   target mismatch (that's why 0.6→0.2 didn't help).
7. **Add the zero-target idempotency group** (your design §13): a few already-harmonised
   real images with `y_true=0`, so the decoder learns "clean in → no-op," preventing it from
   inventing corrections on already-good regions.
8. **If stateful targets stay off en route,** compute `y_true` for temp/contrast/saturation
   against the model's **live running-image stats** rather than the degrade-time stats
   (your doc's noted refinement).

---

## Suggested order of operations
1. Reparameterise Brightness+Tint gain to `exp(K·x)` (Bug 1 fix 1). Re-check arg std recovers.
2. Turn per-filter grad logging on; drop global grad clipping.
3. In the arg loss: pair by name, add the 5 inverses, match `local_stats`/whole-image stats,
   assert per-pixel shapes+units (Bug 2 fixes 1–3).
4. Run Ablation A (λ=0 vs λ>0) and Ablation B (local_stats off) on the decoder.
5. Switch λ to a warm-up-then-decay schedule; re-tune to <20% arg share (Bug 2 fixes 4–6).
6. Add the zero-target idempotency group if the decoder still over-corrects clean regions.
