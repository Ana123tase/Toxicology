# Gold Directional Signal System — Development Guide

**Instrument:** XAU/USD
**Output:** Buy / Sell / Wait, refreshed hourly
**Purpose:** confirmation signal only — not a standalone trading system, not a market predictor
**Status:** pre-development reference document — read this before writing any model code

---

## 0. Read this before anything else

This system will sit on top of real trading accounts. The standard you're building to, in three concrete ways:

1. **A wrong "Wait" costs nothing. A wrong "Buy" or "Sell" costs someone money.** Every design decision below is biased toward the model staying quiet when it isn't sure, rather than being biased toward looking active or impressive.
2. **A backtest that looks great and is wrong is more dangerous than a backtest that looks mediocre and is honest.** Most of this document is about preventing the specific, well-known ways time-series backtests lie to their own builders.
3. **You are not shipping a black box.** Every module must be able to explain, at minimum, *what regime it thinks the market is in* and *how confident it is* — because "trust me" is not something you can respectfully say to someone else's capital.

If a shortcut in here ever trades development speed for backtest honesty, don't take it. Come back to this document instead.

---

## 1. System architecture (applies to all three modules)

### 1.1 Timeframes and features

| Timeframe | Features computed | Notes |
|---|---|---|
| M5 | Volume, price action, 40EMA | Fastest-moving, noisiest signal layer |
| M15 | Volume, price action, 40EMA | Intermediate structure |
| H1 | Volume, price action, 40EMA | Primary trend/context layer |
| H1 only | Sentiment | Doesn't have a native sub-hourly update cadence — do not fabricate resolution it doesn't have |
| H1 only | Time-of-candle analysis | Learned patterns tied to specific hourly candle position (session opens, rollover, etc.) |

**Rule:** never upsample sentiment or time features down to M5/M15 by forward-filling and pretending it's fresh information at that resolution. If it only updates hourly, it enters the model as an hourly feature, full stop — attached at the H1 timestep, not synthetically duplicated across faster timesteps. Misrepresenting feature freshness is one of the most common ways people accidentally build leakage into a live system.

### 1.2 Feature definitions to lock down before coding

These need firm, written definitions before a single line of training code is written, because "price action" and "volume" are not self-defining:

- **Price action**, per timeframe: candle body % of range, upper/lower wick ratio, N-candle momentum (define N per timeframe — likely different for M5 vs H1), distance from recent swing high/low.
- **Volume**: For XAU/USD spot, we will use broker tick volume. Normalize per-timeframe
- **40EMA**: use *distance from price to the 40EMA* (in ATR units, not raw price units, so it's comparable across volatility regimes) rather than the raw EMA value.
- **Sentiment**: Sentiment is a slow-changing/observed contextual feature whose timestamp and publication delay must be respected. Name the exact source (news NLP score, positioning data, options skew, etc.) and its true update frequency and publication lag before building the pipeline. This determines how "live" the H1 sentiment feature actually is.
- **Time-of-candle**: cyclical encoding (sin/cos of hour-of-day, day-of-week).

### 1.3 Labeling scheme (shared across all three models)

The label is the thing every part of this system depends on getting right, so define it once, in one place, and use it identically for all three modules:

- **Buy**: price moves ≥ your defined threshold (you've referenced ~$10) upward within the next hour, without first moving against you by more than a defined invalidation distance.
- **Sell**: mirror of the above, downward.
- **Wait**: neither condition is met with sufficient probability — includes both "truly flat" markets and "moved but whipsawed through both directions first."

Train Buy/Sell as the core two-class target, and derive Wait from calibrated probability thresholds** — this tends to produce better-calibrated confidence than training Wait as a native third class, because "choppy" is a much more heterogeneous category than "went up" or "went down," and heterogeneous classes are harder for a model to learn cleanly.

### 1.4 Validation discipline (non-negotiable, applies to every module)

- **Walk-forward validation only.** Train on a window, validate on the period immediately after it, roll forward, repeat. Never randomly shuffle time-series data into train/test splits — this leaks future information backward and is the single most common way people fool themselves in this exact kind of project.
- **Purge and embargo around the split boundary.** Because your label looks one hour into the future, a training sample whose label window overlaps the validation period leaks information. Drop (purge) those boundary samples and add a small embargo gap after each split.
- **No feature may use same-candle-close information the live system wouldn't have at signal time.** If it's computed from a candle that hasn't closed yet at the moment you'd actually generate the signal, it doesn't belong in training data either.
- **Report performance across multiple non-overlapping walk-forward folds**, not one lucky backtest window. A model that performs well in one 3-month slice and collapses in the next is not done.

### 1.5 Future-leakage prevention (mandatory, checked at every stage)

Leakage is the single most likely way this system quietly lies to you — a leaked model can look excellent in every backtest and fail immediately the moment it runs live, because live data doesn't have the future information the backtest accidentally handed it. Treat leakage prevention as a standing checklist, not a one-time design decision.

- **Point-in-time feature store.** Every feature value must be tagged with the exact timestamp it was *actually knowable* at — not the timestamp of the candle it describes. A 40EMA "for the H1 candle closing at 14:00" is only knowable at 14:00, not before. Build the feature pipeline so it is physically incapable of returning a value from before it existed, rather than trusting yourself to remember not to misuse it.
- **Sentiment and time-of-candle publication lag.** Sentiment sources often publish with a delay (news processed minutes after the event, positioning data released on a schedule). The feature pipeline must attach sentiment to the timestamp it was *published*, not the timestamp of the market event it describes. If a sentiment score reflecting 13:00 conditions wasn't actually available until 13:20, it cannot be used to generate a signal at 13:05.
- **Multi-timeframe alignment audit.** Because M5/M15 sequences are preserved (not flattened) for LSTM/TFT, double-check that no M5 or M15 candle inside a given H1 window ever includes information from *after* the H1 signal timestamp — it's easy to accidentally include the M5 candle that closes at the same moment as, or after, the H1 decision point.
- **Automated leakage tests, not just manual review.** Before each model is considered "built," run a mechanical check: shuffle or null out the label, retrain, and confirm accuracy collapses to chance. If a model still "predicts" well with a scrambled label, a leaked feature — not real signal — is very likely present.
- **Same rules apply to warm-start retraining (see Section 7).** Every retraining cycle re-triggers this checklist. A leakage-free initial build can still leak later if the feature pipeline, sentiment source, or timestamp handling changes between retrains without being re-audited.

---

## 2. Module 1 — LightGBM

**Goal of this stage:** answer, as fast and honestly as possible, whether this feature set carries any real signal at all before investing in the two heavier models.

### Build order
1. Build the M5/M15/H1 feature pipeline first — this pipeline is shared by all three models, so getting it right here saves rework later.
2. Flatten all timeframe features into one row per H1 timestep for this module only (M5/M15 features aggregated up to H1 — e.g. last value, mean, or slope over the H1 window; decide and document which). For LSTM/TFT, do not do this flattening — preserve the sequence structure per timeframe (e.g. sequence of M5 candles within each H1, sequence of M15 candles, sequence of H1 candles) otherwise you throw away much of the multi-timeframe information.
3. Train a baseline two-class (Buy vs Sell) LightGBM classifier with no Wait class yet, to sanity-check that price-action/volume/EMA features carry directional signal above chance.
4. Add sentiment and time-of-candle features, confirm they improve out-of-sample performance before keeping them — don't include an input just because you have it.
5. Calibrate probability outputs (Platt scaling or isotonic regression) — raw LightGBM probabilities are not well-calibrated out of the box, and calibration is what makes your Wait threshold meaningful rather than arbitrary.
6. Sweep the Wait threshold on validation folds and report the accuracy/coverage tradeoff (tighter threshold → higher accuracy on signals issued, fewer signals issued). Let the fund managers see this tradeoff explicitly rather than picking one point for them.

### What to watch for
- With ~11+ engineered features (3 vars × 3 timeframes + time + sentiment) on a lean dataset, watch `max_depth`, `min_child_weight`, and `gamma` closely — this is a small-feature, high-noise problem, not a big-data problem, and LightGBM will overfit fast if left unregularized.
- Feature importance output isn't optional here — review it every retrain. If sentiment or time-of-candle consistently shows near-zero importance, that's a real finding to report, not a bug to hide.

---

## 3. Module 2 — LSTM

**Goal of this stage:** capture genuine sequential/momentum structure in price action that Module 1's snapshot features can't see.

### Architecture
- **Multi-branch design**: one LSTM branch per timeframe (M5, M15, H1), each ingesting a sequence of [price action, volume, EMA-distance] for that timeframe's own lookback window, preserving the native sequence structure per timeframe rather than aggregating M5/M15 into a single H1 row.
- Each branch's final hidden state is concatenated, then merged with time-of-candle and sentiment as slow-changing/observed contextual features whose timestamp and publication delay must be respected (injected at the dense merge layer, not inside the recurrent branches — they don't have the same "sequence of candles" structure as the price/volume/EMA inputs, and their true update cadence must be respected).
- Output head: same probability structure as Module 1 (Buy/Sell probabilities, calibrated, with Wait derived from threshold) so the two modules are directly comparable.

### Build order
1. Fix the lookback window per timeframe first (e.g., how many M5 candles, how many M15, how many H1) — this is the first design decision and it directly trades off signal vs. noise vs. how much training data you actually need.
2. Build and validate the M15 or H1 single-branch version alone first, to confirm the recurrent approach adds anything over Module 1 before building the full multi-branch model.
3. Add the remaining branches one at a time, checking out-of-sample performance after each addition — don't add complexity that doesn't earn its place.
4. Apply the same calibration and Wait-threshold sweep as Module 1.

### What to watch for
- This is the most data-hungry of the three models. If your usable hourly history is limited, say so honestly rather than forcing a deep network onto too little data — a shallow 1-2 layer LSTM per branch is very likely the right call here, not a deep stack.
- Dropout and recurrent dropout are your main regularization tools — tune them deliberately, don't leave them at framework defaults.
- Compare directly against Module 1 on the *same* walk-forward folds. If LSTM doesn't beat LightGBM's calibrated accuracy/coverage tradeoff, that's a legitimate outcome to report, not a failure to hide.

---

## 4. Module 3 — TFT (Temporal Fusion Transformer)

**Goal of this stage:** let the model learn which input and which timeframe matters most, and let that weighting change over time, rather than fixing it by hand as Modules 1 and 2 do.

### Input structure
- **Time-varying known inputs:** time-of-candle encodings (known in advance).
- **Time-varying observed inputs:** price action, volume, EMA-distance at M5/M15/H1, preserving sequence structure per timeframe (do not aggregate M5/M15 into an H1 row).
- **Slow-changing/observed contextual feature:** sentiment, whose timestamp and publication delay must be respected (given its H1-only, slower-updating nature).

### Build order
1. Develop a market-competitive TFT — no minimal variant. We are optimizing for accuracy. Implement the full competitive architecture: variable selection networks, LSTM encoder-decoder, gated residual networks, and multi-head attention for multi-timeframe fusion, configured for a single 1-hour-ahead horizon.
2. Once the competitive version is validated and trains stably, incrementally tune components (multi-head attention heads, gating layers, hidden sizes) based on out-of-sample performance, always comparing against Modules 1 and 2.
3. Use the built-in variable selection network's output as an explainability tool — report which timeframe and which feature the model is weighting most heavily at different times. This is one of TFT's genuine advantages over Modules 1 and 2, and it's worth surfacing to the fund managers as a feature, not just an internal diagnostic.
4. Apply the same calibration and Wait-threshold methodology as the other two modules.

### What to watch for
- This is the most complex model to implement correctly and the easiest one to quietly get wrong (attention layers and gating mechanisms fail silently — they train, they just don't learn anything useful). Compare against Modules 1 and 2 at every stage. If TFT isn't beating them, don't ship it just because it's the most sophisticated architecture on paper.
- Needs the most data of the three. If your hourly gold history is a limiting factor, this module will feel it hardest.

---

## 5. Cross-model evaluation (apply identically to all three)

Every module gets judged on the same scorecard, computed on the same out-of-sample walk-forward folds:

- **Accuracy on signals issued** (Buy/Sell calls only — excludes Wait periods).
- **Coverage** (% of hours where the model actually issued Buy or Sell rather than Wait).
- **Calibration** (does a 70%-confidence signal actually resolve correctly ~70% of the time? — plot a reliability curve, don't just report a single accuracy number).
- **Stability across folds** (does performance hold across different multi-month windows, or does it swing wildly?).
- **Max drawdown in a simple simulated equal-risk backtest** (not because this is a trading system, but because "how bad does a losing streak look" is exactly what the fund managers will ask, and you should already know the answer).

Present all three models' scorecards side by side. The winner isn't necessarily the highest raw accuracy — it's whichever model gives the best accuracy-at-usable-coverage, with the most stable calibration across folds.

---

## 6. Before this touches a live account

- **Paper-trade every module** against live data for a meaningful stretch before anyone acts on its output with real capital — a clean backtest is necessary but never sufficient.
- **Version and log every signal the system ever issues**, with the feature values that produced it. If something goes wrong, or a fund manager asks "why did it say Buy at 14:00 on the 12th," you need to be able to answer that from logs, not memory.
- **Monitor for drift.** Markets change regime. Set a review cadence (e.g. monthly) to check whether live performance is still tracking backtest expectations, and have a plan for what happens when it doesn't — pause the signal, retrain, or both.
- **Be explicit, every time you present this, that it is a confirmation layer, not a standalone strategy.** That framing needs to survive contact with people who might otherwise be tempted to over-trust an "AI signal." You're building a tool that helps disciplined traders be a little more disciplined — not an oracle.

---

## 7. Retraining strategy — warm-start offline retraining

The system does not learn live, tick-by-tick, against real signals. It retrains on a fixed schedule, offline, against data that has already fully closed and been leakage-audited — and each retrain starts from the *previous* model's learned state rather than from scratch. This is deliberate: pure online learning on noisy hourly gold data is a fast way to let a bad week silently corrupt the model, and retraining fully from scratch every cycle wastes the model's accumulated understanding of stable, slow-moving structure (session patterns, EMA-regime behavior) for no benefit.

### 7.1 What "warm-start" means per model

- **LightGBM**: warm-start by continuing to boost additional trees on top of the existing trained ensemble using the new data window, rather than refitting all trees from iteration zero. Periodically (e.g. every few cycles) also train a fresh from-scratch model on the current rolling window and compare it against the warm-started version on the same out-of-sample fold — if the from-scratch version is meaningfully better, that's a signal the warm-started model has drifted somewhere the incremental trees can't fully correct, and a full refit is warranted.
- **LSTM**: load the last deployed checkpoint's weights, continue training on the new data window at a reduced learning rate (a full-learning-rate restart risks overwriting what the model already learned, i.e. catastrophic forgetting). Freeze earlier layers for the first portion of each retrain cycle if the new data window is small relative to the full training history.
- **TFT**: same checkpoint-continuation approach as LSTM, with particular attention to the variable-selection network's weights — verify after each retrain that the feature/timeframe weightings it's learned haven't collapsed or degenerated, since this component is the most prone to quietly stop learning anything useful under naive fine-tuning.

### 7.2 Retraining cycle (applies to all three modules)

1. **Trigger:** fixed schedule (e.g. weekly or monthly — decide based on how fast gold's regime actually shifts, not an arbitrary convenience interval), not continuous/streaming updates.
2. **Data window:** only fully-closed, leakage-audited candles up to the retrain cutoff. Re-run the Section 1.5 leakage checklist against the new data batch specifically — a new sentiment source hiccup or a broker feed gap is exactly the kind of thing that sneaks in between retrains.
3. **Warm-start fit:** continue training from the last deployed checkpoint using the method described in 7.1.
4. **Validation gate:** the newly retrained model must be evaluated on a walk-forward out-of-sample fold it has never trained on, using the full Section 5 scorecard (accuracy on signals issued, coverage, calibration, stability, drawdown).
5. **Promotion decision:** the retrained model only replaces the live model if it meets or beats the currently deployed model's scorecard on that fresh fold. If it doesn't, the current live model stays in place and the retrain is logged as a rejected candidate, not silently discarded.
6. **Versioning:** every retrained model, its exact training data window, its scorecard, and the promotion decision get logged and version-tagged. You must always be able to answer "which model version generated this signal, and what was it trained on" for any historical signal.
7. **Rollback path:** keep the previous N deployed checkpoints available so a newly promoted model that starts behaving badly live can be rolled back quickly, without waiting for a new retrain cycle to fix it.

### 7.3 Why offline, not online

Online/streaming learning (updating weights after every new candle or every new signal outcome) is deliberately avoided here. It removes the validation gate in 7.2 entirely — there's no walk-forward fold to check a live-updating model against before it's already acting on real signals. Offline, scheduled, gated retraining is slower to adapt, but it's the version of "the model keeps learning" that doesn't also mean "the model can silently get worse with no one noticing until real money already moved."

---

## 8. Build order summary

```
1. Shared feature pipeline (M5/M15/H1 + H1-only sentiment/time)  →  build once, reuse for all 3, preserving M5/M15 sequence structure for LSTM/TFT
2. LightGBM   →  validate signal exists at all, establish baseline scorecard
3. LSTM       →  validate sequential structure adds value over Module 1
4. TFT        →  validate learned multi-input/multi-timeframe fusion adds value over Modules 1 & 2, as market-competitive TFT optimized for accuracy
5. Cross-model scorecard comparison, presented together
6. Paper-trading period before any live use
7. Warm-start offline retraining cycle established and validated before ongoing live operation
```

Do not skip a stage to save time. Each stage's job is to earn the right to build the next, more complex one.