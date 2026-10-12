# Ukraine air alerts forecast

Hourly probabilistic forecast of whether a new air raid alert will start in Kyiv City within the next hour, built on the history of alerts in Kyiv, its neighboring oblasts and the rest of Ukraine.

## Final model choice (fixed before the test run)
Final model: logistic regression without decay weights.
Chosen by Brier skill score on validation (2026-01-05 to 2026-08-09).

## 1. Task

At the start of every hour `t` the model outputs the probability that **a new air raid alert starts in Kyiv City in `[t, t + 1h)`**.

- **Unit:** one row per hour, on a UTC grid. Using UTC avoids the duplicated hour at the October DST switch.
- **Target:** 1 if at least one Kyiv City alert starts in the next hour, else 0. About 6.5% of hours are positive over the full grid and 12.9% in the test period.
- **Why a probability, not a yes/no:** there is no single user with a known cost of false alarms, so any threshold would be arbitrary. A calibrated probability can be turned into a decision later.
- **Why "starts" rather than "is active":** if an alert is already running at `t`, "will it be active in the next hour" is almost trivially yes and inflates the metrics.

## 2. Data

**Source:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), file `datasets/official_data_en.csv`. It is parsed from the official Telegram channel [@air_alert_ua](https://t.me/s/air_alert_ua). Timestamps are in UTC.

**Data snapshot used for all results:**

| field | value |
|---|---|
| dataset commit | `58ee75bc20113181b9ddf8029d4cbf3a62d8cd10` (committed 2026-09-08 03:44 UTC) |
| downloaded | 2026-10-12 14:00 UTC |
| rows (before deduplication) | 299,275 |
| latest alert start | 2026-09-07 05:28 UTC |

The exact file can be fetched at that commit: `https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/58ee75bc20113181b9ddf8029d4cbf3a62d8cd10/datasets/official_data_en.csv`. `load_data.py` writes the same information to `data/snapshot.json` on every download.

**Why the data ends on 7 September 2026.** The official file stopped updating on 8 September 2026; two open issues in the dataset repository report the same. The channel itself has no posts after 7 September 2026, so this is the most recent official data available. The grid ends at 2026-09-07 00:00 UTC, a few hours before the channel's last message (05:39 UTC). The parser writes an alert only after its all-clear message, so alerts still running when the channel stopped are missing; the margin keeps them from turning into false zeros in the target.

**Why not the volunteer dataset**, which is still updated daily. Two checks showed it is unreliable for this target:

- **Manual check against [alerts.in.ua](https://alerts.in.ua) for Kyiv City, 10–11 October 2026.** Of 17 alerts on alerts.in.ua:
  - 6 are missing from the volunteer data entirely;
  - only 4 have a start time within 6 minutes;
  - other starts are recorded 26–85 minutes late;
  - two alerts are merged into one row, and one alert appears that alerts.in.ua does not show.

  End times, on the other hand, match within a minute in 8 of 8 alerts that have a real end time. The target is defined by start times, so this source would train the model on noise.
- **Overlap with the official data, Kyiv City, 9 June – 7 September 2026:**

  | | volunteer | official |
  |---|---|---|
  | alerts | 236 | 199 |
  | overlap an alert in the other source | 73% | 86% |
  | median duration | 23.4 min | 30.0 min |

**Duplicates.** 113,845 of 299,275 rows (38%) in the official CSV are exact duplicates; before 2026 the monthly counts are doubled. All code applies `drop_duplicates()`. After deduplication Kyiv City has 2,267 alerts, all at level `oblast`, with no overlapping intervals.

**Training window:** from 1 September 2024. Features that look back 30 days use earlier history.

## 3. Features

All features are computed **as of `t`**, using only alerts that started before `t`. For alerts still running at `t`, the end time is treated as unknown.

**Kyiv City:**

| feature | definition |
|---|---|
| `starts_24h`, `starts_7d`, `starts_30d` | number of Kyiv alerts that started in `[t − window, t)` |
| `alert_active` | a Kyiv alert started before `t` and has not ended by `t` |
| `minutes_since_last_start` | minutes from the start of the latest Kyiv alert to `t` |
| `hour_local`, `dow_local` | hour of day and day of week of `t` in Kyiv time |

**Neighboring oblasts.** These are Kyivska, which surrounds the city, and all oblasts bordering it: Chernihivska, Zhytomyrska, Vinnytska, Cherkaska and Poltavska. Sumska is added on the assumption that drones often approach Kyiv from the north-east through it; this was not verified against attack-route data.

| feature | definition |
|---|---|
| `active_<oblast>` (7 features) | the oblast is under alert at `t` |
| `neighbors_active` | how many of the 7 are under alert at `t` |
| `oblasts_active_ukraine` | how many oblasts in Ukraine (excluding Kyiv City) are under alert at `t` |
| `neighbor_start_1h` | an alert started in any of the 7 oblasts in `[t − 1h, t)` |
| `neighbors_active_kyiv_quiet` | `neighbors_active × (1 − alert_active)` |

**"Oblast under alert" after the district-level reform.** Since December 2025 most alerts are declared per raion (district) rather than per oblast. An oblast counts as under alert at `t` if **at least one oblast-level or raion-level alert in it started before `t` and has not ended by `t`**.

Hromada-level alerts are excluded: they are very local, and in border communities they could keep the feature permanently at 1. In the seven neighboring oblasts they turned out to be almost absent in the official data (about 1% of hours in Zhytomyrska), so the exclusion changes little.

The reform does change the meaning of the feature for border oblasts:

| share of hours under alert | before Dec 2025 | after Dec 2025 |
|---|---|---|
| Chernihivska | 39% | 67% |
| Sumska | 67% | 79% |

The relation between the number of active neighbors and the Kyiv start rate stayed similar before and after the reform. For example, with 5 neighbors active the rate was 21% before and 19% after.

The interaction feature encodes a simple rule: a new Kyiv alert cannot start while the current one is still running. During a massive attack many neighbors are under alert, but if Kyiv already is too, the chance of a *new* start is low. With Kyiv quiet and 5 neighbors active, 27% of hours in the training period had a new start; with Kyiv already under alert, 3%. A linear model cannot express this without the interaction term. The feature was noticed while analyzing validation results (see section 5).

## 4. Leakage checks

`test_leakage.py` is a **truncation test**:

1. It picks 514 forecast times `t`: 300 random hours plus hours around the starts and ends of 100 random Kyiv alerts, where leaks usually hide.
2. For each `t` it builds the world as it was known at `t`: only alerts with `started_at < t`, with `finished_at` set to unknown for alerts still running.
3. It recomputes every feature with a separate, deliberately simple reference implementation (`features_from_history`) and compares it with the vectorized pipeline. Any mismatch fails the test and names the feature.
4. **Self-check:** the test then replaces `alert_active` with a leaky version ("any alert during `[t, t + 1h)`") and verifies that the test catches it (114 mismatches). This shows the test can fail, not just pass.

Other guards:

- `train_eval.py` asserts that every training row's target window `[t, t + 1h)` ends before the first forecast of the block.
- Scalers and encoders live inside an sklearn `Pipeline` and are fit only on training data.
- `train_eval.py test` refuses to run if `outputs/test_metrics.json` already exists. This is a discipline reminder, not a lock.

## 5. Validation

**Rolling-origin evaluation.** For each block of the evaluation period, all baselines and models are refit on every hour whose target was known before the block starts (an expanding window from September 2024), then predict the block.

**Baselines:**
- `base_rate_28d`: share of positive hours over the last 28 days;
- `hour_of_day_28d`: the same share computed separately for each hour of day.

The second baseline is the reference: a model that cannot beat it adds nothing beyond "what time is it".

**Metrics:**
- Brier score and Brier skill score (BSS) relative to `hour_of_day_28d` — the main metrics;
- PR-AUC, where a random model scores the positive rate;
- ROC-AUC and log loss for context.

**What was known before the test was run.** During data exploration we saw monthly alert counts that include the test period. In particular, alert frequency in Kyiv roughly doubled from August 2026. The intensity features (`starts_7d`, `starts_30d`) and daily refits were chosen after seeing this. Intensity also varied within the training data (e.g. 3.2% positive hours in December 2025 vs 7.5% in January 2026), so these choices are defensible without the test, but the exposure is real.

**Changes to the validation design, made after seeing validation results:**

1. **Validation window.** The original plan was 4 weeks immediately before the test (13 July – 9 August 2026). That window has only 60 positive hours.
   - With Kyiv-only features, logistic regression beat the hour-of-day baseline by +0.2% BSS.
   - After adding neighbor features it was −1.8%, while on a longer window the same features gave about +7%.
   - Monthly BSS showed a stable gain in January–June and almost none in July–August.

   A single month with 60 positives cannot separate the candidates, so validation was extended to **5 January – 9 August 2026** (5,208 hours, 344 positives).
2. **Refit frequency in validation.** To keep the longer window fast (≈1.5 min instead of ≈7 min), validation refits **weekly**; the test refits **daily**, as the model would run in practice. This affects all candidates equally.

The **interaction feature** was also added after analyzing validation data, as noted in section 3.

**Final model, fixed before the test was run:** logistic regression without sample weights, chosen by BSS on the validation window above. The test was run once.

## 6. Results

**Validation: 5 January – 9 August 2026, weekly refit, 5,208 hours, 6.6% positive**

| model | Brier | BSS vs hour-of-day | log loss | PR-AUC | ROC-AUC | mean prediction |
|---|---|---|---|---|---|---|
| base_rate_28d | 0.0619 | +2.1% | 0.246 | 0.066 | 0.50 | 0.062 |
| hour_of_day_28d | 0.0632 | 0 | 0.285 | 0.075 | 0.52 | 0.062 |
| **logreg (final)** | **0.0588** | **+7.1%** | **0.226** | **0.169** | **0.69** | 0.070 |
| logreg_decay90 | 0.0591 | +6.6% | 0.228 | 0.161 | 0.68 | 0.064 |
| hgb | 0.0600 | +5.1% | 0.233 | 0.157 | 0.68 | 0.069 |
| hgb_decay90 | 0.0626 | +0.9% | 0.248 | 0.122 | 0.64 | 0.061 |

`_decay90` models use sample weights that halve every 90 days of age; `hgb` is sklearn's `HistGradientBoostingClassifier`.

**Test: 10 August – 6 September 2026, daily refit, 672 hours, 12.9% positive (87 hours), run once**

| model | Brier | BSS vs hour-of-day | log loss | PR-AUC | ROC-AUC | mean prediction |
|---|---|---|---|---|---|---|
| base_rate_28d | 0.1135 | +2.0% | 0.390 | 0.129 | 0.50 | 0.105 |
| hour_of_day_28d | 0.1158 | 0 | 0.428 | 0.146 | 0.55 | 0.105 |
| **logreg (final)** | **0.1028** | **+11.2%** | **0.354** | **0.356** | **0.71** | 0.091 |
| logreg_decay90 | 0.0960 | +17.2% | 0.327 | 0.425 | 0.77 | 0.101 |
| hgb | 0.1013 | +12.6% | 0.345 | 0.346 | 0.76 | 0.094 |
| hgb_decay90 | 0.0987 | +14.8% | 0.352 | 0.384 | 0.78 | 0.092 |

**Reading the results:**

- The final model beats the hour-of-day baseline by 11% in Brier score on the test period. Its PR-AUC (0.36) is 2.4× the baseline's and 2.8× the positive rate.
- All models **under-predict** on the test period: mean prediction 0.09–0.10 vs an actual rate of 0.129. They were trained mostly on calmer months.
- **Observation, not a basis for changing the model:** on the test period, time-decay weighting is better than the final model (logreg_decay90: +17.2% BSS). On validation it was slightly worse. With 87 positive test hours the difference may be noise. A plausible reason is that weighting recent data helps after the summer 2026 change in alert patterns. This is a candidate for future work, to be validated on future data, not adopted from the test.

### What the final model gets right and wrong on the test period

The aggregate metrics above hide a split picture; `analyze_test.py` breaks it down.

**It ranks hours well at the top.**

| hours with the highest predicted risk | alert-start rate: final model | alert-start rate: hour-of-day baseline | share of all 87 alerts in these hours |
|---|---|---|---|
| top 5% (34 hours) | **59%** (4.5× average) | 14% | 23% |
| top 10% (67 hours) | **43%** (3.3×) | 15% | 33% |
| top 20% (134 hours) | **28%** (2.2×) | 17% | 44% |

The hour-of-day baseline takes only about 10 distinct values on the test period, so many hours are tied at each cutoff. Its numbers average over the tied hours instead of depending on sort order.

All 15 highest-risk hours share one pattern: Kyivska oblast under alert, 3–6 neighboring oblasts under alert, 8–13 oblasts active nationwide, and Kyiv itself still quiet. In 10 of these 15 hours an alert did start in Kyiv.

**It misses about half of the alerts.** In 42 of the 87 alert hours (48%) the model gave less than 10%, often 5–7%.
- These alerts look like ordinary hours in the alert data. Kyivska oblast was under alert in only 24% of them, compared with 80% for the alerts the model caught and 37% for hours without an alert. On average 2.7 neighbors were active, compared with 2.5 in hours without an alert.
- From the alert history alone, these alerts arrive without warning: at the forecast time there is usually no alert in Kyivska oblast, and the neighbors look no different from a quiet hour.

**Back-to-back alerts became common.** In 10 test hours a new alert started within the hour while a previous Kyiv alert was still running at `t`. This happened in 10.4% of such hours in the test period, compared with 2.7% before it. The model learned the old pattern and rates these hours low.

**Probabilities are too low at both ends.**

| predicted probability | hours | mean prediction | actual alert-start rate |
|---|---|---|---|
| ≤ 5% | 256 | 3.4% | 6.2% |
| 5–10% | 230 | 6.9% | 11.3% |
| 10–20% | 115 | 14.0% | 13.0% |
| 20–30% | 45 | 25.0% | 28.9% |
| > 30% | 26 | 35.6% | 65.4% |

The middle range is calibrated. The low range under-predicts by roughly a factor of 1.7, consistent with the higher alert rate in the test period. The highest-risk hours start an alert about twice as often as predicted (26 hours, so this estimate is noisy).

**Bottom line:** the model is useful as a ranking of risk — "this hour is much more dangerous than usual" is right more often than not at the top. As an absolute probability it is too low, and it cannot see roughly half of the alerts, which give no signal in the alert history of Kyiv and its neighbors.

## 7. Limitations

- **Data stops on 7 September 2026.** Forecasts after that date would need a different live source, and the volunteer source is not accurate enough for start times.
- **Small evaluation samples.** The test has 87 positive hours, so its metrics have wide uncertainty; no confidence intervals are reported.
- **The validation design changed twice after seeing validation results, and test-period alert counts were seen during exploration** (section 5). The test itself was run once, after the final model was fixed.
- **Changing relationships.** In summer 2026 the link between alerts in neighboring oblasts and alerts in Kyiv weakened: Kyiv alerts more often started while the neighbors were quiet. One possible explanation is a change in attack tactics, but this cannot be verified from alert data, so it remains a hypothesis.
- **Feature drift from the district-level reform.** The reform changed what "oblast under alert" means for Chernihivska and Sumska (section 3).
- **Under-prediction.** The models are not calibrated to the current alert rate: mean prediction 0.091 vs an actual 0.129 on test, and in the highest-risk bin 36% predicted vs 65% observed. Do not read the outputs as exact probabilities.
- **Half of the alerts are invisible to these features.** 48% of test alert hours got a prediction below 10%. These alerts start without preceding alerts in Kyivska oblast or other neighbors. Alert history cannot anticipate them; a leading signal from outside the alert data (e.g. launch reports) would be needed.
- **Back-to-back alerts.** Hours in which Kyiv was already under alert at `t` and a new alert started within the next hour became about 4× more frequent in the test period (10.4% vs 2.7%). The interaction feature and the training history both push such hours down.
- **The target follows the official channel.** Alerts still running when the channel stopped are not in the data. Some long night alerts (e.g. February 2025, many lasting 2–8 hours) look real but were not verified against an external source.
- **No hyperparameter tuning.** Model settings are reasonable defaults.

## 8. Future work

- **A live source:** the [alerts.in.ua API](https://alerts.in.ua/api-request) (history limited to about 30 days), to extend the evaluation past 7 September 2026.
- **Text data:** posts from the Air Force or monitoring Telegram channels about launches and drone routes are likely the strongest leading signal. Leakage rule: use publication time, not edit time, and only posts published before `t`.
- **Recency weighting and calibration:** tune the decay half-life on validation, and recalibrate the intercept on recent weeks to fix under-prediction.
- **Uncertainty:** block bootstrap by day for confidence intervals on BSS and PR-AUC.
- **Other horizons and targets:** e.g. 3 hours ahead, or time to next alert.

## 9. How to run

Requires Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python load_data.py              # download the official CSV into data/, record the snapshot
python test_leakage.py           # truncation test; expect two PASS lines
python train_eval.py val         # validation table -> outputs/val_*
python train_eval.py test        # test table -> outputs/test_*; refuses to run twice
python analyze_test.py           # ranking, calibration and missed-alert diagnostics for the final model
```

Exploration scripts, not needed for the pipeline:
- `explore_kyiv.py`: Kyiv monthly counts, duplicates, levels;
- `load_volunteer.py`: volunteer data and its comparison with official;
- `check_neighbors.py`: sanity check of the neighbor features.

`data/` is not committed. `load_data.py` downloads the current file from the dataset's `main` branch; it has not changed since 8 September 2026. To reproduce the results exactly, check that the commit SHA in `data/snapshot.json` matches the one in section 2, or download the file from the pinned URL given there into `data/official_data_en.csv`.