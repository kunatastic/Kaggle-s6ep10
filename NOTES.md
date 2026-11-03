# Learning log

## EDA (`eda.py`)

- 699,635 train rows, 299,844 test rows, 22 features. 44.4% of passengers are satisfied, so the classes are close to balanced.
- The only missing values are in `Arrival Delay in Minutes` (292 train, 130 test).
- No duplicate rows. Train and test match closely: the largest gap in any feature's mean is 0.004 standard deviations, so a random stratified K-fold split should track the leaderboard.
- There are 13 survey ratings on a 0–5 scale. A 0 most likely means "not rated": `Baggage handling` never has a 0.
- The categorical columns carry a lot of signal on their own. Satisfaction rates by group:
  - Class: Business 73%, Eco 17%, Eco Plus 24%
  - Type of travel: business 58%, personal 10%
  - Customer type: loyal 50%, disloyal 20%
  - Gender: about 44% for both, so almost no signal
- Single-feature AUC: `Online boarding` scores 0.84 on its own, then Inflight entertainment 0.75 and Seat comfort 0.73. The delay columns, `Gate location` and departure/arrival time convenience are close to random (0.51–0.53).
- A HistGradientBoosting model with default settings gets 0.9579 AUC on a 20% holdout. Models on the original real dataset reach 0.99+, so the synthetic noise sets a much lower ceiling.

## Baseline (`train_lgb.py`)

- LightGBM on the raw features with 5-fold stratified CV: **out-of-fold AUC 0.95892**. The folds range from 0.9577 to 0.9597.
- First submission: **public 0.95836, private 0.95942**. The CV and leaderboard scores agree within about 0.0005, which confirms there's no meaningful drift between train and test.
- Leaderboard context: the competition closed on 2026-10-31, and the top public score is 0.96318. The top 20 are all between 0.9625 and 0.9632, so about 0.963 is the practical ceiling and 0.97 isn't reachable on this data.

## Visual EDA (`figures.py`)

### A 0 rating isn't a bad rating
![Share satisfied at each rating value](figures/rating_heatmap.png)

- Most ratings rise steadily from 1 to 5. `Online boarding` has the biggest range, from about 10% satisfied at ratings 1–3 to 88% at 5.
- **A 0 behaves nothing like a 1.** Passengers who gave wifi a 0 are 89% satisfied, and 0 on online booking or boarding gives 71% and 62%. In the original survey, 0 most likely means the service didn't apply (for example, no wifi on the flight). It's a separate category, not the bottom of the scale.
- Passengers with 3 or more zeros across their ratings are 86–94% satisfied, against a 44% average. That makes `n_zero` a real signal, although trees can find it without help.
- `Departure/Arrival time convenient` is flat at 40–51% and slightly *negatively* related to satisfaction. `Gate location` is U-shaped. Neither works as a plain ordered rating.

### Segments interact
![Share satisfied by trip type, loyalty and class](figures/segments.png)

- Loyal passengers on business trips in Business class are **79%** satisfied. Change only the class to Eco or Eco Plus and that drops to about 40%.
- Personal-travel passengers are about 10% satisfied **whatever their class**: paying for Business doesn't help them.
- Disloyal business travellers in Eco Plus are the least satisfied group, at 6%.
- The effect of one segment column depends on the others, which is why pair features are worth trying.

### Age and distance only matter for business trips
![Share satisfied by age and distance](figures/age_distance.png)

- For business travellers, satisfaction climbs with age to a plateau of about 75% at ages 40–58, then **collapses past 60** to about 20%. A single split can't capture that shape, so trees need several.
- Business travellers jump from about 42% to about 75% satisfied once flights pass ~1,200 miles, which probably reflects long-haul service.
- For personal travellers, both lines stay flat at about 10%.

### Delays mostly don't matter
![Share satisfied by delay](figures/delays.png)

- 91% of flights have no departure delay, so delay can't explain much overall.
- Even long delays only lower satisfaction from 45% to about 30%. Arrival delay has its effect earlier (from 6 minutes) than departure delay.
- The two delay columns are highly correlated (r = 0.84).

### The top two ratings interact strongly
![Online boarding x wifi](figures/boarding_wifi.png)

- A **wifi rating of 5 gives 92–96% satisfaction whatever the boarding rating**, and a wifi rating of 0 behaves almost the same way.
- In the middle, the two ratings combine in non-additive ways: boarding 3 with wifi 3 is only 6%, while boarding 1 with wifi 4 is 81%.
- Patterns like these are what target-encoding feature pairs is meant to capture.

## Feature pipeline (`train.py`)

| Features | LightGBM OOF AUC | Change |
|---|---|---|
| raw (baseline) | 0.95892 | |
| + rating stats, zero count, delay diff/max (`--feats base`) | 0.95878 | −0.0001 |
| + target encoding of all 22 columns and 45 pairs of the top 10 (`--feats te`) | **0.96031** | **+0.0014** |

- **Hand-made aggregates didn't help.** Trees already learn rating sums, zero counts and delay gaps through splits, so writing those features by hand only adds noise.
- **Target encoding did help, on every fold.** Treating each value, and each pair of values, as a category and replacing it with its out-of-fold satisfaction rate hands the model the non-additive interaction grid from the boarding × wifi chart in one feature. Trees would otherwise need many splits to approximate it.
- To avoid leaking the target, the encoder is fitted only on the training part of each fold. sklearn's `TargetEncoder` also cross-fits within that part, so no row's encoding uses its own label.
- LightGBM doesn't allow JSON special characters such as `:` in feature names, so the encoded columns are named `te_0…te_N`.
