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
