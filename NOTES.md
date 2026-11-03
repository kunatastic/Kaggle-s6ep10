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
