## Final Project Proposal – Sports Match Outcome Prediction ##

This project aims to build a data-driven prediction framework for football match outcomes
using historical performance data and supervised machine learning techniques. The objective
is to estimate the probability of each of the three possible results, bHome Win, Draw, or Away
Win (W/D/L),and evaluate whether statistical models can capture meaningful patterns in
match data.
The analysis will focus on one primary league to ensure model consistency and avoid
performance dilution across heterogeneous competitions. If data availability permits,
additional leagues may be explored, either through separate league-specific models or by
incorporating league identity as an explicit feature. Historical match information (2018–2024)
will be collected from publicly accessible databases and include variables such as recent
team form, goals scored and conceded, home-advantage indicators, pre-match bookmaker
odds, and other contextual factors. Feature engineering will follow a transparent
methodology, detailing lookback windows (e.g., last 5 matches) and addressing early-season
cases with limited data.
The predictive model will be trained on seasons 2018–2023 and tested exclusively on season
2024 to maintain strict temporal separation and avoid data leakage. Several machine-learning
techniques will be evaluated, such as multinomial logistic regression and tree-based
classifiers. Model performance will be assessed through accuracy, log loss, and Brier score,
allowing evaluation of both classification correctness and probability calibration.
To contextualize model performance, two baselines will be used:
1. Random guessing, to establish a naïve benchmark;
2. Bookmaker implied probabilities, which represent a highly efficient market standard.
The comparison with bookmakers will serve as the key measure of whether the model
adds predictive value, with no intention to design or deploy a real betting system.
Ultimately, this project combines sports analytics, predictive modelling, feature engineering,
and rigorous validation to determine how effectively machine learning can support rational,
evidence-based decision-making in the context of football match outcomes.
