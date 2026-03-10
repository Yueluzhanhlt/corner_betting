import pandas as pd
import helper as hp
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import mean_poisson_deviance
 
WEALTH = 341
train_df = pd.read_excel('train.xlsx')
test_df  = pd.read_excel('test.xlsx')
FEATURES = ["HomeTeamId","AwayTeamId","LeagueId","Season", 'Month', 'DayOfWeek', 'isWeekend', "SeasonPhase","DaysFromSeasonStart"]

#train_df = hp.add_features(train_df)
train_df = hp.prepare_df(train_df)

hp.plot_total_hist_poisson(train_df)
hp.plot_home_vs_away_box(train_df)
hp.plot_home_away_diff(train_df)
hp.plot_mean_league(train_df)
hp.plot_month_avg(train_df)

#CV for alpha tuning
tuned = hp.cv_tune_alpha(train_df,
                         alphas=(0.0, 0.001, 0.003, 0.01, 0.011, 0.012, 0.013), # 0.03, 0.1, 0.3, 1.0, 3.0),
                         n_splits=5,
                         scoring="neg_mean_poisson_deviance",
                         n_jobs=-1,
                         verbose=0)

#home_pip = tuned["home_pipe"]
#away_pip = tuned["away_pipe"]
print("Best alpha (home):", tuned["best_alpha_home"])
print("Best alpha (away):", tuned["best_alpha_away"])
best_home_alpha = tuned["best_alpha_home"]
best_away_alpha = tuned["best_alpha_away"]
model = hp.model_fit(train_df, alpha_home=best_home_alpha, alpha_away=best_away_alpha)
hp.plot_pred_vs_true(model['y_pred_total'], model['y_test_total'])
hp.plot_residual(model['y_pred_total'] - model['y_test_total']) 

#try nb
y_train_pred = model['home_pipe'].predict(model['x_train']) + model['away_pipe'].predict(model['x_train'])
y_train_true = model['y_home_train'] + model['y_away_train']
alpha_hat = hp.find_alpha_nb2(y_train_true, y_train_pred)
alpha = alpha_hat
r = (1.0 / alpha) if alpha > 0 else np.inf
print(f"alpha_hat (NB2): {alpha_hat:.6f}")

#make predictions on test df
test_df = hp.add_features(test_df)
x_test  = test_df[FEATURES]

y_home_pred = model['home_pipe'].predict(x_test)
y_away_pred = model['away_pipe'].predict(x_test)
y_tot_pred  = model['y_pred_total']#y_home_pred + y_away_pred
y_tot_true  = model['y_test_total'] 

n = len(y_tot_pred)
ll_pois = poisson.logpmf(y_tot_true, y_tot_pred).sum() / n
ll_nb   = hp.nb_ll(y_tot_true, y_tot_pred, alpha_hat) / n
print("Avg log-score Poisson:", round(ll_pois, 3))
print("Avg log-score NB     :", round(ll_nb, 3))

ybar = np.full_like(y_tot_true, y_train_true.mean(), dtype=float)  # global-mean predictor
dev_null  = mean_poisson_deviance(y_tot_true, ybar)
dev_model = mean_poisson_deviance(y_tot_true, y_tot_pred)
pseudo_r2 = 1 - dev_model/dev_null
print("Pseudo-R2 vs global mean:", round(pseudo_r2, 3))

avg_lp, avg_lnb, d_mean, d_se, (ci_lo, ci_hi) = hp.logscore(y_tot_true, y_tot_pred, alpha)
print(f"Avg log-score Poisson: {avg_lp:.3f}")
print(f"Avg log-score NB     : {avg_lnb:.3f}")
print(f"Mean NB-Poisson: {d_mean:.6f}  SE: {d_se:.6f}  95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
#try single value prediction
#model_full = hp.model_fit_tot(train_df, alpha=0.01)
#y_pred = model_full['y_pred']
#y_true = model_full['y_test']
#n = len(y_pred)
#ll_pois = poisson.logpmf(y_true, y_pred).sum() / n
#ll_nb   = hp.nb_ll(y_true, y_pred, alpha_hat) / n
#print("Avg log-score Poisson:", round(ll_pois, 3))
#print("Avg log-score NB     :", round(ll_nb, 3))

lines = test_df['Line']
P_U, P_A, P_O = zip(*[hp.nb_probs_for_line(m, L, alpha, r) for m, L in zip(y_tot_pred, lines)])

out_df = test_df.copy()
out_df['P(Under)'] = np.round(P_U, 3)
out_df['P(Over)']  = np.round(P_O, 3)
out_df['P(At)']    = np.round(P_A, 3)
out_df = out_df.set_index("MatchId", drop=False)

out_df = hp.decide_bet_dir(out_df, min_ev=0.01)

out_df = hp.construct_porfolio(out_df, wealth=WEALTH, frac=0.5)

n_O = int((out_df["Bet (U/O)"] == "O").sum())
n_U = int((out_df["Bet (U/O)"] == "U").sum())
n_N = int((out_df["Bet (U/O)"] == "No Bet").sum())
print(f"Over: {n_O}  Under: {n_U}  No Bet: {n_N}")
print(f"Total stake: {out_df['Stake'].sum():.2f}  (Total wealth {WEALTH})")

out_df['Date'] = pd.to_datetime(out_df['Date']).dt.strftime("%d/%m/%Y")
out_df = out_df.drop(['Month', 'Season', 'DayOfWeek', 'isWeekend', "SeasonPhase","DaysFromSeasonStart"], axis=1)
#out_df.to_excel("test_bet_added.xlsx", index=False)
out_df.to_csv("test.csv", index=False)

#try to find the best mini bet cut off to maximize expected log growth

bet = out_df['Stake']
#print(out_df.head())
p_win, p_push, b = hp.extract_bet(out_df)

grid = np.round(np.linspace(0.0, 1.50, 16), 2)
best = hp.find_best_min(p_win, p_push, b, bet, wealth=341)
print(best)
#mask_o = (out_df['Bet (U/O)'] == 'O')
#mask_u = (out_df['Bet (U/O)'] == 'U')

#stake = np.zeros(len(out_df))
##print(mask_o)
#PO = out_df['P(Over)'] #np.asarray(P_O, dtype=float)
#PU = out_df['P(Under)']#np.asarray(P_U, dtype=float)
#PA = out_df['P(At)']#np.asarray(P_A, dtype=float)
#Over = out_df['Over']
#Under = out_df['Under']
#stake[mask_o] = [hp.kelly_fraction(p, o, p_push=push, frac=0.5) for p, o, push in zip(PO[mask_o], Over[mask_o], PA[mask_o])]
#stake[mask_u] = [hp.kelly_fraction(p, o, p_push=push, frac=0.5) for p, o, push in zip(PU[mask_u], Under[mask_u], PA[mask_u])]

#stake = stake * WEALTH
##print(stake[0:5])
#tot_stake = sum(stake)
#print(f"Used stakes before scale : {tot_stake}")
#Norm_scale = WEALTH / tot_stake
#stake *= Norm_scale
##print(stake[0:5])

#out_df['Stake'] = stake

#number_bets = (out_df['Stake'] != 0).sum()
#print(f"number of matches bet {number_bets}")
#print(sum(out_df['Stake']))

#print(out_df.head())














