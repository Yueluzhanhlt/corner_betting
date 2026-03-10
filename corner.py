import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from scipy.stats import poisson, nbinom

import helper as hp

Train_df = pd.read_excel('train.xlsx')
Test_df = pd.read_excel('test.xlsx')

cols = ["MatchId","LeagueId","Date","HomeTeamId","AwayTeamId","Home_Corners","Away_Corners"]

df = Train_df[cols].copy()

df['Date'] = pd.to_datetime(df['Date'], errors="coerce")
df['Season'] = df['Date'].dt.year
df.dropna(subset=['Date'], inplace=True)

df['Total_Corners'] = df['Home_Corners'] + df['Away_Corners']

print("Rows:", len(df))
print("Mean(total corners):", np.round(df["Total_Corners"].mean(), 3))
print("Var(total corners):", np.round(df["Total_Corners"].var(), 3))
print("Home - Away corner mean diff:", np.round(df["Home_Corners"].mean() - df["Away_Corners"].mean(), 3))

#build stratum
df['stratum'] = df['LeagueId'].astype(str) + '_' + df['Season'].astype(str)

counts = df['stratum'].value_counts()

print("Min stratum size:", counts.min())
print(df.head())

Features = ["HomeTeamId","AwayTeamId","LeagueId","Season"]
X = df[Features]
Y = df[['Home_Corners', 'Away_Corners']].astype(float)
#train test split now
#add shuffle so that do not bias the split (i.e. all later season games are drifting into test_df) 
x_train, x_test, y_train, y_test = train_test_split(
                                         X, Y, 
					 test_size = 0.1, 
    					 random_state=42, 
					 shuffle=True, 
					 stratify=df['stratum']
				   )

y_train_home = y_train['Home_Corners']
y_train_away = y_train['Away_Corners']

y_test_home = y_test['Home_Corners']
y_test_away = y_test['Away_Corners']

#construct model pipeline (for GLM)
def build_pipe(alpha=1.0, max_iter=2000):
    pre = ColumnTransformer(
          [('cat', OneHotEncoder(handle_unknown='ignore'), Features)],
          remainder = 'drop', 
          verbose_feature_names_out=True,
    )
    return Pipeline([('pre', pre), ('glm', PoissonRegressor(alpha=alpha, max_iter=max_iter))])

alpha_home, alpha_away = 1.0, 1.0

home_pip = build_pipe(alpha_home).fit(x_train, y_train_home)
away_pip = build_pipe(alpha_away).fit(x_train, y_train_away)

home_pred = home_pip.predict(x_test)
away_pred = away_pip.predict(x_test)
y_pred = home_pred + away_pred

y_tot = (y_test_home + y_test_away).values
y_train_tot = (y_train_home + y_train_away).values

mae_mu = mean_absolute_error(y_tot, y_pred)
dev    = mean_poisson_deviance(y_tot, y_pred)
ll     = float(poisson.logpmf(y_tot, y_pred).sum())

print(f"Test rows: {len(y_tot)}")
print(f"MAE on mu_total: {mae_mu:.3f}")
print(f"Mean Poisson deviance: {dev:.3f}")
print(f"Log-likelihood: {ll:.2f}")

pearson = (y_tot - y_pred) / np.sqrt(np.maximum(y_pred, 1e-8))
print("Pearson resid mean/std:", pearson.mean(), pearson.std())

print("Test var/mean:", np.var(y_tot, ddof=1) / np.mean(y_tot))

#Try NB
y_pred_train = home_pip.predict(x_train) + away_pip.predict(x_train)
numer = np.sum((y_train_tot - y_pred_train)**2 - y_pred_train)
denom = np.sum(y_pred_train**2)
alpha_hat = max(0.0, numer / denom)
print("alpha_hat (NB2):", alpha_hat)

ll_pois = hp.poisson_ll(y_tot, y_pred)
ll_nb   = hp.nb_ll(y_tot, y_pred, alpha_hat)
print("Test log-lik (Poisson):", ll_pois)
print("Test log-lik (NB)     :", ll_nb)

alpha = alpha_hat
r = (1.0 / alpha) if alpha > 0 else np.inf

#now generate the probabilities for test df
Test_df = Test_df.copy()
Test_df['Date'] = pd.to_datetime(Test_df['Date'], errors='coerce')
Test_df['Season'] = Test_df['Date'].dt.year
Test_X = Test_df[Features]

final_home_test = home_pip.predict(Test_X)
final_away_test = away_pip.predict(Test_X)
final_tot = final_home_test + final_away_test

lines = Test_df['Line'].values
alpha = alpha_hat

P_U, P_A, P_O = zip(*[hp.nb_probs_for_line(m, L, alpha, r) for m, L in zip(final_tot, lines)])

out_df = Test_df.copy()
out_df['P(under)'] = P_U
out_df['P(over)']  = P_O
out_df['P(At)']    = P_A

print(out_df.head())
