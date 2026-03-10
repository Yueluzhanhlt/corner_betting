import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

REQUIRED_COLS = ["MatchId","LeagueId","Date","HomeTeamId","AwayTeamId", "Home_Corners","Away_Corners"]
FEATURES_cat = ["HomeTeamId","AwayTeamId","LeagueId","Season", 'Month', 'DayOfWeek', 'isWeekend'] #"Home_Goals", "Away_Goals"]
FEATURES_num = ["SeasonPhase","DaysFromSeasonStart"] #,"GamesHomeSoFar","GamesAwaySoFar"]
FEATURES = FEATURES_cat + FEATURES_num
def add_features(df):
    out = df.copy()
    out['Date'] = pd.to_datetime(out['Date']) #, error='coerce')
    out['Season'] = out['Date'].dt.year
    out['Month']  = out['Date'].dt.month
    out['DayOfWeek'] = out['Date'].dt.dayofweek
    out['isWeekend'] = (out['DayOfWeek'] >= 5).astype(int)

    grp = out.groupby(["LeagueId","Season"])["Date"]
    smin = grp.transform("min")
    smax = grp.transform("max")
    dur = (smax - smin).dt.days.replace(0, 1)
    out["DaysFromSeasonStart"] = (out["Date"] - smin).dt.days.clip(lower=0)
    out["SeasonPhase"] = out["DaysFromSeasonStart"] / dur
    # interactions
#    out["LeagueSeason"]   = out["LeagueId"].astype(str) + "_" + out["Season"].astype(str)
#    out["HomeTeamSeason"] = out["HomeTeamId"].astype(str) + "_" + out["Season"].astype(str)
#    out["AwayTeamSeason"] = out["AwayTeamId"].astype(str) + "_" + out["Season"].astype(str)
    return out

def prepare_df(df):
    df = df[REQUIRED_COLS].copy()
    df = add_features(df)
    df = df.dropna(subset=['Date'])
    df['Total_Corners'] = df['Home_Corners'] + df['Away_Corners']
    df['Home_Away_diff'] = df['Home_Corners'] - df['Away_Corners']
    return df

def build_pipe(alpha=1.0, max_iter=2000):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_cat),
          ("num", StandardScaler(with_mean=False), FEATURES_num)],
        remainder="drop",
       # verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("glm", PoissonRegressor(alpha=alpha, max_iter=max_iter))])
    

def model_fit(df, test_size=0.1, random_state=42, alpha_home=1.0, alpha_away=1.0):
    X = df[FEATURES]
    y_home = df["Home_Corners"]
    y_away = df["Away_Corners"]

    stratum = df['LeagueId'].astype(str) + '_' + df['Season'].astype(str)
    x_train, x_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
										X, y_home, y_away,
										test_size=test_size,
										random_state=random_state,
										shuffle=True, stratify=stratum
    )

    home_pip = build_pipe(alpha_home).fit(x_train, y_home_train)
    away_pip = build_pipe(alpha_away).fit(x_train, y_away_train)

    y_pred_home = home_pip.predict(x_test)
    y_pred_away = away_pip.predict(x_test)

    y_pred_tot = y_pred_home + y_pred_away
    y_test_tot = y_home_test + y_away_test

    mae = mean_absolute_error(y_test_tot, y_pred_tot)    
    mpd = mean_poisson_deviance(y_test_tot, y_pred_tot)    
    print(f"MAE on total corners: {mae:.3f}")
    print(f"Mean Poisson deviance: {mpd:.3f}") 
    return {
        "home_pipe": home_pip,
        "away_pipe": away_pip,
        "x_train": x_train, "x_test": x_test,
        "y_home_train": y_home_train, "y_home_test": y_home_test,
        "y_away_train": y_away_train, "y_away_test": y_away_test,
        "y_pred_total": y_pred_tot,
        "y_test_total": y_test_tot,
    }

def model_fit_tot(df, test_size=0.1, random_state=42, alpha=1.0):
    X = df[FEATURES]
    y = df["Home_Corners"] + df["Away_Corners"]
    stratum = df['LeagueId'].astype(str) + '_' + df['Season'].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
                                                         X, y,
                                                         test_size=test_size,
                                                         random_state=random_state,
                                                         shuffle=True, stratify=stratum)
    model = build_pipe(alpha).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mpd = mean_poisson_deviance(y_test, y_pred)
    print(f"MAE on total corners: {mae:.3f}")
    print(f"Mean Poisson deviance: {mpd:.3f}")
    return{
        "model" : model,
        "y_pred": y_pred,
        "y_test": y_test,
        "x_train":x_train,
        "y_train":y_train
    }




def plot_total_hist_poisson(df):
    tot_corners = df['Total_Corners']
    lam = tot_corners.mean()
    max_ = tot_corners.max()
    bins = np.arange(-0.5, max_+1.5, 1.0)
    
    plt.figure()

    plt.hist(tot_corners, bins=bins, density=True)

    #Poisson pdf at integers 0 to max_
    vals = np.arange(0, max_+1)
    pdf_vals = np.exp(-lam)*(lam**vals) / np.maximum(1, np.array([np.math.factorial(k) for k in vals]))
    plt.plot(vals, pdf_vals, marker='o')
    
    plt.title("Total Corners Histogram with Poisson pdf")
    plt.xlabel('Total Corners')
    plt.ylabel('Density')
    plt.savefig('figs/Tot_corner_density.png', dpi=160)    


def plot_home_vs_away_box(df):
    plt.figure()
    plt.boxplot([df['Home_Corners'], df['Away_Corners']], labels=["Home", "Away"])
    plt.title("Home vs Away Corners")
    plt.ylabel('Corners')
    plt.savefig('figs/Home_vs_Away_box.png', dpi=160)


def plot_home_away_diff(df):
    plt.figure()
    plt.hist(df['Home_Away_diff'], bins=31)
    plt.title('Home - Away Corners (Home advantage)')
    plt.xlabel('Home Away diff')
    plt.ylabel('Frequency')
    plt.savefig('figs/home_away_diff.png', dpi=160)


def plot_mean_league(df, top=10):
    league_group = (df.groupby('LeagueId').agg(n=("MatchId", 'count'), mean_tot=('Total_Corners', 'mean'))
                                         .sort_values("n", ascending=False)
					 .head(top)		
					 .reset_index())
    x = np.arange(len(league_group)) 
    y = league_group['mean_tot'].values
    labels = league_group['LeagueId'].astype(str).values
    fig, ax = plt.subplots(figsize=(8,5), dpi=160)
    ax.bar(x, y, width=0.75)                            # thicker bars (try 0.9 if you want)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('LeagueId')
    ax.set_ylabel('Mean total corners')
    ax.set_title(f'Mean Total Corners by League, Top {top} by Match numbers')
   # fig.tight_layout()
   # fig.savefig('figs/mean_total_by_League.png', dpi=160)
   # plt.figure()
   # plt.bar(league_group['LeagueId'], league_group['mean_tot'])
   # plt.title(f'Mean Total Corners by League, Top {top} by Match numbers')
   # plt.xlabel('LeagueId')
   # plt.ylabel('Mean total corners')
   # plt.savefig('figs/mean_total_by_League.png', dpi=160)



def plot_month_avg(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    monthly = (df.groupby(['Year', 'Month'])['Total_Corners'].agg(matches='count', mean_tot='mean').reset_index())
    monthly['MonthStart'] = pd.to_datetime({"year": monthly["Year"], "month": monthly["Month"], "day": 1})
    
    plt.figure()
    plt.plot(monthly['MonthStart'], monthly['mean_tot'])
    plt.title('Monthly Average Corners')
    plt.xlabel('Month')
    plt.ylabel("Avg total corners")
    plt.savefig('figs/month_avg_corners.png', dpi=160)

    plt.figure()
    plt.bar(monthly["MonthStart"], monthly["matches"])
    plt.title("Matches per Month")
    plt.xlabel("Month")
    plt.ylabel("Number of matches")
    plt.savefig('figs/monthly_matches.png', dpi=160)


def plot_pred_vs_true(y_pred, y_true):
    plt.figure()
    plt.scatter(y_pred, y_true, alpha=0.6)
    plt.title("Predicted vs Actual Total Corners (Test Split)")
    plt.xlabel("predicted corners")
    plt.ylabel("Truth corners")
    plt.savefig('figs/pred_vs_true.png', dpi=160)


def plot_residual(residuals, bins=31):
    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.title("Residuals, Truth - Prediction")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.savefig('figs/residual.png', dpi=160)


def find_alpha_nb2(y, mu):
    # Var = mu + alpha * mu^2  (NB2)
    numer = np.sum((y - mu)**2 - mu)
    denom = np.sum(mu**2)
    return max(0.0, numer / denom) if denom > 0 else 0.0 




def nb_probs_for_line(mu, line, alpha, r):
    if alpha <= 0:  # fallback to Poisson limits
        if float(line).is_integer():
            k = int(line)
            p_at  = poisson.pmf(k, mu)
            p_u   = poisson.cdf(k-1, mu) if k>0 else 0.0
            p_o   = 1.0 - poisson.cdf(k, mu)
        else:
            k = int(np.floor(line))
            p_at = 0.0
            p_u  = poisson.cdf(k, mu)
            p_o  = 1.0 - p_u
        return p_u, p_at, p_o

    r = 1.0 / alpha                      # shape
    p = r / (r + mu)                     # success prob
    if float(line).is_integer():
        k = int(line)
        p_at = nbinom.pmf(k, r, p)
        p_u  = nbinom.cdf(k-1, r, p) if k>0 else 0.0
        p_o  = 1.0 - nbinom.cdf(k, r, p)
    else:
        k = int(np.floor(line))
        p_at = 0.0
        p_u  = nbinom.cdf(k, r, p)
        p_o  = 1.0 - p_u
    return p_u, p_at, p_o

def poisson_ll(y, mu):
    return float(poisson.logpmf(y, mu).sum())

def nb_ll(y, mu, alpha):
    if alpha <= 0:
        return poisson_ll(y, mu)
    r = 1.0 / alpha             
    p = r / (r + mu)            
    return float(nbinom.logpmf(y, r, p).sum())


def kelly_fraction(p_win, odds, p_push=0.0, frac=0.5):
    b = odds - 1.0
    if b <= 0: return 0.0
    q = 1.0 - p_win - p_push
    if p_push:
       denom = b * (1.0 - p_push)
       if denom <= 0: return 0.0
       numer = p_win * b - q
       f = numer / denom
    else:
       f = (b * p_win - q) / b
    return max(0.0, frac*f) 

def decide_bet_dir(df, min_ev=0.0):
    over_prob  = df['P(Over)']
    under_prob = df['P(Under)']
    push_prob  = df['P(At)']
 
    over_odd  = df['Over'] - 1
    under_odd = df['Under'] - 1
    
    EV_over  = over_prob * (over_odd) - (1 - over_prob - push_prob)
    EV_under = under_prob * (under_odd) - (1 - under_prob - push_prob)

    choose_over  = (EV_over > EV_under) & (EV_over > min_ev)
    choose_under = (EV_under > EV_over) & (EV_under > min_ev)

    df['Bet (U/O)'] = 'No Bet'
   # sel_over  = np.where(choose_over)[0]
   # sel_under = np.where(choose_under)[0]
    
  #  df['Bet (U/O)'].iloc[sel_over]  = 'O'
  #  df['Bet (U/O)'].iloc[sel_under] = 'U'
    df.loc[choose_over,  'Bet (U/O)'] = 'O'
    df.loc[choose_under, 'Bet (U/O)'] = 'U'
    print((df['Bet (U/O)'] == 'O').sum())
    print((df['Bet (U/O)'] == 'U').sum())
    return df

def construct_porfolio(df, wealth=314.0, frac=0.5, bet_cap=None, scale_down_only=True):
    out_df = df.copy()
    PO = out_df['P(Over)'] #np.asarray(P_O, dtype=float)
    PU = out_df['P(Under)']#np.asarray(P_U, dtype=float)
    PA = out_df['P(At)']#np.asarray(P_A, dtype=float)
    Over = out_df['Over']
    Under = out_df['Under']

    mask_o = (out_df['Bet (U/O)'] == 'O')
    mask_u = (out_df['Bet (U/O)'] == 'U')
    stake = np.zeros(len(out_df)) 
    stake[mask_o] = [kelly_fraction(p, o, p_push=push, frac=frac) for p, o, push in zip(PO[mask_o], Over[mask_o], PA[mask_o])]
    stake[mask_u] = [kelly_fraction(p, o, p_push=push, frac=frac) for p, o, push in zip(PU[mask_u], Under[mask_u], PA[mask_u])] 
  
    stake = stake * wealth 
    if bet_cap:
       stake = np.minimum(stake, bet_cap)
    tot_stake = np.sum(stake)
    if tot_stake > wealth: #maybe only scale down for now
       print(f"Total stake {tot_stake}, scaled down to {wealth}")
       stake *= (wealth / tot_stake)   
    out_df['Stake'] = np.round(stake, 3)
    return out_df
    


def cv_tune_alpha(train_df: pd.DataFrame,
                  alphas=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
                  n_splits=5,
                  scoring="neg_mean_poisson_deviance",
                  n_jobs=-1,
                  verbose=0):
    """
    Time-aware CV for GLM ridge alpha (home/away).
    Groups = Season to avoid leakage across years.
    Returns best alphas, best pipelines, and CV tables.
    """
    df = train_df.copy()
    X = df[FEATURES]
    y_home = df["Home_Corners"].astype(float)
    y_away = df["Away_Corners"].astype(float)
    groups = df["Season"].astype(int)

    # If too few seasons, reduce splits
    n_splits = min(n_splits, max(2, groups.nunique()))

    cv = GroupKFold(n_splits=n_splits)
    grid = {"glm__alpha": list(alphas)}

    gs_home = GridSearchCV(
        estimator=build_pipe(),
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
    )
    gs_away = GridSearchCV(
        estimator=build_pipe(),
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
    )

    gs_home.fit(X, y_home, groups=groups)
    gs_away.fit(X, y_away, groups=groups)

    best_alpha_home = gs_home.best_params_["glm__alpha"]
    best_alpha_away = gs_away.best_params_["glm__alpha"]

    cv_home = (pd.DataFrame(gs_home.cv_results_)
                 .loc[:, ["param_glm__alpha","mean_test_score","std_test_score"]]
                 .sort_values("param_glm__alpha"))
    cv_away = (pd.DataFrame(gs_away.cv_results_)
                 .loc[:, ["param_glm__alpha","mean_test_score","std_test_score"]]
                 .sort_values("param_glm__alpha"))

    return {
        "best_alpha_home": float(best_alpha_home),
        "best_alpha_away": float(best_alpha_away),
        "home_pipe": gs_home.best_estimator_,
        "away_pipe": gs_away.best_estimator_,
        "cv_table_home": cv_home.rename(columns={"param_glm__alpha":"alpha",
                                                 "mean_test_score":"neg_mean_poiss_dev"}),
        "cv_table_away": cv_away.rename(columns={"param_glm__alpha":"alpha",
                                                 "mean_test_score":"neg_mean_poiss_dev"}),
    }

#def evaluation()

def extract_bet(df):
    bet = df['Bet (U/O)']
    mask_o = bet == 'O'
    mask_u = bet == 'U'
    mask_B = mask_o | mask_u
 
    p_win = np.zeros(len(df))
    p_push = np.zeros(len(df))
    b = np.zeros(len(df))
    
    p_win[mask_o] = df['P(Over)'][mask_o]
    p_win[mask_u] = df['P(Under)'][mask_u]
    p_push[mask_B] = df['P(At)'][mask_B]

    b[mask_o] = df['Over'][mask_o] - 1.0
    b[mask_u] = df['Under'][mask_u] - 1.0
    
    return p_win, p_push, b


def log_growth(p_win, p_push, b, stakes, wealth=341, eps=1e-12):
    q = 1.0 - p_win - p_push
    w = stakes / wealth
    w = np.clip(w, 0.0, 1.0 - eps)
    return float((p_win * np.log1p(b * w + eps) + q * np.log(1.0 - w + eps)).sum())

def find_best_min(p_win, p_push, b, stakes, wealth=341.0):
   # grid = np.round(np.linspace(0.0, 1.5, 16), 2)
    grid = [0.0, 0.01, 0.02, 0.05]
    best = (-1e99, None, None) 
    total_before = stakes.sum()

    for m in grid:
        stake = stakes.where(stakes >= m, 0.0)
        idx = np.where(stakes >= m)
        G = log_growth(p_win, p_push, b, stake, wealth=wealth)
        if G > best[0]:
           best = (G, m, len(idx))
    return best








def logscore(y_true, mu, alpha, conf=0.95):
    """
    y_true : 1D array of observed total corners (ints)
    mu     : 1D array of predicted means (same length)
    alpha  : NB2 dispersion (>0). If 0, Poisson reduces to NB with r=inf.
    conf   : confidence level for CI (default 0.95)

    Returns:
      avg_log_pois, avg_log_nb, delta_mean, delta_se, (ci_lo, ci_hi)
    """
    y_true = np.asarray(y_true, dtype=int)
    mu     = np.asarray(mu, dtype=float)
    mu     = np.clip(mu, 1e-12, None)

    # per-match log pmfs
    log_pois = poisson.logpmf(y_true, mu)

    if alpha > 0:
        r = 1.0 / alpha
        p = r / (r + mu)
        log_nb = nbinom.logpmf(y_true, r, p)
    else:
        log_nb = log_pois.copy()  # alpha=0 ⇒ Poisson

    # differences (NB − Poisson)
    d = log_nb - log_pois
    n = d.size
    delta_mean = float(d.mean())
    delta_se   = float(d.std(ddof=1) / np.sqrt(n))

    # symmetric (Wald) CI
    z = 1.96 if np.isclose(conf, 0.95) else float(scipy.stats.norm.ppf(0.5*(1+conf)))
    ci = (delta_mean - z*delta_se, delta_mean + z*delta_se)

    return float(log_pois.mean()), float(log_nb.mean()), delta_mean, delta_se, ci
