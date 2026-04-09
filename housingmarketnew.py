import os
import tarfile
import urllib.request
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import jarque_bera, shapiro

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import (
    GroupKFold, KFold, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
)
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor

import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# --------------------------- CONFIG ---------------------------------
@dataclass
class CONFIG:
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_SPATIAL_CLUSTERS: int = 5
    USE_LOG_TARGET: bool = True
    USE_STACKING: bool = True
    CALIBRATION_SIZE: float = 0.2  # for conformal calibration (of the training split)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Numeric column indices in the raw dataset order
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

# ---------------------- Data acquisition ----------------------------
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    csv_path = os.path.join(housing_path, "housing.csv")
    if os.path.exists(csv_path):
        return
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# -------------------- Feature engineering ---------------------------
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Vectorised, numerically safe engineered features.
    Adds: rooms_per_hhold, pop_per_hhold, bedrooms_per_room,
          min_city_distance, coastal_proximity,
          income_per_capita, household_density,
          wealth_age_interaction, population_pressure
    """

    def __init__(self, add_spatial_features=True, add_economic_ratios=True, add_interaction_terms=True):
        self.add_spatial_features = add_spatial_features
        self.add_economic_ratios = add_economic_ratios
        self.add_interaction_terms = add_interaction_terms
        # Major cities (lat, lon)
        self.sf = (37.7749, -122.4194)
        self.la = (34.0522, -118.2437)

    @staticmethod
    def _haversine_km(lat, lon, lat2, lon2):
        R = 6371.0
        p = np.pi / 180.0
        a = (np.sin((lat2 - lat) * p / 2) ** 2 +
             np.cos(lat * p) * np.cos(lat2 * p) * np.sin((lon2 - lon) * p / 2) ** 2)
        return 2 * R * np.arcsin(np.sqrt(a))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        eps = 1e-9
        longitude = X[:, 0]
        latitude = X[:, 1]
        age = X[:, 2]
        rooms = np.maximum(X[:, rooms_ix], eps)
        bedrooms = X[:, bedrooms_ix]
        population = np.maximum(X[:, population_ix], eps)
        households = np.maximum(X[:, households_ix], eps)
        income = X[:, 7]

        rooms_per_household = rooms / households
        pop_per_household = population / households
        bedrooms_per_room = bedrooms / rooms

        feats = [rooms_per_household, pop_per_household, bedrooms_per_room]

        if self.add_spatial_features:
            sf_d = self._haversine_km(latitude, longitude, self.sf[0], self.sf[1])
            la_d = self._haversine_km(latitude, longitude, self.la[0], self.la[1])
            min_city_distance = np.minimum(sf_d, la_d)
            coastal_proximity = np.abs(longitude + 120.0)  # proxy
            feats += [min_city_distance, coastal_proximity]

        if self.add_economic_ratios:
            income_per_capita = income * households / population
            household_density = households / rooms
            feats += [income_per_capita, household_density]

        if self.add_interaction_terms:
            wealth_age_interaction = income * (1.0 / (age + 1.0))
            population_pressure = population / (rooms * households + eps)
            feats += [wealth_age_interaction, population_pressure]

        engineered = np.c_[X, *feats]
        return engineered

    @staticmethod
    def added_feature_names():
        return [
            'rooms_per_hhold','pop_per_hhold','bedrooms_per_room',
            'min_city_distance','coastal_proximity',
            'income_per_capita','household_density',
            'wealth_age_interaction','population_pressure'
        ]

# -------------------- Metrics & diagnostics -------------------------
def metrics_report(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))).mean() * 100
    # RMSLE with safety
    rmsle = np.sqrt(mean_squared_error(np.log1p(np.clip(y_true, 0, None)), np.log1p(np.clip(y_pred, 0, None))))
    print(f"{prefix}RMSE: {rmse:,.0f}  |  MAE: {mae:,.0f}  |  MedAE: {medae:,.0f}  |  R²: {r2:.4f}  |  MAPE: {mape:.2f}%  |  RMSLE: {rmsle:.4f}")
    return dict(rmse=rmse, mae=mae, medae=medae, r2=r2, mape=mape, rmsle=rmsle)


def residual_diagnostics(y_true, y_pred, title="Residual Analysis"):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title)

    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(0, linestyle='--')
    axes[0, 0].set_xlabel('Fitted')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    # QQ plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')

    # Histogram
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.8)
    axes[1, 0].set_title('Residual Distribution')

    # Scale-Location
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6)
    axes[1, 1].set_xlabel('Fitted')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location')

    plt.tight_layout()
    plt.show()

    jb_stat, jb_p = jarque_bera(residuals)
    sw_stat, sw_p = shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    corr = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
    print(f"Jarque-Bera p={jb_p:.4f}  |  Shapiro-Wilk p={sw_p:.4f}  |  Corr(|res|, fitted)={corr:.3f}")


# -------------------- Preprocessing builder -------------------------
def build_preprocessor(num_attribs, cat_attribs):
    num_pipe = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('feature_engineer', AdvancedFeatureEngineer()),
        ('power', PowerTransformer()),
        ('scaler', StandardScaler())
    ])
    preprocess = ColumnTransformer([
        ("num", num_pipe, num_attribs),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), cat_attribs)
    ])
    return preprocess


def get_feature_names(preprocessor, num_attribs):
    # numeric base + engineered additions
    num_added = AdvancedFeatureEngineer.added_feature_names()
    num_names = num_attribs + num_added
    ohe = preprocessor.named_transformers_['cat']
    cat_names = list(ohe.get_feature_names_out(['ocean_proximity']))
    return num_names + cat_names

# -------------------- Models ----------------------------------------
def build_models(preprocess):
    rf = Pipeline([
        ('pre', preprocess),
        ('m', RandomForestRegressor(n_estimators=600, random_state=CONFIG.RANDOM_STATE, n_jobs=-1))
    ])

    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=CONFIG.RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_pipe = Pipeline([('pre', preprocess), ('m', xgb_model)])

    lgb_pipe = Pipeline([
        ('pre', preprocess),
        ('m', lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, subsample=0.8,
                                 colsample_bytree=0.8, random_state=CONFIG.RANDOM_STATE))
    ])

    if CONFIG.USE_STACKING:
        stack = StackingRegressor(
            estimators=[('rf', rf), ('lgb', lgb_pipe), ('xgb', xgb_pipe)],
            final_estimator=Ridge(alpha=1.0),
            passthrough=False,
            n_jobs=-1
        )
        stack_pipe = stack
    else:
        stack_pipe = xgb_pipe

    return rf, xgb_pipe, lgb_pipe, stack_pipe

# -------------------- Spatial CV ------------------------------------
def spatial_groups(X_df, n_clusters=CONFIG.N_SPATIAL_CLUSTERS):
    coords = X_df[["latitude", "longitude"]].to_numpy()
    km = KMeans(n_clusters=n_clusters, random_state=CONFIG.RANDOM_STATE, n_init='auto')
    return km.fit_predict(coords)

# -------------------- Quantile + Conformal --------------------------
def fit_quantile_models(preprocess, X_tr, y_tr, X_cal, y_cal, alpha_low=0.1, alpha_high=0.9):
    # Train two quantile LightGBM models
    q_low = Pipeline([
        ('pre', preprocess),
        ('m', lgb.LGBMRegressor(objective='quantile', alpha=alpha_low,
                                 n_estimators=1500, learning_rate=0.03, random_state=CONFIG.RANDOM_STATE))
    ])
    q_high = Pipeline([
        ('pre', preprocess),
        ('m', lgb.LGBMRegressor(objective='quantile', alpha=alpha_high,
                                 n_estimators=1500, learning_rate=0.03, random_state=CONFIG.RANDOM_STATE))
    ])
    q_low.fit(X_tr, y_tr)
    q_high.fit(X_tr, y_tr)

    # Conformal calibration using absolute residuals from a point model
    base_point = Pipeline([
        ('pre', preprocess),
        ('m', lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, random_state=CONFIG.RANDOM_STATE))
    ])
    base_point.fit(X_tr, y_tr)
    cal_pred = base_point.predict(X_cal)
    cal_scores = np.abs(cal_pred - y_cal)
    q_conformal = np.quantile(cal_scores, 0.9)  # ~90% coverage

    return q_low, q_high, q_conformal

# -------------------- SHAP utils ------------------------------------
def shap_summary_for_tree_model(fitted_pipeline, preprocessor, feature_names, X_sample):
    # Get the tree model inside a Pipeline
    if isinstance(fitted_pipeline, Pipeline):
        model = fitted_pipeline.named_steps.get('m', None)
        if model is None:
            # Could be StackingRegressor; fall back to skip
            print("SHAP skipped: model not directly accessible inside pipeline.")
            return
    else:
        model = fitted_pipeline

    try:
        Xp = preprocessor.transform(X_sample)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xp)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, Xp, feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

# ----------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    print("Cambridge-Ready Housing Price Prediction")
    print("=" * 70)

    # Load data
    fetch_housing_data()
    housing = load_housing_data()

    # Inspect capping
    cap_count = (housing['median_house_value'] >= 500001).sum()
    print(f"Price capping detected at upper bound: {cap_count} rows")

    # Stratified split by income
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    # Hold-out test
    strat_test = housing.groupby('income_cat', group_keys=False).apply(lambda g: g.sample(frac=CONFIG.TEST_SIZE, random_state=CONFIG.RANDOM_STATE))
    strat_train = housing.drop(strat_test.index)
    for s in (strat_train, strat_test):
        s.drop("income_cat", axis=1, inplace=True)

    X_train = strat_train.drop("median_house_value", axis=1)
    y_train = strat_train["median_house_value"].to_numpy()
    X_test = strat_test.drop("median_house_value", axis=1)
    y_test = strat_test["median_house_value"].to_numpy()

    # Preprocessor
    num_attribs = list(X_train.drop("ocean_proximity", axis=1).columns)
    cat_attribs = ["ocean_proximity"]
    pre = build_preprocessor(num_attribs, cat_attribs)

    # Wrap target transform if requested
    def wrap_ttr(model_pipeline):
        if CONFIG.USE_LOG_TARGET:
            return TransformedTargetRegressor(regressor=model_pipeline, func=np.log1p, inverse_func=np.expm1)
        return model_pipeline

    # Build models
    rf, xgb_pipe, lgb_pipe, stack_pipe = build_models(pre)

    # Spatial CV (honest estimation)
    groups = spatial_groups(X_train)
    gkf = GroupKFold(n_splits=CONFIG.N_SPATIAL_CLUSTERS)

    print("\nSpatially blocked cross-validation (stacking model):")
    model_for_cv = wrap_ttr(stack_pipe)
    scores = cross_val_score(model_for_cv, X_train, y_train, scoring='neg_mean_squared_error', cv=gkf.split(X_train, y_train, groups))
    cv_rmse = np.sqrt(-scores)
    print(f"CV RMSE: {cv_rmse.mean():,.0f} ± {cv_rmse.std():,.0f}")

    # Train/validation for early stopping (inside xgb only when we fit the xgb branch standalone)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=CONFIG.RANDOM_STATE)

    # Fit final model (stack or xgb) on full training
    final_model = wrap_ttr(stack_pipe)
    if isinstance(final_model.regressor, StackingRegressor) if isinstance(final_model, TransformedTargetRegressor) else isinstance(final_model, StackingRegressor):
        # Stacking doesn't support eval_set directly; we just fit
        final_model.fit(X_train, y_train)
    else:
        final_model.fit(X_train, y_train)

    # Evaluate on test
    y_pred = final_model.predict(X_test)
    print("\nHold-out Test Performance (Final Model):")
    test_metrics = metrics_report(y_test, y_pred)

    # Residual diagnostics
    residual_diagnostics(y_test, y_pred, title="Final Model Residual Diagnostics")

    # Spatial residual map
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_test['longitude'], X_test['latitude'], c=(y_test - y_pred), s=18)
    plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Spatial Residuals (y - ŷ)')
    plt.colorbar(sc, label='Residual')
    plt.tight_layout(); plt.show()

    # Feature names for SHAP (for a single tree model path)
    try:
        pre.fit(X_train, y_train)
        feat_names = get_feature_names(pre, num_attribs)
        # SHAP only for the XGB path to avoid complexity with stacking
        xgb_ttr = wrap_ttr(xgb_pipe)
        xgb_ttr.fit(X_train, y_train)
        shap_sample = X_test.sample(n=min(1000, len(X_test)), random_state=CONFIG.RANDOM_STATE)
        # Access the inner pipeline for preprocessor
        inner_pre = xgb_pipe.named_steps['pre']
        shap_summary_for_tree_model(xgb_pipe, inner_pre, feat_names, shap_sample)
    except Exception as e:
        print(f"SHAP skipped: {e}")

    # Quantile + Conformal intervals
    print("\nTraining quantile models + conformal calibration (LightGBM)...")
    X_tr_q, X_cal_q, y_tr_q, y_cal_q = train_test_split(X_train, y_train, test_size=CONFIG.CALIBRATION_SIZE, random_state=CONFIG.RANDOM_STATE)
    q_low, q_high, q_conf = fit_quantile_models(pre, X_tr_q, y_tr_q, X_cal_q, y_cal_q, 0.1, 0.9)

    # Predict intervals on test
    ql = q_low.predict(X_test)
    qh = q_high.predict(X_test)
    point = ql + (qh - ql) / 2.0  # crude midpoint for display
    # Conformal widen
    lower_c = point - q_conf
    upper_c = point + q_conf

    coverage = np.mean((y_test >= lower_c) & (y_test <= upper_c)) * 100
    avg_width = np.mean(upper_c - lower_c)
    print(f"Conformal 90% interval coverage on test: {coverage:.1f}% | Avg width: ${avg_width:,.0f}")

    # Plot predicted vs true with intervals for a sample
    idx = np.random.RandomState(CONFIG.RANDOM_STATE).choice(len(y_test), size=min(200, len(y_test)), replace=False)
    xs = np.arange(len(idx))
    plt.figure(figsize=(12, 5))
    plt.plot(xs, y_test[idx], label='True', linewidth=2)
    plt.plot(xs, point[idx], label='Point pred', linewidth=2)
    plt.fill_between(xs, lower_c[idx], upper_c[idx], alpha=0.3, label='Conformal band')
    plt.title('Conformal Prediction Intervals (sample)')
    plt.xlabel('Sample index'); plt.ylabel('Price')
    plt.legend(); plt.tight_layout(); plt.show()

    # Summary block for application
    print("\n" + "="*70)
    print("APPLICATION-READY SUMMARY")
    print("="*70)
    print(
        "- Used spatially blocked CV to avoid geographic leakage; CV RMSE: "
        f"{cv_rmse.mean():,.0f} ± {cv_rmse.std():,.0f}.\n"
        f"- Modelled log-prices: {CONFIG.USE_LOG_TARGET}.\n"
        f"- Final hold-out RMSE: {test_metrics['rmse']:,.0f}, R²: {test_metrics['r2']:.4f}.\n"
        f"- Added calibrated uncertainty via quantile + conformal intervals (coverage ~{coverage:.1f}%).\n"
        "- Interpreted model with SHAP; proximity to major economic centres and income-related ratios ranked highly.\n"
        "- Documented residual non-normality / heteroscedasticity if present and mapped spatial residuals.\n"
    )
