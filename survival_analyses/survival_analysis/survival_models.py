from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

# import sys
# from pathlib import Path
# # Add the parent directory to the system path
# sys.path.append(str(Path().resolve().parent))
from pycox.models import DeepHitSingle, CoxPH
import torchtuples as tt
import inspect

from utils import get_concordance_score, get_integrated_brier_score, get_cumulative_dynamic_auc


############################################################
###### Survival Analysis Model for Causal Inference ########
############################################################
class SurvivalModel:
    def __init__(self, model_type, hyperparams=None, extrapolate_median=False, random_seed=42):
        self.model_type = model_type
        self.hyperparams = hyperparams if hyperparams else {}
        self.random_seed = random_seed
        self.extrapolate_median = extrapolate_median
        self.model = self._initialize_model()
        self.time_grid = None
        self.survival_train = None

    def _initialize_model(self):
        # if self.model_type == "KaplanMeier":
            # return KaplanMeierFitter()
        if self.model_type == "CoxPH_pycox":
            return CoxPHFitter()
        elif self.model_type == "CoxPH":
            return CoxPHSurvivalAnalysis()
        elif self.model_type == "RandomSurvivalForest":
            filtered_hyperparams = {k: v for k, v in self.hyperparams.items() if k in inspect.signature(RandomSurvivalForest).parameters}
            return RandomSurvivalForest(**filtered_hyperparams)
        elif self.model_type == "DeepSurv":
            return None  # Initialized in `fit` for PyCox
        elif self.model_type == "DeepHit":
            return None  # Initialized in `fit` for PyCox
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X_train, Y_train, covariate_names=None):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)

        self.survival_train = self._prepare_sksurv_data(Y_train)

        # if self.model_type == "KaplanMeier":
            # self.model.fit(Y_train[:, 0], event_observed=Y_train[:, 1])
        if self.model_type == "CoxPH_pycox":
            Y_train_df = self._prepare_cox_data(X_train, Y_train, covariate_names)
            self.model.fit(Y_train_df, duration_col='time', event_col='event')
        elif self.model_type == "CoxPH":
            self._fit_coxph_ss(X_train, Y_train)
        elif self.model_type == "RandomSurvivalForest":
            self._fit_rsf(X_train, Y_train)
        elif self.model_type == "DeepSurv":
            self._fit_deepsurv(X_train, Y_train)
        elif self.model_type == "DeepHit":
            self._fit_deephit(X_train, Y_train)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _prepare_cox_data(self, X, Y, covariate_names=None):
        if covariate_names is None:
            columns = [f"X{i}" for i in range(X.shape[1])]
        else:
            columns = covariate_names
        df = pd.DataFrame(X, columns=columns)
        df["time"] = Y[:, 0]
        df["event"] = Y[:, 1]
        return df
    
    def _prepare_sksurv_data(self, Y):
        return np.array([(bool(event), time) for time, event in Y], dtype=[("event", "bool"), ("time", "float64")])
    
    def _fit_coxph_ss(self, X_train, Y_train):
        y_train_struct = self._prepare_sksurv_data(Y_train)
        self.model = CoxPHSurvivalAnalysis()
        self.model.fit(X_train, y_train_struct)

    def _fit_rsf(self, X_train, Y_train):
        y_train_struct = self._prepare_sksurv_data(Y_train)
        self.model = RandomSurvivalForest(
            n_estimators=self.hyperparams.get("n_estimators", 100),
            min_samples_split=self.hyperparams.get("min_samples_split", 10),
            min_samples_leaf=self.hyperparams.get("min_samples_leaf", 5),
            n_jobs=-1,
            random_state=self.random_seed,
        )
        self.model.fit(X_train, y_train_struct)

    def _fit_deepsurv(self, X_train, Y_train):
        X_train_split, X_valid, Y_train_split, Y_valid = train_test_split(X_train, Y_train, 
                                                                          test_size=0.2, random_state=self.random_seed)
        net = tt.practical.MLPVanilla(
            X_train_split.shape[1],
            self.hyperparams.get("num_nodes", 256),
            out_features=1,
            batch_norm=self.hyperparams.get("batch_norm", True),
            dropout=self.hyperparams.get("dropout", 0.1),
            output_bias=False
        )
        self.model = CoxPH(net, tt.optim.Adam)
        self.model.optimizer.set_lr(self.hyperparams.get("lr", 0.01))

        val = (X_valid.astype(np.float32), (Y_valid[:, 0].astype(np.float32), Y_valid[:, 1].astype(int)))
        self.model.fit(
            X_train_split.astype(np.float32),
            (Y_train_split[:, 0].astype(np.float32), Y_train_split[:, 1].astype(int)),
            self.hyperparams.get("batch_size", 256),
            self.hyperparams.get("epochs", 512),
            callbacks=[tt.callbacks.EarlyStopping()],
            verbose=False,
            val_data=val, val_batch_size=self.hyperparams.get("batch_size", 256)
        )
        self.model.compute_baseline_hazards()

    def _fit_deephit(self, X_train, Y_train):
        X_train_split, X_valid, Y_train_split, Y_valid = train_test_split(X_train, Y_train, 
                                                                          test_size=0.2, random_state=self.random_seed)
        num_durations = np.unique(Y_train[:, 0]).shape[0]
        labtrans = DeepHitSingle.label_transform(num_durations)
        Y_train_combined = labtrans.fit_transform(Y_train_split[:, 0].astype(np.float32), Y_train_split[:, 1].astype(int))
        Y_val_combined = labtrans.transform(Y_valid[:, 0].astype(np.float32), Y_valid[:, 1].astype(int))

        net = tt.practical.MLPVanilla(
            X_train_split.shape[1],
            self.hyperparams.get("num_nodes", 256),
            labtrans.out_features,
            self.hyperparams.get("batch_norm", True),
            self.hyperparams.get("dropout", 0.1)
        )
        self.model = DeepHitSingle(
            net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts
        )
        self.model.optimizer.set_lr(self.hyperparams.get("lr", 0.01))

        val = (X_valid.astype(np.float32), Y_val_combined)
        self.model.fit(
            X_train_split.astype(np.float32),
            Y_train_combined,
            self.hyperparams.get("batch_size", 256),
            self.hyperparams.get("epochs", 512),
            callbacks=[tt.callbacks.EarlyStopping()],
            verbose=False,
            val_data=val,
        )
        self.time_grid = labtrans.cuts

    def evaluate(self, X_test, Y_test):
        if self.model_type in ["CoxPH", "RandomSurvivalForest", "DeepSurv", "DeepHit"]:
            surv = self.predict_survival_curve(X_test)
            times = surv.index.to_numpy()
            concordance_td = get_concordance_score(Y_test, surv, times)
            
            self.survival_test = self._prepare_sksurv_data(Y_test)
            ibs = get_integrated_brier_score(self.survival_train, self.survival_test, surv, times)
            # td_auc = get_cumulative_dynamic_auc(self.survival_train, self.survival_test, surv, times)

            results = {
                "concordance_td": concordance_td,
                "integrated_brier_score": ibs,
                # "td_auc": td_auc,
                "times": times
            }
            return results
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict_survival_curve(self, X):
        if self.model_type == "CoxPH_pycox":
            return self.model.predict_survival_function(X.astype(np.float32))
        elif self.model_type == "CoxPH":
            return pd.DataFrame(self.model.predict_survival_function(X, return_array=True).T, index=self.model.unique_times_)
        elif self.model_type == "RandomSurvivalForest":
            return pd.DataFrame(self.model.predict_survival_function(X, return_array=True).T, index=self.model.unique_times_)
        elif self.model_type == "DeepSurv":
            return self.model.predict_surv_df(X.astype(np.float32))
        elif self.model_type == "DeepHit":
            return self.model.interpolate(10).predict_surv_df(X.astype(np.float32))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def predict_metric(self, X, metric="median", max_time=np.inf):
        if self.model_type == "CoxPH_pycox":
            if metric == "median" and not(self.extrapolate_median):
                return np.array(self.model.predict_median(X))
            else:
                survival_curves = self.model.predict_survival_function(X.astype(np.float32))
                times = survival_curves.index.to_numpy()
        elif self.model_type == "CoxPH":
            survival_curves = self.model.predict_survival_function(X, return_array=True).T
            times = self.model.unique_times_
        elif self.model_type == "RandomSurvivalForest":
            survival_curves = self.model.predict_survival_function(X, return_array=True).T
            times = self.model.unique_times_
        elif self.model_type == "DeepSurv":
            survival_curves = self.model.predict_surv_df(X.astype(np.float32))
            times = survival_curves.index.to_numpy()
        elif self.model_type == "DeepHit":
            survival_curves = self.model.interpolate(10).predict_surv_df(X.astype(np.float32))
            times = survival_curves.index.to_numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if metric == "median":
            return self._compute_median_survival_times(survival_curves, times)
        elif metric == "mean":
            return self._compute_restricted_mean_survival_times(survival_curves, times, max_time=max_time)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics: ['median', 'mean']")

    def _compute_restricted_mean_survival_times(self, survival_curves, times, max_time=np.inf):
        rmt = []
        for curve in np.array(survival_curves).T:
            rmt.append(np.trapz(y=curve[:np.searchsorted(times, max_time, side='right')], x=times[:np.searchsorted(times, max_time, side='right')]))
        return np.array(rmt)
    
    def _compute_median_survival_times(self, survival_curves, times):
        median_survival_times = []
        for curve in np.array(survival_curves).T:
            # use same code logic as lifelines
            if curve[-1] > .5:
                # extrapolate last time point to be the median survival time
                if self.extrapolate_median: 
                    median_survival_times.append(times[-1])
                else:
                    median_survival_times.append(np.inf)
            else:
                median_survival_times.append(
                    times[np.searchsorted(-curve, [-.5])[0]])
        return np.array(median_survival_times)