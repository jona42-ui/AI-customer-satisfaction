import logging
import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin


class HyperparameterOptimization:
    """
    Class for performing hyperparameter optimization.
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Initialize the class with training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize_random_forest(self, trial: optuna.Trial) -> float:
        """Method for optimizing Random Forest."""
        try:
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 5, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )
            model.fit(self.x_train, self.y_train)
            return model.score(self.x_test, self.y_test)
        except Exception as e:
            logging.error("Error optimizing Random Forest: %s", e)
            raise

    def optimize_lightgbm(self, trial: optuna.Trial) -> float:
        """Method for optimizing LightGBM."""
        try:
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 5, 30)
            learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.3)

            model = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
            )
            model.fit(self.x_train, self.y_train)
            return model.score(self.x_test, self.y_test)
        except Exception as e:
            logging.error("Error optimizing LightGBM: %s", e)
            raise

    def optimize_xgboost(self, trial: optuna.Trial) -> float:
        """Method for optimizing XGBoost."""
        try:
            params = {
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            }

            model = xgb.XGBRegressor(**params)
            model.fit(self.x_train, self.y_train)
            return model.score(self.x_test, self.y_test)
        except Exception as e:
            logging.error("Error optimizing XGBoost: %s", e)
            raise


class ModelTrainer:
    """
    Class for training machine learning models.
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Initialize the class with training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _train_model_with_optimization(self, optimizer_func, n_trials: int) -> RegressorMixin:
        """Helper function to optimize and train a model."""
        hyper_opt = HyperparameterOptimization(
            self.x_train, self.y_train, self.x_test, self.y_test
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(optimizer_func, n_trials=n_trials)
        best_params = study.best_trial.params
        logging.info("Best parameters: %s", best_params)
        return best_params

    def random_forest_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """Train a Random Forest model."""
        try:
            if fine_tuning:
                best_params = self._train_model_with_optimization(
                    lambda trial: HyperparameterOptimization(
                        self.x_train, self.y_train, self.x_test, self.y_test
                    ).optimize_random_forest(trial),
                    n_trials=10,
                )
                model = RandomForestRegressor(**best_params)
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=10)
            model.fit(self.x_train, self.y_train)
            return model
        except Exception as e:
            logging.error("Error in training Random Forest: %s", e)
            raise

    def lightgbm_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """Train a LightGBM model."""
        try:
            if fine_tuning:
                best_params = self._train_model_with_optimization(
                    lambda trial: HyperparameterOptimization(
                        self.x_train, self.y_train, self.x_test, self.y_test
                    ).optimize_lightgbm(trial),
                    n_trials=10,
                )
                model = LGBMRegressor(**best_params)
            else:
                model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=20)
            model.fit(self.x_train, self.y_train)
            return model
        except Exception as e:
            logging.error("Error in training LightGBM: %s", e)
            raise

    def xgboost_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """Train an XGBoost model."""
        try:
            if fine_tuning:
                best_params = self._train_model_with_optimization(
                    lambda trial: HyperparameterOptimization(
                        self.x_train, self.y_train, self.x_test, self.y_test
                    ).optimize_xgboost(trial),
                    n_trials=10,
                )
                model = xgb.XGBRegressor(**best_params)
            else:
                model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=20)
            model.fit(self.x_train, self.y_train)
            return model
        except Exception as e:
            logging.error("Error in training XGBoost: %s", e)
            raise
