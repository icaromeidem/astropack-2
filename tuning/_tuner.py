from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score, recall_score, precision_score

import time
import numpy as np
import pandas as pd
from astropack.models import create_model


class Tuner:
    def __init__(self, model_type, n_splits, n_repeats, tuning_id, mag_type, param_name, metrics, cutoffs = None):
        self.model_type = model_type
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.tuning_id = tuning_id
        self.mag_type = mag_type
        self.param_name = param_name
        self.metrics = metrics
        self.cutoffs = cutoffs
        
    def evaluate_combination_reg(self, hp_combination):
        """
        Evaluate a STEPE-RF model (Feature Selector + Random Forest) initialized with a
        certain combination of hyperparameters by using a k-fold n-repeat cross validation
        and return the average values and standard deviations of a set of
        metrics (MAE, RMSE, MaxError, R2 Score and elapsed time).

        Keyword arguments:
        Hyperparams - A list containing the combination of hyperpameters to test, in the
                    format [n_features, n_trees, min_samples_leaf, max_features, criterion]

        X - Dataframe containing the input values of the development sample
            (will be split into training and validation samples)

        y - Dataframe containing the target values of the development sample
            (will be split into training and validation samples)

        n_splits - Number of samples that the data will be split into during the k-fold,
                   n-repeat cross-validation (corresponds to k)

        n_repeats - Number of times that the data will be shuffled and split again during
                    the k-fold, n-repeat cross-validation (corresponds to n)

        verbose - Indicates whether the function will print information on the screen or not
        """
        times = []
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = []

        filename = (
            f"results/rf_tuning_{self.mag_type}_{self.param_name}_{self.tuning_id}.csv"
        )

        x_dev = pd.read_csv(
            f"temp_dataframes/x_dev_{self.mag_type}_{self.tuning_id}.csv", index_col=0
        )
        y_dev = pd.read_csv(
            f"temp_dataframes/y_dev_{self.mag_type}_{self.tuning_id}.csv", index_col=0
        )[self.param_name]

        # Initialize the cross-validation function
        kf_splitter = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)

        # Loop through all the training/validation combinations given by the cross-validation function
        for train_index, validation_index in kf_splitter.split(x_dev):
            # Get the training and validation samples
            x_train, x_validation = x_dev.iloc[train_index], x_dev.iloc[validation_index]
            y_train, y_validation = y_dev.iloc[train_index], y_dev.iloc[validation_index]

            # Initialize the model
            pipeline = create_model(self.model_type, hp_combination)
            # Start a timer to time the process
            start_time = time.time()

            # Fit the pipeline to the training data
            pipeline = pipeline.fit(x_train, y_train.values.reshape(len(y_train)))

            # Predict the target values of the validation sample
            predictions = pipeline.predict(x_validation)
            # Stop the timer
            end_time = time.time()
            for metric in self.metrics:
                if metric == 'MAD':
                # Calculate the median and mad for the errors of the model
                    errors = y_validation - predictions
                    mad = np.median(np.abs(errors))
                    metrics[metric].append(mad)
                    
            times.append(end_time - start_time)

        results = pd.DataFrame()
        results[
            ["n_features", "n_trees", "min_samples_leaf", "bootstrap", "max_features"]
        ] = [hp_combination]

        for metric in self.metrics:
            results[f'all_{metric}'] = str(metrics[metric])
            results[f'avg_{metric}'] = np.array(metrics[metric]).mean()
            results[f'std_{metric}'] = np.array(metrics[metric]).std()
            
        results["time"] = np.array(times).mean()

        results.to_csv(filename, index=False, header=False, mode="a")

        return "Success!"

    def evaluate_combination_cla(self, hp_combination):
        """
        Evaluate a STEPE-RF model (Feature Selector + Random Forest) initialized with a
        certain combination of hyperparameters by using a k-fold n-repeat cross validation
        and return the average values and standard deviations of a set of
        metrics (MAE, RMSE, MaxError, R2 Score and elapsed time).

        Keyword arguments:
        Hyperparams - A list containing the combination of hyperpameters to test, in the
                    format [n_features, n_trees, min_samples_leaf, max_features, criterion]

        X - Dataframe containing the input values of the development sample
            (will be split into training and validation samples)

        y - Dataframe containing the target values of the development sample
            (will be split into training and validation samples)

        n_splits - Number of samples that the data will be split into during the k-fold,
                   n-repeat cross-validation (corresponds to k)

        n_repeats - Number of times that the data will be shuffled and split again during
                    the k-fold, n-repeat cross-validation (corresponds to n)

        verbose - Indicates whether the function will print information on the screen or not
        """
        times = []
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = {}
            for cutoff in self.cutoffs:
                metrics[metric][str(cutoff)] = []

        filename = (
            f"results/rf_tuning_{self.mag_type}_{self.param_name}_{self.tuning_id}.csv"
        )

        x_dev = pd.read_csv(
            f"temp_dataframes/x_dev_{self.mag_type}_{self.tuning_id}.csv", index_col=0
        )
        y_dev = pd.read_csv(
            f"temp_dataframes/y_dev_{self.mag_type}_{self.tuning_id}.csv", index_col=0
        )

        # Initialize the cross-validation function
        kf_splitter = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)

        # Loop through all the training/validation combinations given by the cross-validation function
        for train_index, validation_index in kf_splitter.split(x_dev):
            # Get the training and validation samples
            x_train, x_validation = x_dev.iloc[train_index], x_dev.iloc[validation_index]
            y_train, y_validation = y_dev.iloc[train_index], y_dev.iloc[validation_index]

            # Initialize the model
            pipeline = create_model(self.model_type, hp_combination)
            # Start a timer to time the process
            start_time = time.time()

            # Fit the pipeline to the training data
            pipeline = pipeline.fit(x_train, y_train.values.reshape(len(y_train)))

            # Predict the target values of the validation sample
            scores = pipeline.predict_proba(x_validation)
            scores = [x[1] for x in scores]
            
            # Stop the timer
            end_time = time.time()
            
            for cutoff in self.cutoffs:
                predictions = [1 if score >= cutoff else 0 for score in scores]
                for metric in self.metrics:
                    if metric == 'RECALL':
                    # Calculate the recall score of the model
                        recall = recall_score(y_validation, predictions, zero_division = 0)
                        metrics[metric][str(cutoff)].append(recall)
                    elif metric == 'PRECISION':
                    # Calculate the precision score of the model
                        precision = precision_score(y_validation, predictions, zero_division = 0)
                        metrics[metric][str(cutoff)].append(precision)
                    elif metric == 'F1':
                    # Calculate the f1 score of the model
                        f1 = f1_score(y_validation, predictions, zero_division = 0)
                        metrics[metric][str(cutoff)].append(f1)
                        
                times.append(end_time - start_time)

        for cutoff in self.cutoffs:
            results = pd.DataFrame()
            results[
                ["n_features", "n_trees", "min_samples_leaf", "bootstrap", "max_features", "class_weight"]
            ] = [hp_combination]

            results["cutoff"] = cutoff

            for metric in self.metrics:
                results[f'all_{metric}'] = str(metrics[metric][str(cutoff)])
                results[f'avg_{metric}'] = np.array(metrics[metric][str(cutoff)]).mean()
                results[f'std_{metric}'] = np.array(metrics[metric][str(cutoff)]).std()

            results["time"] = np.array(times).mean()

            results.to_csv(filename, index=False, header=False, mode="a")

        return "Success!"