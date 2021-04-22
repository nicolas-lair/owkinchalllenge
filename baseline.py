"""Train the baseline model i.e. a logistic regression on the average of the resnet features and
and make a prediction.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection

from utils import get_params, save_test_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path,
                    help="directory where data is stored", default=Path().absolute())
parser.add_argument("--num_runs", default=3, type=int,
                    help="Number of runs for the cross validation")
parser.add_argument("--num_splits", default=5, type=int,
                    help="Number of splits for the cross validation")


def load_features(filenames, average=False):
    """Load and aggregate the resnet features by the average.

    Args:
        filenames: list of filenames of length `num_patients` corresponding to resnet features

    Returns:
        features: np.array of mean resnet features, shape `(num_patients, 2048)`
    """
    # Load numpy arrays
    features = []
    for f in filenames:
        patient_features = np.load(f)

        # Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]

        if average:
            patient_features = np.mean(patient_features, axis=0)
        features.append(patient_features)

    features = np.stack(features, axis=0)
    return features


if __name__ == "__main__":
    args = parser.parse_args()
    filenames_train, filenames_test, labels_train, ids_test = get_params(args)

    # Get the resnet features and aggregate them by the average
    features_train = load_features(filenames_train, average=True)
    features_test = load_features(filenames_test, average=True)

    # -------------------------------------------------------------------------
    # Use the average resnet features to predict the labels

    # Multiple cross validations on the training set
    aucs = []
    for seed in range(args.num_runs):
        # Use logistic regression with L2 penalty
        estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="liblinear")

        cv = sklearn.model_selection.StratifiedKFold(n_splits=args.num_splits, shuffle=True,
                                                     random_state=seed)

        # Cross validation on the training set
        auc = sklearn.model_selection.cross_val_score(estimator, X=features_train, y=labels_train,
                                                      cv=cv, scoring="roc_auc", verbose=0)

        aucs.append(auc)

    aucs = np.array(aucs)

    print("Predicting weak labels by mean resnet")
    print("AUC: mean {}, std {}".format(aucs.mean(), aucs.std()))

    # -------------------------------------------------------------------------
    # Prediction on the test set

    # Train a final model on the full training set
    estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    estimator.fit(features_train, labels_train)

    preds_test = estimator.predict_proba(features_test)[:, 1]

    # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them in the suitable format
    # to the data challenge platform
    save_test_predictions(ids_test, predictions=preds_test, data_dir=args.data_dir)

