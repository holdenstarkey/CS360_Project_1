"""Run with: python3 -m pytest -v test_gradient_boosting.py"""

import pytest

import numpy as np
import pandas as pd

import helpers
from gradientboostmodel import gradient_boost_model
from regression_tree import build_tree


def tree_depth(tree):
    if not isinstance(tree, dict):
        return 0
    return 1 + max(tree_depth(tree["left"]), tree_depth(tree["right"]))


def test_tree_depth():
    X = np.arange(16).reshape(-1, 1)
    y = np.arange(16) ** 2

    for depth in range(4):
        tree = build_tree(X, y, max_depth=depth)
        assert tree_depth(tree) == depth

def test_residuals():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([8.0, 23.0, 30.0])
    np.testing.assert_allclose(
        helpers.residuals(actual, predicted), [2.0, -3.0, 0.0]
    )

def test_prediction_updates():
    X = pd.DataFrame([0.0, 1.0])
    y = np.array([0.0, 10.0])

    expected = [[5.0, 5.0], [4.5, 5.5], [4.05, 5.95]]
    for rounds in range(3):
        model = gradient_boost_model(lr=0.1, maxheight=1)
        model.forest = []
        model.train(X, y, iterations=rounds)
        np.testing.assert_allclose(model.predict_many(X), expected[rounds])

def test_no_split_overlap():
    training_data = pd.read_csv("data/Training_Validation_Testing Insurance CS360 - Training.csv")
    validation_data = pd.read_csv("data/Training_Validation_Testing Insurance CS360 - Validation.csv")
    testing_data = pd.read_csv("data/Training_Validation_Testing Insurance CS360 - Testing.csv")

    assert not training_data.empty
    assert not validation_data.empty
    assert not testing_data.empty

    assert training_data.merge(validation_data).empty
    assert training_data.merge(testing_data).empty
    assert validation_data.merge(testing_data).empty

def test_mse():
    assert helpers.mse(np.array([1.0, -2.0, 3.0])) == pytest.approx(14 / 3)
    assert helpers.mse(np.array([0.0, 0.0])) == 0.0
