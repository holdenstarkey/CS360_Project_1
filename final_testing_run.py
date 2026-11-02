import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from gradientboostmodel import gradient_boost_model


LEARNING_RATE = 0.0025
MAX_DEPTH = 2
ROUNDS = 2500


def main():
    X_train = pd.read_csv("data/Training_cleaned_features.csv")
    y_train = pd.read_csv("data/Training_cleaned_target.csv")["charges"]

    # Clean the testing data the same way as Data.py
    testing = pd.read_csv("data/Training_Validation_Testing Insurance CS360 - Testing.csv")
    testing = testing.dropna()
    y_test = testing["charges"]
    X_test = testing[["age", "sex", "bmi", "children", "smoker", "region"]]
    X_test = pd.get_dummies(X_test, columns=["sex", "smoker", "region"], dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train.columns = range(X_train.shape[1])
    X_test.columns = X_train.columns

    #train both models with the same settings.
    ours = gradient_boost_model(LEARNING_RATE, MAX_DEPTH)
    ours.forest = []
    ours.train(X_train, y_train, ROUNDS)

    sklearn_model = GradientBoostingRegressor(
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        n_estimators=ROUNDS,
        loss="squared_error",
        random_state=42,
    )
    sklearn_model.fit(X_train, y_train)

    baseline_predictions = np.full(len(y_test), y_train.mean())
    our_predictions = ours.predict_many(X_test)
    sklearn_predictions = sklearn_model.predict(X_test)

    def get_metrics(name, predictions):
        mse = mean_squared_error(y_test, predictions)
        return [name, mse, np.sqrt(mse), r2_score(y_test, predictions)]

    results = pd.DataFrame(
        [
            get_metrics("Mean baseline", baseline_predictions),
            get_metrics("From scratch", our_predictions),
            get_metrics("Sklearn", sklearn_predictions),
        ],
        columns=["Model", "MSE", "RMSE", "R2"],
    )
    print(results.to_string(index=False))
    results.to_csv("final_test_results.csv", index=False)

    results.plot.bar(x="Model", y="RMSE", legend=False, rot=0)
    plt.ylabel("Test RMSE ($)")
    plt.title("Insurance cost prediction error (lower is better)")
    plt.tight_layout()
    plt.savefig("final_test_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
