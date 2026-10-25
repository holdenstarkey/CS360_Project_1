
import numpy as np

def best_split(X, y):
    best_error = float("inf")
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]

            error = (np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))

            if error < best_error:
                best_error = error
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def build_tree(X, y, depth=0, max_depth=2):
    #stop splitting and create a leaf
    if depth >= max_depth or len(y) <= 1 or np.all(y == y[0]):
        return float(np.mean(y))

    feature, threshold = best_split(X, y)

    if feature is None:
        return float(np.mean(y))

    mask = X[:, feature] <= threshold

    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X[mask], y[mask], depth + 1, max_depth),
        "right": build_tree(X[~mask], y[~mask], depth + 1, max_depth)
    }


def predict_tree(tree, x):
    if not isinstance(tree, dict):
        return tree

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"], x)
    else:
        return predict_tree(tree["right"], x)


