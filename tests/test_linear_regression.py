import numpy as np
from rice2025.supervised_learning.linear_regression import LinearRegression


def synthetic_linear_data():
    # y = 3x + 2
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = 3 * X[:, 0] + 2
    return X, y


def test_closed_form_fit():
    X, y = synthetic_linear_data()
    model = LinearRegression()
    model.fit(X, y)

    # coefficients should be exactly [3]
    assert np.isclose(model.coef_[0], 3.0, atol=1e-6)
    assert np.isclose(model.intercept_, 2.0, atol=1e-6)


def test_predict_shape():
    X, y = synthetic_linear_data()
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert isinstance(preds, np.ndarray)


def test_r2_perfect_fit():
    X, y = synthetic_linear_data()
    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    assert np.isclose(r2, 1.0, atol=1e-12)


def test_ridge_regularization_changes_params():
    X, y = synthetic_linear_data()

    ols = LinearRegression(regularization=0.0)
    ridge = LinearRegression(regularization=10.0)

    ols.fit(X, y)
    ridge.fit(X, y)

    # Ridge should shrink coefficients
    assert abs(ridge.coef_[0]) < abs(ols.coef_[0])


def test_gradient_descent_fit():
    X, y = synthetic_linear_data()

    gd = LinearRegression(
        use_gradient_descent=True,
        learning_rate=0.1,
        max_iter=5000,
        tol=1e-9,
    )
    gd.fit(X, y)

    assert np.isclose(gd.coef_[0], 3.0, atol=1e-2)
    assert np.isclose(gd.intercept_, 2.0, atol=1e-2)


def test_gd_matches_closed_form():
    X, y = synthetic_linear_data()

    ols = LinearRegression()
    gd = LinearRegression(use_gradient_descent=True, learning_rate=0.1, max_iter=5000)

    ols.fit(X, y)
    gd.fit(X, y)

    # Should be very close
    assert np.isclose(gd.coef_[0], ols.coef_[0], atol=1e-2)
    assert np.isclose(gd.intercept_, ols.intercept_, atol=1e-2)
