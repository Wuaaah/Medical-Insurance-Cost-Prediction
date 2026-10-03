from eda import load_data, loss_curve, predicted_vs_actual, residual
from preprocessing import encode
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import joblib


def predict(w, X, b):
    return X @ w + b

def mse(w, b, X, y):
    y_hat = predict(w, X, b)
    return np.mean((y_hat - y) ** 2)

def gradients(w, b, X, y):
    n = len(y)
    error = predict(w, X, b) - y
    dw = (2 / n) * (X.T @ error)
    db = (2 / n) * np.sum(error)
    return dw, db

def gradient_descent(X, y, lr=0.05, epochs=5000, print_every=1000):
    w = np.zeros(X.shape[1])
    b = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        dw, db = gradients(w, b, X, y)
        w -= lr * dw
        b -= lr * db

        loss = mse(w, b, X, y)
        history.append(loss)
        if epoch == 1 or epoch % print_every == 0:
            print(f"epoch {epoch:5d} | MSE (scaled) = {loss:.5f}")
    
    return w, b, history

def fit_scaler(X: pd.DataFrame):
    mu = X.mean()
    sd = X.std().replace(0,1)
    return mu, sd

def apply_scaler(X: pd.DataFrame, mu, sd):
    return((X-mu)/sd).values


## Ridge regression

def ridge_loss(w, b, X, y, lambda_):
    y_hat = predict(w, X, b)
    mse = np.mean((y_hat - y) ** 2)
    penalty = lambda_ * np.sum(w ** 2)
    return mse + penalty

def ridge_gradients(w, b, X, y, lambda_):
    n = len(y)
    error = predict(w, X, b) - y
    dw = (2/n) * (X.T @ error) + 2 * lambda_ * w
    db = (2/n) * np.sum(error)
    return dw, db

def ridge_gradient_descent(X, y, lr = 0.05, epochs=5000, lambda_ = 0.1, print_every = 1000):
    w = np.zeros(X.shape[1])
    b = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        dw, db = ridge_gradients(w, b, X, y, lambda_)
        w -= lr * dw
        b -= lr * db
        loss = ridge_loss(w, b, X, y, lambda_)
        history.append(loss)
        if epoch == 1 or epoch % print_every == 0:
            print(f"epoch {epoch:5d} | Ridge Loss (scaled) = {loss:.5f}")
    return w, b, history



def save_model(w, b, mu, sd, y_mu, y_sd, cols, path="model.pkl"):
    payload = {"w": w, "b": b, "mu": mu, "sd": sd,
               "y_mu": y_mu, "y_sd": y_sd, "cols": list(cols)}
    joblib.dump(payload, path)
    print(f"Model saved to {path}")


def load_and_predict(person: dict, path="model.pkl") -> float:
    """Load saved model and predict charges for one person.

    Args:
        person: dict with keys age, sex, bmi, children, smoker, region
        path:   path to the saved .pkl file
    Returns:
        Predicted insurance charge in USD
    """
    m   = joblib.load(path)
    row = pd.DataFrame([person])
    row = encode(row).reindex(columns=m["cols"], fill_value=0).astype(float)
    row_s = ((row - m["mu"]) / m["sd"]).values
    pred_s = row_s @ m["w"] + m["b"]
    return float(pred_s[0] * m["y_sd"] + m["y_mu"])




def predict_new(person: dict, path="model.pkl") -> None:
    """Pretty-print a prediction for one person."""
    charge = load_and_predict(person, path)
    print(f"  {person['age']}yo {person['sex']}, BMI {person['bmi']}, "
          f"{person['children']} child(ren), smoker={person['smoker']}, "
          f"{person['region']}  ->  ${charge:,.2f}")


def main():
    df = load_data()
    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X["smoker"])

    Xtr = encode(X_train).astype(float)
    Xte = encode(X_test).reindex(columns=Xtr.columns, fill_value=0).astype(float)

    mu, sd = fit_scaler(Xtr)
    Xtr_s = apply_scaler(Xtr, mu, sd)
    Xte_s = apply_scaler(Xte, mu, sd)

    y_mu, y_sd = y_train.mean(), y_train.std()
    ytr_s = ((y_train - y_mu ) / y_sd).values


    ridge_sk = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="r2")
    ridge_sk.fit(Xtr_s, ytr_s)
    best_alpha = ridge_sk.best_params_["alpha"]
    print(f"Best sklearn alpha: {best_alpha}")

    lasso_sk = GridSearchCV(Lasso(max_iter=10000), {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="r2")
    lasso_sk.fit(Xtr_s, ytr_s)
    lasso_pred = lasso_sk.best_estimator_.predict(Xte_s) * y_sd + y_mu
    


    best_lambda = best_alpha / (2 * len(ytr_s))   
    print(f"Rescaled lambda_ for GD: {best_lambda:.6f}")

    w, b, history = ridge_gradient_descent(Xtr_s, ytr_s, lambda_=best_lambda)
    loss_curve(history)

    pred = predict(w, Xte_s, b) * y_sd + y_mu
    predicted_vs_actual(y_test, pred)

    residuals = pred - y_test
    residual(pred, residuals)

    out = X_test.copy()
    out["actual"] = y_test.values
    out["pred"] = pred
    out["residuals"] = pred - y_test.values

    print(out[out["residuals"] < -7000].sort_values("residuals")) 

    print("\n--- From scratch (test set) ---")
    print(f"R2  : {r2_score(y_test, pred):.4f}")
    print(f"MAE : {mean_absolute_error(y_test, pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):,.2f}")

    print("\n--- sklearn Lasso (test set) ---")
    print(f"R2  : {r2_score(y_test, lasso_pred):.4f}")
    print(f"MAE : {mean_absolute_error(y_test, lasso_pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lasso_pred)):,.2f}")
    print(f"Best Lasso alpha: {lasso_sk.best_params_['alpha']}")
   
    print("Non-zero weights:", np.sum(lasso_sk.best_estimator_.coef_ != 0))
 
   
    sk = LinearRegression().fit(Xtr, y_train)
    sk_pred = sk.predict(Xte)
    print("\n--- sklearn LinearRegression (test set) ---")
    print(f"R2  : {r2_score(y_test, sk_pred):.4f}")
    print(f"MAE : {mean_absolute_error(y_test, sk_pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, sk_pred)):,.2f}")
 
    
    print("\n--- Weights (standardized features) ---")
    for name, weight in sorted(zip(Xtr.columns, w), key=lambda t: -abs(t[1])):
        print(f"{name:20s} {weight:8.4f}")
 
   
    print("\n========== 5-FOLD CROSS VALIDATION ==========")

    X_full   = encode(X).astype(float)
    mu_full, sd_full = fit_scaler(X_full)
    X_full_s = apply_scaler(X_full, mu_full, sd_full)
    y_arr    = y.values

    cv_models = {
        "LinearRegression"  : LinearRegression(),
        "Ridge (alpha=10)"  : Ridge(alpha=10),
        "Lasso (alpha=0.01)": Lasso(alpha=0.01, max_iter=10000),
    }

    print(f"{'Model':<22}  {'CV R²':>8}  {'± std':>7}")
    print("-" * 44)
    for cv_name, cv_model in cv_models.items():
        scores = cross_val_score(cv_model, X_full_s, y_arr, cv=5, scoring="r2")
        print(f"{cv_name:<22}  {scores.mean():>8.4f}  +/-{scores.std():>6.4f}")

    
    save_model(w, b, mu, sd, y_mu, y_sd, Xtr.columns)

    
    print("\n========== PREDICT NEW INPUTS ==========")
    examples = [
        {"age": 35, "sex": "male",   "bmi": 28.5, "children": 2, "smoker": "no",  "region": "southwest"},
        {"age": 52, "sex": "female", "bmi": 34.1, "children": 0, "smoker": "yes", "region": "southeast"},
        {"age": 22, "sex": "male",   "bmi": 21.0, "children": 0, "smoker": "no",  "region": "northwest"},
    ]
    for person in examples:
        predict_new(person)


if __name__ == "__main__":
    main()
