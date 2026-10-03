import numpy as np
import pandas as pd


def encode(data: pd.DataFrame, fe: bool = True) -> pd.DataFrame:
    d = data.copy()
    d["sex"] = (d["sex"] == "male").astype(int)
    d["smoker"] = (d["smoker"] == "yes").astype(int)
    d = pd.get_dummies(d, columns=["region"], drop_first=True, dtype=int)

    if fe:
        d["obese"] = (d["bmi"] >= 30).astype(int)
        d["smoker_obese"] = d["smoker"] * d["obese"]
        d["smoke_bmi"] = d["smoker"] * d["bmi"]
        d["smoker_age"] = d["smoker"] * d["age"]
        d["age2"] = d["age"] ** 2
        d["bmi_excess"] = np.maximum(d["bmi"] - 30, 0)
        d["smoker_bmi_excess"] = d["smoker"] * d["bmi_excess"]
    return d
