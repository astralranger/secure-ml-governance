"""Canonical UCI Adult schema, source files and leakage-free splits."""
from pathlib import Path
import os
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(os.getenv("ADULT_DATA_DIR", Path(__file__).resolve().parents[2] / "data/raw"))
NUMERIC_BOUNDS = {
    "age": (17, 100), "fnlwgt": (1, 2_000_000), "education_num": (1, 16),
    "capital_gain": (0, 99999), "capital_loss": (0, 99999), "hours_per_week": (1, 99),
}
CATEGORIES = {
    "workclass": "Private,Self-emp-not-inc,Self-emp-inc,Federal-gov,Local-gov,State-gov,Without-pay,Never-worked".split(","),
    "education": "Bachelors,Some-college,11th,HS-grad,Prof-school,Assoc-acdm,Assoc-voc,9th,7th-8th,12th,Masters,1st-4th,10th,Doctorate,5th-6th,Preschool".split(","),
    "marital_status": "Married-civ-spouse,Divorced,Never-married,Separated,Widowed,Married-spouse-absent,Married-AF-spouse".split(","),
    "occupation": "Tech-support,Craft-repair,Other-service,Sales,Exec-managerial,Prof-specialty,Handlers-cleaners,Machine-op-inspct,Adm-clerical,Farming-fishing,Transport-moving,Priv-house-serv,Protective-serv,Armed-Forces".split(","),
    "relationship": "Wife,Own-child,Husband,Not-in-family,Other-relative,Unmarried".split(","),
    "race": "White,Asian-Pac-Islander,Amer-Indian-Eskimo,Other,Black".split(","),
    "sex": ["Female", "Male"],
    "native_country": "United-States,Cambodia,England,Puerto-Rico,Canada,Germany,Outlying-US(Guam-USVI-etc),India,Japan,Greece,South,China,Cuba,Iran,Honduras,Philippines,Italy,Poland,Jamaica,Vietnam,Mexico,Portugal,Ireland,France,Dominican-Republic,Laos,Ecuador,Taiwan,Haiti,Columbia,Hungary,Guatemala,Nicaragua,Scotland,Thailand,Yugoslavia,El-Salvador,Trinadad&Tobago,Peru,Hong,Holand-Netherlands".split(","),
}
FEATURE_NAMES = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                 "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                 "hours_per_week", "native_country"]
SCHEMA_VERSION = "adult-income-v1"
EXAMPLE = dict(zip(FEATURE_NAMES, [39, "State-gov", 77516, "Bachelors", 13, "Never-married",
                                 "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40, "United-States"]))


def load_adult(split="train"):
    """Preserve UCI's official train/test files; '?' becomes missing, not a category."""
    filename = {"train": "adult.data", "test": "adult.test"}[split]
    path = DATA_DIR / filename
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/" + filename
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        # Validate before caching so an error page cannot become training data.
        if len(raw) < 1_000_000:
            raise ValueError("Unexpected UCI download; supply official files in ADULT_DATA_DIR")
        path.write_bytes(raw)
    frame = pd.read_csv(path, names=FEATURE_NAMES + ["income"], skipinitialspace=True,
                        comment="|", na_values="?", keep_default_na=True)
    labels = frame.pop("income").str.rstrip(".").map({"<=50K": 0, ">50K": 1})
    if labels.isna().any() or len(frame) != {"train": 32561, "test": 16281}[split]:
        raise ValueError(f"Invalid official Adult {split} file")
    return frame, labels.to_numpy(dtype=int)


def training_split():
    frame, labels = load_adult("train")
    return train_test_split(frame.to_numpy(dtype=object), labels, test_size=0.2,
                            random_state=42, stratify=labels)


def sensitive_sex(values):
    return (values[:, FEATURE_NAMES.index("sex")] == "Male").astype(int)
