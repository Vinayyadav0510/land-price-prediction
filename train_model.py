import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("models/Land_data.csv")


print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("Price", axis=1)

y = df["Price"]


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numerical_columns = [
    "City Dist",
    "Road Dist",
    "Area",
    "Town Dist",
    "Market Dist"
]


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "Land Type",
    "Soil",
    "Water",
    "Road",
    "Electricity",
    "Irrigation",
    "Crop",
    "Ownership"
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),

        (
            "numerical",
            "passthrough",
            numerical_columns
        )

    ]
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(

    n_estimators=300,

    random_state=42,

    n_jobs=-1

)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ]

)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


print("\nTraining model...")


# ============================================================
# TRAIN
# ============================================================

pipeline.fit(
    X_train,
    y_train
)


print("Training completed!")


# ============================================================
# PREDICTION
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "models/land_price_model.pkl"
)


print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print(
    "models/land_price_model.pkl"
)