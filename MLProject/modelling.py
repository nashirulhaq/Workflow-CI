import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Load dataset hasil preprocessing
df = pd.read_csv("medical_cost_preprocessing.csv")

X = df.drop(columns=["annual_medical_cost"])
y = df["annual_medical_cost"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aktifkan autolog
mlflow.autolog()

with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)