import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
df = pd.read_excel("data.xlsx") # Or .csv depending on your file

# 2. SELECT ONLY MEANINGFUL FEATURES
# We drop WARD, DISTRICT, PSA, ANC, LAT/LON because 'NEIGHBORHOOD_CLUSTER' captures the location.
# We drop SHIFT because 'HOUR_OF_DAY' is more precise.
features = ["NEIGHBORHOOD_CLUSTER", "HOUR_OF_DAY", "DAY_OF_WEEK", "MONTH_NAME"]
target = "OFFENSE_GROUPED"

# Clean Data (Basic check)
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 3. Save unique options for the App (So we don't need the CSV in the app)
# This allows the dropdowns to be accurate
unique_values = {
    "clusters": sorted(df["NEIGHBORHOOD_CLUSTER"].unique().tolist()),
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "months": ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]
}
joblib.dump(unique_values, "app_options.pkl")

# 4. Build Pipeline
# Categorical columns: Neighborhood, Day, Month
cat_cols = ["NEIGHBORHOOD_CLUSTER", "DAY_OF_WEEK", "MONTH_NAME"]
# Numerical columns: Hour
num_cols = ["HOUR_OF_DAY"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# We use class_weight='balanced' to handle rare crimes better
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 5. Train
print("Training optimized model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 6. Save Model
#joblib.dump(model, "optimized_model.pkl")
# NEW LINE (Add compression):
y_pred = model.predict(X_test)

# Calculate Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")

# Detailed Report (Precision, Recall, F1-Score)
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "optimized_model.pkl", compress=3)
print("Success! Model saved as 'optimized_model.pkl'.")
print("Options saved as 'app_options.pkl'.")