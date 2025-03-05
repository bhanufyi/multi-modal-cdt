import pandas as pd
import numpy as np

# Load the saved Excel file
file_path = "nhats_clock_drawing_scores_comparison.xlsx"  # Use the correct file path
df = pd.read_excel(file_path)

# Extract relevant columns
actual_scores = df["Actual Clock Drawing Score"]
predicted_scores = df["Predicted Clock Drawing Score"]

# Drop rows where predictions are NaN
valid_rows = df.dropna(
    subset=["Actual Clock Drawing Score", "Predicted Clock Drawing Score"]
)

# Compute absolute errors
absolute_errors = abs(
    valid_rows["Actual Clock Drawing Score"]
    - valid_rows["Predicted Clock Drawing Score"]
)

# Compute accuracy as per regression
regression_accuracy = 1 - (
    absolute_errors.sum() / valid_rows["Actual Clock Drawing Score"].sum()
)

# Compute Mean Absolute Error (MAE)
mae = absolute_errors.mean()

# Compute Mean Squared Error (MSE)
mse = (
    (
        valid_rows["Actual Clock Drawing Score"]
        - valid_rows["Predicted Clock Drawing Score"]
    )
    ** 2
).mean()

# Compute RMSE
rmse = np.sqrt(mse)

# Print results
print(f"Regression Accuracy: {regression_accuracy:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Save metrics back to the Excel file
metrics_data = {
    "Metric": [
        "Regression Accuracy",
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
    ],
    "Value": [regression_accuracy, mae, rmse],
}

df_metrics = pd.DataFrame(metrics_data)

with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
    df_metrics.to_excel(writer, sheet_name="Regression Metrics", index=False)

print("Regression metrics saved to the Excel file.")
