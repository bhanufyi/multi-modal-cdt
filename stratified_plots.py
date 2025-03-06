import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "cdt_stratified.xlsx"

# Read the sheets into DataFrames
df_with_info = pd.read_excel(file_path, sheet_name="With Patient Info")
df_no_info = pd.read_excel(file_path, sheet_name="Without Patient Info")

# Define categories (CDT Scores: 0-5)
categories = sorted(df_with_info["Actual Score"].unique())
print(categories)


def compute_regression_accuracy(df, pred_col, actual_col):
    regression_accuracy_per_category = {}

    for category in sorted(
        df[actual_col].unique()
    ):  # Ensure all categories are covered
        df_category = df[df[actual_col] == category]

        if len(df_category) == 0:
            continue  # Skip empty categories

        # Compute absolute errors
        abs_error = np.abs(df_category[pred_col] - df_category[actual_col])

        # Regression Accuracy Formula: 1 - (Sum of Absolute Errors / Sum of Actual Scores)
        regression_accuracy = 1 - (np.sum(abs_error) / np.sum(df_category[actual_col]))

        regression_accuracy_per_category[category] = regression_accuracy * 100

    return regression_accuracy_per_category


# Compute accuracies for both sheets
accuracy_with_info = compute_regression_accuracy(
    df_with_info, "Predicted Score (With Patient Info)", "Actual Score"
)
accuracy_no_info = compute_regression_accuracy(
    df_no_info, "Predicted Score (No Patient Info)", "Actual Score"
)

# Convert results to lists for plotting
x_labels = list(accuracy_with_info.keys())  # Categories (0-5)
y_with_info = list(accuracy_with_info.values())  # Accuracy %
y_no_info = list(accuracy_no_info.values())  # Accuracy %

# --- Plot the graph ---
plt.figure(figsize=(8, 5))
bar_width = 0.35
x = np.arange(len(x_labels))  # Position of bars

# Plot bars
bars_with_info = plt.bar(
    x - bar_width / 2,
    y_with_info,
    width=bar_width,
    label="With Patient Info",
    alpha=0.8,
    color="royalblue",
)

bars_no_info = plt.bar(
    x + bar_width / 2,
    y_no_info,
    width=bar_width,
    label="Without Patient Info",
    alpha=0.8,
    color="orange",
)

# Add actual values on top of bars
for bar in bars_with_info:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

for bar in bars_no_info:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Formatting
plt.xlabel("CDT Score Category", fontsize=12)
plt.ylabel("Accuracy Percentage (%)", fontsize=12)
plt.title("Prediction Accuracy per CDT Score Category", fontsize=14, fontweight="bold")
plt.xticks(x, x_labels)  # Set category labels
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("stratified_accuracy_plot.png", dpi=300, bbox_inches="tight")
