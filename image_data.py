import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import openai
import base64

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# --- Mappings ---
race_mapping = {
    1: "White, non-Hispanic",
    2: "Black, non-Hispanic",
    3: "Other (Am Indian/Asian/Native Hawaiian/Pacific Islander/Other), non-Hispanic",
    4: "Hispanic",
    5: "More than one (DKRF primary)",
    6: "DKRF",
    -9: "Missing",
}

occupation_mapping = {
    1: "Management Occupations",
    2: "Business and Financial Operations",
    3: "Computer and Mathematical Occupations",
    4: "Architecture and Engineering",
    5: "Life, Physical, and Social Sciences",
    6: "Community and Social Services",
    7: "Legal Occupations",
    8: "Education, Training, and Library",
    9: "Arts, Design, Entertainment, Sports, Media",
    10: "Healthcare Practitioners and Technical",
    11: "Healthcare Support Occupations",
    12: "Protective Service Occupations",
    13: "Food Preparation and Serving",
    14: "Building and Grounds Cleaning and Maintenance",
    15: "Personal Care and Service",
    16: "Sales and Related Occupations",
    17: "Office and Administrative Support",
    18: "Farming, Fishing, and Forestry",
    19: "Construction and Extraction",
    20: "Installation, Maintenance, and Repair",
    21: "Production Occupations",
    22: "Transportation and Material Moving",
    23: "Military Specific Occupations",
    24: "No Current Occupation (Unemployed, No Work in Last 5 Years, Never Worked)",
    25: "Blank Field",
    26: "Code Did Not Match",
    94: "Uncodable",
    95: "Never Worked Entire Life",
    96: "Homemaker/Raised Children",
    -7: "RF (Refused)",
    -8: "DK (Don’t Know)",
    -1: "Inapplicable",
    -9: "Missing",
}


# --- Function to Convert Image to Base64 ---
def encode_image(image_path):
    """Encodes an image file to Base64 for sending it to OpenAI's GPT-4o."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# --- Filter DataFrame to Only Rows with Existing Images ---
def image_exists(row):
    sample_id = row["NHATS SAMPLED PERSON ID"]
    image_filename = f"{sample_id}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    return os.path.exists(image_path)


# --- Function to Score Clock Drawing Using GPT-4o ---
def score_clock_drawing(
    image_path, age_group=None, race=None, education=None, use_patient_info=True
):
    """
    Sends the clock drawing image (with or without patient context) to GPT-4o for scoring.
    """
    json_template = '{"score": 0.0}'

    # Define the system prompt for GPT-4o
    system_prompt = (
        "You are a neuropsychology expert analyzing digital clock drawings. "
        "Evaluate the provided clock drawing test and assign a score from 0 to 5. "
        "The score reflects cognitive performance, considering standard neuropsychology criteria. "
        "Respond strictly in JSON format using the template: " + json_template
    )

    # Construct the user prompt
    if use_patient_info:
        user_prompt = (
            f"Patient Information:\n"
            f"- Age Group: {age_group}\n"
            f"- Race: {race}\n"
            f"- Education: {education}\n\n"
            "This is a clock from a neuropsychology assessment known as the Digital Clock Drawing Test. "
            "Please analyze the provided image and assign a score (0-5) based on cognitive assessment measures.\n"
            f"Respond with JSON in this format: {json_template}."
        )
    else:
        user_prompt = (
            "This is a clock from a neuropsychology assessment known as the Digital Clock Drawing Test. "
            "Please analyze the provided image and assign a score (0-5) based on cognitive assessment measures.\n"
            f"Respond with JSON in this format: {json_template}."
        )

    # Encode image to Base64
    base64_image = encode_image(image_path)

    try:
        # Send request to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
                    ],
                },
            ],
            temperature=0.0,
        )

        response_content = response.choices[0].message.content
        result = json.loads(response_content)
        score = result.get("score", None)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        score = None

    return score


# --- Process Each Row and Get Scores ---
def process_scoring(df_filtered, use_patient_info=True):
    predicted_scores = []
    score_differences = []
    actual_scores = []

    for idx, row in df_filtered.iterrows():
        sample_id = row["NHATS SAMPLED PERSON ID"]
        image_filename = f"{sample_id}.jpg"
        image_path = os.path.join(image_folder, image_filename)

        # Retrieve patient info if using patient details
        if use_patient_info:
            age_group = row.get("R13 WB3 AGE YOU FEEL MOST OF TIME", "Unknown")
            race = row.get("R13 D RACE AND HISPANIC ETHNICITY WHEN ADDED", "Unknown")
            education = row.get("R13 D LONGEST OCCUPATION CATEGORY", "Unknown")
        else:
            age_group, race, education = None, None, None

        # Get GPT-4o predicted score
        pred_score = score_clock_drawing(
            image_path, age_group, race, education, use_patient_info
        )
        predicted_scores.append(pred_score)

        # Compare with actual score
        excel_score = row["R13 D SCORE OF CLOCK DRAWING TEST"]
        actual_scores.append(excel_score)

        if pred_score is not None and pd.notna(excel_score):
            diff = pred_score - excel_score
        else:
            diff = None
        score_differences.append(diff)

    # Create DataFrame with results
    result_df = df_filtered.copy()
    if use_patient_info:
        result_df["Predicted Score (With Patient Info)"] = predicted_scores
    else:
        result_df["Predicted Score (No Patient Info)"] = predicted_scores
    result_df["Actual Score"] = actual_scores
    result_df["Score Difference"] = score_differences

    return result_df


# Calculate RMSE for Both Versions
def calculate_rmse(df, col_predicted):
    valid_scores = df.dropna(subset=[col_predicted, "Actual Score"])
    if not valid_scores.empty:
        rmse = np.sqrt(
            np.mean((valid_scores[col_predicted] - valid_scores["Actual Score"]) ** 2)
        )
    else:
        rmse = None
    return rmse


# --- Function to Calculate RMSE and Regression Accuracy ---
def calculate_metrics(df, col_predicted):
    valid_scores = df.dropna(subset=[col_predicted, "Actual Score"])
    if not valid_scores.empty:
        rmse = np.sqrt(
            np.mean((valid_scores[col_predicted] - valid_scores["Actual Score"]) ** 2)
        )
        regression_accuracy = 1 - (
            np.sum(abs(valid_scores[col_predicted] - valid_scores["Actual Score"]))
            / np.sum(valid_scores["Actual Score"])
        )
    else:
        rmse, regression_accuracy = None, None
    return rmse, regression_accuracy


# --- Load Excel Data ---
excel_file = "sp_clock_drawing.xlsx"  # Update with your actual file path
df = pd.read_excel(excel_file)

# --- Set Folder Path for Images ---
image_folder = "NHATS_R13_ClockDrawings_JPG"


# Create a filtered DataFrame containing only rows that have images
df_filtered = df[df.apply(image_exists, axis=1)].copy()
df_filtered = df_filtered.head(50)  # Use only first 50 rows for testing


# --- Save Both Versions to Excel ---
df_with_info = process_scoring(df_filtered, use_patient_info=True)
df_no_info = process_scoring(df_filtered, use_patient_info=False)

# Merge Results for Comparison
df_merged = df_with_info.merge(
    df_no_info,
    on=["NHATS SAMPLED PERSON ID", "Actual Score"],
    suffixes=("_with_info", "_no_info"),
)


rmse_with_info, acc_with_info = calculate_metrics(
    df_merged, "Predicted Score (With Patient Info)"
)
rmse_no_info, acc_no_info = calculate_metrics(
    df_merged, "Predicted Score (No Patient Info)"
)


# --- Save to Excel ---
output_file = "nhats_clock_drawing_scores_comparison.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df_merged.to_excel(writer, sheet_name="Predictions", index=False)
    pd.DataFrame(
        {
            "Version": ["With Patient Info", "No Patient Info"],
            "RMSE": [rmse_with_info, rmse_no_info],
            "Regression Accuracy": [acc_with_info, acc_no_info],
        }
    ).to_excel(writer, sheet_name="Metrics", index=False)

print(f"Results saved to {output_file}")
