import os
import json
import pandas as pd
import openai
import base64


# Initialize OpenAI Client (Latest API Standard)
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    """Encodes an image file to Base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# --- Test Function Using Excel Details ---
def test_gpt4o_with_excel_details(excel_file, sample_id, image_path):
    # Load the Excel file
    df = pd.read_excel(excel_file)

    # Filter for the record with the given sample_id
    record = df[df["NHATS SAMPLED PERSON ID"] == sample_id]
    if record.empty:
        print(f"No record found for sample ID: {sample_id}")
        return

    # Get the first matching record
    row = record.iloc[0]

    # Retrieve details from Excel
    age_group = row.get("R13 WB3 AGE YOU FEEL MOST OF TIME", "Unknown")
    race_code = row.get("R13 D RACE AND HISPANIC ETHNICITY WHEN ADDED", "Unknown")
    race = race_mapping.get(race_code, "Unknown")
    occupation_code = row.get("R13 D LONGEST OCCUPATION CATEGORY", -9)
    education = occupation_mapping.get(occupation_code, "Unknown")

    # Define the JSON template for the response
    json_template = '{"score": 0.0}'

    # Define the system prompt
    system_prompt = (
        "You are a neuropsychology expert analyzing digital clock drawings. "
        "Evaluate the provided clock drawing test and assign a score from 0 to 5. "
        "The score reflects cognitive performance, considering standard neuropsychology criteria. "
        "Respond strictly in JSON format using the template: " + json_template
    )

    # Build the user prompt using details from Excel
    user_prompt = (
        f"Patient Information:\n"
        f"- Age Group: {age_group}\n"
        f"- Race: {race}\n"
        f"- Education: {education}\n\n"
        "This is a clock from a neuropsychology assessment known as the Digital Clock Drawing Test. "
        "Please analyze the provided image and assign a score (0-5) based on cognitive assessment measures.\n"
        f"Respond with JSON in this format: {json_template}."
    )

    # Encode the image
    base64_image = encode_image(image_path)

    try:
        # Send request to GPT-4o using latest OpenAI API standard
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

        # Print the raw response
        response_content = response.choices[0].message.content
        print("Raw Response Content:", response_content)

        # Attempt to parse the JSON output
        try:
            result = json.loads(response_content)
            print("Parsed Result:", result)
        except Exception as parse_error:
            print("Error parsing JSON:", parse_error)

    except Exception as e:
        print("Error during GPT-4o API call:", e)


# --- Test Parameters ---
excel_file = "sp_clock_drawing.xlsx"  # Update with your Excel file path
sample_id = 10000008  # The NHATS ID for the test
test_image_path = "NHATS_R13_ClockDrawings_JPG/10000008.jpg"  # Update path if needed

if os.path.exists(test_image_path):
    test_gpt4o_with_excel_details(excel_file, sample_id, test_image_path)
else:
    print(f"Test image not found at {test_image_path}")
