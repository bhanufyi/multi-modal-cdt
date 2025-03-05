from openai import OpenAI
import base64

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def encode_image(image_path):
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_image(image_path):
    """Sends an image to OpenAI GPT-4o and gets a description."""
    base64_image = encode_image(image_path)

    # Define the JSON template for the response
    json_template = '{"score": 0.0}'

    # Define the system prompt
    system_prompt = (
        "You are a neuropsychology expert analyzing digital clock drawings. "
        "Evaluate the provided clock drawing test and assign a score from 0 to 5. "
        "The score reflects cognitive performance, considering standard neuropsychology criteria. "
        "Respond strictly in JSON format using the template: " + json_template
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ],
    )

    return response.choices[0].message.content


# Example usage
image_path = "NHATS_R13_ClockDrawings_JPG/10000008.jpg"  # Change this to your image path
description = describe_image(image_path)
print("Image Description:", description)
