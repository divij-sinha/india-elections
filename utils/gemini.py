import os

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def get_text_gemini(img):
    model = "gemini-2.0-flash-lite-001"
    contents = [
        """
        Return the text from this image. Only return the text translated into English. 
        DO NOT return anything else. Do not return any other explanation or commentary. 
        DO NOT change the text. Just translate the text word-for-word. 
        The MOST IMPORTANT THING IS THE TITLES OF THE SECTIONS. USE ONLY THE FOLLOWING APPROVED TITLES --
        "Name", "Father's Name", "Mother's Name", "Husband's Name", "Wife's Name", "Address", "Age", "Gender"
        The Gender should IDEALLY be one of ["Male", "Female", "Other"]
        The Age should be a number.
        The Address should be a single line.
        Combine age and Gender into a single line.
        Sample output:
        "Name: John Doe
        Father's Name: John Doe Sr.
        Address: 123 Main St, Springfield
        Age: 30  Gender: Male"
        DO NOT TRANSLATE THE VALUES OF THE NAMES AND ADDRESSES, BUT TRANSLITERATE THEM TO ENGLISH.

        !!THE ENTIRE OUTPUT SHOULD BE IN THE LATIN SCRIPT!!

        Check that the text is in English, and that it perfectly corresponds to the image.
        """,
        img,
    ]
    err = None
    for _ in range(3):
        try:
            result = talk_gemini(contents=contents, model=model)
            print(result.usage_metadata)
            return result.text
        except Exception as e:
            print("Retrying! Error in get_text_gemini", e)
            err = e
    raise err


def talk_gemini(contents: list, model: str = "gemini-2.0-flash"):
    # total_tokens = client.models.count_tokens(
    #     model=model,
    #     contents=contents,
    # )
    # print(f"Total tokens: {total_tokens}")
    resp = client.models.generate_content(
        model=model,
        contents=contents,
    )
    return resp
