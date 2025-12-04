# from google import genai
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# client = genai.Client(api_key=os.getenv("GENERATIVE_LANGUAGE_API_KEY"))

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)

import time
from google import genai
from google.genai import types

# Lấy API key từ biến môi trường, báo lỗi nếu thiếu
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GENERATIVE_LANGUAGE_API_KEY")
if not api_key:
    raise ValueError("Missing GENERATIVE_LANGUAGE_API_KEY environment variable! Vui lòng export/set biến này với API key của bạn.")
client = genai.Client(api_key=api_key)

operation = client.models.generate_videos(
    model="veo-3.0-generate-001",
    prompt="A cinematic shot of a majestic lion in the savannah.",
    config=types.GenerateVideosConfig(
        negative_prompt="cartoon, drawing, low quality",
        aspect_ratio="16:9",
        resolution="1080p",
    ),
)

while not operation.done:
    print("Waiting...")
    time.sleep(10)
    operation = client.operations.get(operation)

video = operation.response.generated_videos[0]
client.files.download(file=video.video)
video.video.save("lion.mp4")
print("Video saved!")
