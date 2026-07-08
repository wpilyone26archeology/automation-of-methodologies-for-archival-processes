from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

client = OpenAI(
    api_key="lm-studio",
    base_url="http://localhost:1234/v1",
)

# First just check the model list
print("Checking available models...")
try:
    models = client.models.list()
    print("Server is reachable. Models available:")
    for m in models.data:
        print(f"  - {m.id}")
except Exception as e:
    print(f"Cannot reach server: {e}")
    exit()

# Then try a minimal VL call with a tiny test image
print("\nTesting vision call...")
img = Image.new("RGB", (64, 64), color=(128, 128, 128))
buf = BytesIO()
img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

try:
    response = client.chat.completions.create(
        model=models.data[0].id,   # uses whatever model is loaded
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "What color is this image? Reply in one word."}
            ]
        }],
        max_tokens=20,
    )
    print(f"Vision call succeeded: {response.choices[0].message.content}")
except Exception as e:
    print(f"Vision call failed: {e}")
