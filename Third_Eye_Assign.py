''' Project -  Bhavya Trivedi '''

from flask import Flask, request, render_template_string
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
model.to("cuda")

# Initialize the Flask app
app = Flask(__name__)

# HTML template embedded as a string
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text-to-Image Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            color: #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        h2 {
            color: #4CAF50;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        label {
            font-size: 18px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        .container {
            max-width: 500px;
            width: 100%;
            margin: auto;
        }
        .image-container {
            margin-top: 20px;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Text-to-Image Generator</h2>
        <form method="POST">
            <label for="prompt">Enter text prompt:</label><br><br>
            <input type="text" id="prompt" name="prompt" placeholder="Type a creative prompt..." required><br><br>
            <input type="submit" value="Generate Image">
        </form>
        
        {% if image_url %}
        <div class="image-container">
            <h3>Generated Image:</h3>
            <img src="{{ image_url }}" alt="Generated Image">
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Define the Flask route for the homepage
@app.route("/", methods=["GET", "POST"])
def generate_image():
    image_url = None
    if request.method == "POST":
        text_prompt = request.form["prompt"]
        # Generate the image from the text prompt
        image = model(text_prompt).images[0]
        # Save the image locally
        image_path = "static/generated_image.png"
        image.save(image_path)
        image_url = image_path

    # Render the template with the image (if any)
    return render_template_string(html_template, image_url=image_url)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
