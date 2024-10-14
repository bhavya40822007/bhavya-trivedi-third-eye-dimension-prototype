from flask import Flask, request, render_template
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)

# Load pre-trained model
model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
model.to("cuda")

@app.route("/", methods=["GET", "POST"])
def generate_image():
    if request.method == "POST":
        text_prompt = request.form["prompt"]
        image = model(text_prompt).images[0]
        image.save("static/generated_image.png")
        return render_template("index.html", image_url="static/generated_image.png")
    return render_template("index.html", image_url=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
