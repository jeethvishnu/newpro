from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import torch
from torchvision import transforms
from PIL import Image
from src.model import PretrainedCNNModel
# from src.data_loading import race_mapping

app = Flask(__name__)

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

race_mapping = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others"
}

# Function to predict the race of an input image
def predict_race(image_path, model_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    model = PretrainedCNNModel(num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    race = race_mapping[predicted.item()]
    return race

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            static_dir = os.path.join(app.root_path, 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
            file_path = os.path.join(static_dir, file.filename)
            file.save(file_path)
            model_path = os.path.join(app.root_path, 'models', 'utkface_model.pth')
            race = predict_race(file_path, model_path)
            return render_template('index.html', filename=file.filename, race=race)
    return render_template('index.html', filename=None, race=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
