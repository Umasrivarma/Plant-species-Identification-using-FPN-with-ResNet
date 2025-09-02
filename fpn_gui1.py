import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from fpn_resnet import ResNetFPNClassifier

# --- Load models and class labels ---
model_paths = [
    ("fpn_classifier_model_final.pth", "classes1.txt"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = []
class_lists = []

for model_path, class_file in model_paths:
    with open(class_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    model = ResNetFPNClassifier(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    models.append(model)
    class_lists.append(classes)

# --- Transform for inference ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Original inference function ---
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    top_confidence = 0.0
    top_label = None

    for model, classes in zip(models, class_lists):
        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            confidence = conf.item()
            predicted_class = classes[pred.item()]
            if confidence > top_confidence:
                top_confidence = confidence
                top_label = predicted_class

    threshold = 0.75
    if top_confidence < threshold:
        return f"🌱 Prediction: Unknown Plant (Confidence: {top_confidence:.2f})"
    else:
        return f"🌿 Prediction: {top_label} (Confidence: {top_confidence:.2f})"

# --- Styled prediction output with black background and white font ---
def predict_with_style(image):
    raw_pred = predict(image)
    styled_pred = f"""
    <div style="
        font-size: 24px; 
        color: white; 
        background-color: black;
        border: 2px solid black; 
        padding: 10px; 
        border-radius: 5px; 
        max-width: 400px;
        white-space: pre-wrap;
    ">
        {raw_pred}
    </div>
    """
    return styled_pred

# --- Gradio UI ---
with gr.Blocks(css="""
    html, body, .gradio-container {
        background-color:black !important;
        color: black !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    #left-column {
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    #left-column button {
        width: 100% !important;
        font-size: 16px;
        margin-top: 10px;
    }
    #right-column {
        max-width: 600px;
        margin-left: 20px;
    }
""") as demo:

    gr.Markdown("## 🌿 Plant Species Identifier")
    gr.Markdown("Upload a plant image to identify its species. Returns 'Unknown Plant' for unfamiliar species.")

    with gr.Row():
        with gr.Column(elem_id="left-column"):
            image_input = gr.Image(type="pil", label="Upload Plant Image")
            submit_btn = gr.Button("Submit")

        with gr.Column(elem_id="right-column"):
            # Initial empty prediction box with black background and white text
            output_text = gr.Markdown(
                value="""
                <div style="
                    font-size: 24px; 
                    color: black; 
                    background-color: black;
                    border: 2px solid black; 
                    padding: 10px; 
                    border-radius: 5px; 
                    max-width: 400px;
                    min-height: 80px;
                    white-space: pre-wrap;
                ">
                </div>
                """,
                label="Prediction"
            )

    submit_btn.click(fn=predict_with_style, inputs=image_input, outputs=output_text)

demo.launch()
