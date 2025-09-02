
#  Plant Species Identification using ResNet + FPN

This project is a **Plant Species Identification System** that uses a **ResNet-based Feature Pyramid Network (FPN)** for accurate multi-scale image classification.  
Users can upload an image of any plant part (leaf, flower, fruit, bark, etc.), and the model predicts the species with high accuracy.

---

##  Features
- Uses **ResNet with Feature Pyramid Network (FPN)** for multi-scale feature extraction  
- **Data Augmentation** for better generalization  
- **80/20 Train-Validation Split**  
- **Gradio-based GUI** for interactive predictions  
- Returns **"Unknown Plant"** if confidence is below 75%  
- GPU acceleration support

---

##  Project Structure
```
plant_species_identifier/
│── fpn_train.py                # Training script
│── fpn_gui1.py                # Gradio-based GUI for predictions
│── fpn_resnet.py              # ResNet + FPN model definition
│── data_loader.py             # Custom dataset loader
│── plant_species/             # Dataset folder
│    ├── my_plant_dataset/     # Organized plant species images
│    │    ├── Species_1/
│    │    ├── Species_2/
│    │    ├── Species_3/
│── classes1.txt               # Contains class labels
│── fpn_classifier_model_final.pth   # Trained model weights (generated after training)
│── README.md                  # Project documentation
```

---

##  Dataset Instructions
- Place your dataset in:
  ```
  plant_species/my_plant_dataset/
  ```
- The dataset structure must follow:
  ```
  my_plant_dataset/
  ├── Species_1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  ├── Species_2/
  │   ├── img1.jpg
  │   ├── img2.jpg
  ├── Species_3/
      ├── img1.jpg
      ├── img2.jpg
  ```
- **Each species must have its own folder**.

---

##  Installation

### 1 Clone the Repository
```bash
git clone https://github.com/your-username/plant-species-identifier.git
cd plant-species-identifier
```

### Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# OR
source venv/bin/activate      # Linux / Mac
```

### Install Dependencies
```bash
pip install torch torchvision gradio pillow matplotlib
```

---

##  Model Training

Run the training script:
```bash
python fpn_train.py
```

### What it does:
- Loads the dataset from `plant_species/my_plant_dataset`
- Performs data augmentation
- Splits into **80% training** and **20% validation**
- Trains the **ResNet + FPN** model
- Saves the trained model as:
  ```
  fpn_classifier_model_final.pth
  ```
- Also generates `classes1.txt` automatically (if not already provided)

---

## Running the GUI App

After training, launch the Gradio-based web app:
```bash
python fpn_gui1.py
```

### Steps:
1. Upload a **plant image**
2. The model predicts the species
3. If confidence < **75%**, you'll get **"Unknown Plant"**
4. If confidence ≥ **75%**, you'll get the **species name** 

---

##  Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Torchvision ≥ 0.15
- Gradio ≥ 3.0
- Pillow
- Matplotlib

To install everything at once:
```bash
pip install -r requirements.txt
```

---

## Future Improvements
🔹 Improve model accuracy using **Vision Transformers (ViT)**  
🔹 Add **Medicinal / Poisonous Plant Detection**  
🔹 Deploy as a **Web + Mobile App**  
🔹 Integrate **Explainable AI (XAI)** for visualizing feature maps  
