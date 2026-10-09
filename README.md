# Image Based Breed Recognition for Cattle and Buffaloes of India

An AI-powered image classification system designed to recognize cattle and buffalo breeds and support livestock health assessment using deep learning and computer vision.

## Overview

This project uses deep learning models to analyze livestock images and provide predictions for breed recognition, health classification, and Body Condition Score (BCS) classification.

The application is developed using Python and Flask, with a web interface that allows users to upload images and view prediction results.

## Features

- **Breed Recognition:** Identifies the predicted breed from an uploaded livestock image.
- **Health Classification:** Classifies images into supported health categories.
- **Body Condition Score (BCS):** Predicts a supported body condition category.
- **Image Upload Interface:** Provides a web interface for submitting images.
- **Prediction API:** Uses Flask routes to process requests and return prediction results.
- **Grad-CAM Visualization:** Includes a heatmap image for model interpretability experiments.

## Technologies Used

- **Programming Language:** Python
- **Backend:** Flask
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** Image classification and Grad-CAM visualization
- **Frontend:** HTML, CSS, JavaScript (where implemented)
- **Model Architecture:** MobileNetV2-based deep learning models
- **Data Format:** JSON for class labels
- **Development Tools:** Jupyter Notebook, VS Code, Git, GitHub

## Project Structure

```text
Cattle-Buffalo-Breed-Recognition-AI/
├── app.py
├── CattleAI_FINAL.ipynb
├── requirements.txt
├── README.md
├── breed_classes.json
├── health_classes.json
├── bcs_classes.json
├── models/
│   ├── breed_model.keras
│   ├── health_model.keras
│   └── bcs_model.keras
├── templates/
│   └── index.html
└── static/
    ├── heatmaps/
    └── uploads/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Muskan-Gupta-tech/Cattle-Buffalo-Breed-Recognition-AI.git
cd Cattle-Buffalo-Breed-Recognition-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

## Model Categories

The application includes class-label files for the following prediction tasks:

- **Breed classification:** Cattle and buffalo breed categories defined in `breed_classes.json`.
- **Health classification:** Categories defined in `health_classes.json`.
- **BCS classification:** Categories defined in `bcs_classes.json`.

Predictions depend on the trained models and the image preprocessing implemented in the application.

## Future Improvements

- Improve model accuracy with more diverse and balanced datasets.
- Expand breed coverage across Indian cattle and buffalo populations.
- Improve prediction confidence reporting and model evaluation.
- Enhance the user interface and mobile responsiveness.
- Deploy the application for broader accessibility.

## Disclaimer

This project is intended for educational and research purposes. Model predictions should not be treated as a substitute for professional veterinary assessment.

## Author

**Muskan Gupta**

GitHub: [Muskan-Gupta-tech](https://github.com/Muskan-Gupta-tech)
