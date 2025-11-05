# circuitz.project – Emotion Detection Model & GUI

This repository contains the AI part of our ENEC200 Electrical Circuits project **“Integration of Artificial Intelligence with Circuit Design for Emotion Detection”**.

It includes:
- The **trained emotion recognition model**
- The **training / experimentation notebook**
- The **GUI script** used to test the model in a simple interface

---

## Repository contents

- `trainmodel.ipynb`  
  Jupyter notebook used to train and evaluate the emotion detection model (data loading, training loop, evaluation, saving the best model, etc.).

- `emotion_model_best.pt`  
  Saved version of the best-performing trained model.

- `artifacts/`  
  Folder for any extra files used by the model (preprocessing objects, label mapping, logs, etc.).

- `thegui.py`  
  Python GUI that loads `emotion_model_best.pt` and allows the user to test the model on new inputs in a simple interface.

---

## How to use

1. **Clone the repository**
   ```bash
   git clone https://github.com/manarghr/circuitz.project.git
   cd circuitz.project
