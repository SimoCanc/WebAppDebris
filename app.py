# app.py
import io
import numpy as np
from utils import *
from loss import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fastai.vision.all import * #load_learner, PILImage
from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)

MODEL_PATH = 'Pippo.pkl'
IMG_SIZE = 512

# Caricamento modello
learn = load_learner(MODEL_PATH, cpu=True)  
learn.dls.to(device='cpu') # device='cuda'
learn.model.to(device='cpu') # device='cuda'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "File non selezionato!!"}), 400
    file = request.files['file']
    try:
        # Ridimensionamento e trasformazione a 8-bit dell'immagine
	img_resized_8bit = from_16_to_8(file, IMG_SIZE)
    except Exception as e:
        return jsonify({"errore": str(e)}), 400

    # Predizione maschera
    pred, _, _ = learn.predict(img_resized_8bit)
    pred_mask = pred.argmax(dim=0).numpy()
    # Estrapolazione coordinate estremi maschera
    y_coords, x_coords  = find_highest_lowest_xy(pred_mask)
    
    # Graficazione coordinate
    buf = plot_coords(img_8bit)
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
