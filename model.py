import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Create models/ directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save models and preprocessing objects
joblib.dump(logreg, "models/logreg_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
