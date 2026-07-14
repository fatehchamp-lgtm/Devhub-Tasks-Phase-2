# Task 3: Multimodal Machine Learning – Housing Price Prediction Using Images + Tabular Data

An advanced Deep Learning system designed to predict housing prices by simultaneously processing and fusing multi-modal data: **tabular features** (numerical/structural) and **house images** (visual patterns) using a hybrid **Convolutional Neural Network (CNN)** and **Dense Neural Network (MLP)** architecture.

---

## 🏗️ System Architecture & Feature Fusion

The core strength of this system is its **Feature Fusion Architecture**. Instead of relying on a single data source, the network extracts intelligence from two distinct branches before making a regression prediction:

1. **Tabular Branch (Dense MLP):** Processes structural properties (number of rooms, bathrooms, total area in sqft, and age of the house) through dense fully-connected layers to extract statistical patterns.
2. **Visual Branch (CNN):** Processes visual properties of the house using Conv2D layers and Max-Pooling blocks to capture spatial representations, geometry, and structural colors.
3. **Fusion Head:** Merges outputs from both branches using a `Concatenate` layer, passing the combined feature maps to a final regression network to output the estimated house price.

---

## 📁 Repository Structure

```text
Task 3 - Multimodal ML/
│
├── data/                       # Dataset directory
│   ├── house_images/           # Generated house visual schemas (.jpg format)
│   └── housing_data.csv        # Tabular features (rooms, area, prices, etc.)
│
├── saved_model/                # Model artifacts storage
│   ├── multimodal_house_model.h5  # Trained hybrid Keras model
│   ├── scaler_mean.npy         # StandardScaler mean values for deployment
│   └── scaler_var.npy          # StandardScaler variance values for deployment
│
├── generate_data.py            # Automatic synthetic dataset & image generator
├── train_multimodal.py         # Multi-input model design, training & evaluation pipeline
└── app.py                      # Interactive Streamlit Web UI Application