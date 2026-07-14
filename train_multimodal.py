import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

# 1. Base Paths Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "housing_data.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "house_images")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_model", "multimodal_house_model.h5")

# Ensure saved_model directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

print("📊 Loading and preparing datasets...")
df = pd.read_csv(DATA_CSV)


# 2. Load and Preprocess Images
def load_and_preprocess_images(dataframe, images_dir):
    image_list = []
    for house_id in dataframe['house_id']:
        img_path = os.path.join(images_dir, f"{house_id}.jpg")
        # Read image, resize to 224x224, convert to numpy array and normalize
        img = Image.open(img_path).resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        image_list.append(img_array)
    return np.array(image_list)


X_images = load_and_preprocess_images(df, IMAGES_DIR)

# 3. Prepare Tabular Features & Target
X_tabular = df[['rooms', 'bathrooms', 'area_sqft', 'age_years']].values
y_prices = df['price'].values

# Split into Train and Test sets (80% Train, 20% Test)
X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
    X_tabular, X_images, y_prices, test_size=0.2, random_state=42
)

# Normalize Tabular Features
scaler = StandardScaler()
X_tab_train_scaled = scaler.fit_transform(X_tab_train)
X_tab_test_scaled = scaler.transform(X_tab_test)

# Save the scaler values for later use in Streamlit app
np.save(os.path.join(BASE_DIR, "saved_model", "scaler_mean.npy"), scaler.mean_)
np.save(os.path.join(BASE_DIR, "saved_model", "scaler_var.npy"), scaler.var_)

print(f"✅ Data split complete! Training samples: {len(y_train)}, Testing samples: {len(y_test)}")


# 4. Define Multimodal Neural Network Model
def build_multimodal_model():
    # --- Tabular Data Branch ---
    tabular_input = layers.Input(shape=(4,), name="tabular_input")
    tab_dense1 = layers.Dense(32, activation="relu")(tabular_input)
    tab_dense2 = layers.Dense(16, activation="relu")(tab_dense1)

    # --- Image Data Branch (CNN) ---
    image_input = layers.Input(shape=(224, 224, 3), name="image_input")

    cnn_conv1 = layers.Conv2D(16, (3, 3), activation="relu")(image_input)
    cnn_pool1 = layers.MaxPooling2D((2, 2))(cnn_conv1)

    cnn_conv2 = layers.Conv2D(32, (3, 3), activation="relu")(cnn_pool1)
    cnn_pool2 = layers.MaxPooling2D((2, 2))(cnn_conv2)

    cnn_flatten = layers.Flatten()(cnn_pool2)
    cnn_dense = layers.Dense(16, activation="relu")(cnn_flatten)

    # --- Feature Fusion (Concatenate) ---
    fused_features = layers.concatenate([tab_dense2, cnn_dense])

    # --- Regression Head ---
    fc1 = layers.Dense(16, activation="relu")(fused_features)
    output_price = layers.Dense(1, activation="linear", name="price_output")(fc1)

    model = models.Model(inputs=[tabular_input, image_input], outputs=output_price)
    return model


print("🏗️ Building the fused neural network architecture...")
model = build_multimodal_model()

# Compile model using MAE and Root Mean Squared Error (RMSE)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss="mse",
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

# 5. Train the Model
print("🚀 Training starting (15 epochs for quick convergence)...")
history = model.fit(
    x=[X_tab_train_scaled, X_img_train],
    y=y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=16,
    verbose=1
)

# 6. Model Evaluation (MAE & RMSE)
print("\n📊 Evaluating model on Test Set...")
eval_results = model.evaluate(x=[X_tab_test_scaled, X_img_test], y=y_test, verbose=0)
loss, test_mae, test_rmse = eval_results

print("\n" + "=" * 40)
print(f"🏆 TEST EVALUATION METRICS:")
print(f"➡️ Mean Absolute Error (MAE): ${test_mae:.2f}")
print(f"➡️ Root Mean Squared Error (RMSE): ${test_rmse:.2f}")
print("=" * 40 + "\n")

# 7. Save the Model
print(f"💾 Saving trained multimodal model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("✅ Step 3 Complete!")