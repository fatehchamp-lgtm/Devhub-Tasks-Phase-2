import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Har file path ko dynamic rakhne ke liye base directory find karte hain
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "house_images")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# Define total samples
NUM_SAMPLES = 200
print(f"🔄 Generating synthetic dataset with {NUM_SAMPLES} samples...")

# 1. Generate Tabular Data
np.random.seed(42)
house_ids = [f"house_{i:03d}" for i in range(NUM_SAMPLES)]
rooms = np.random.randint(2, 6, size=NUM_SAMPLES)
bathrooms = np.random.randint(1, 4, size=NUM_SAMPLES)
area_sqft = np.random.randint(800, 4000, size=NUM_SAMPLES)
age_years = np.random.randint(1, 50, size=NUM_SAMPLES)

# Price calculation formula with a logical base + some noise
# Standard formula: (rooms * 25k) + (bathrooms * 15k) + (area * 120) - (age * 800) + random noise
base_price = (rooms * 25000) + (bathrooms * 15000) + (area_sqft * 120) - (age_years * 800)
noise = np.random.normal(10000, 5000, size=NUM_SAMPLES)
prices = base_price + noise

# Create DataFrame
df = pd.DataFrame({
    'house_id': house_ids,
    'rooms': rooms,
    'bathrooms': bathrooms,
    'area_sqft': area_sqft,
    'age_years': age_years,
    'price': prices
})

# Save tabular data to CSV
csv_path = os.path.join(DATA_DIR, "housing_data.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Tabular dataset saved successfully at: {csv_path}")

# 2. Generate House Images (Solid colors with schematic outlines for deep learning input)
print("🖼️ Generating corresponding house images...")
for i, house_id in enumerate(house_ids):
    # Dynamic colors based on house properties to make it somewhat realistic for CNN
    r = int(np.clip((rooms[i] / 6) * 255, 0, 255))
    g = int(np.clip((area_sqft[i] / 4000) * 255, 0, 255))
    b = int(np.clip(255 - (age_years[i] / 50) * 255, 0, 255))

    img = Image.new("RGB", (224, 224), color=(r, g, b))

    # Draw a small basic shape (roof/door) on the image so the CNN has simple geometry to learn
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 100, 170, 200], fill=(50, 50, 50))  # House body outline
    draw.polygon([(40, 100), (112, 40), (184, 100)], fill=(150, 0, 0))  # Roof outline

    # Save image
    img_path = os.path.join(IMAGES_DIR, f"{house_id}.jpg")
    img.save(img_path)

print(f"✅ {NUM_SAMPLES} Images generated and saved successfully in: {IMAGES_DIR}")
print("\n🚀 Step 2: Data Generation Completed Successfully!")