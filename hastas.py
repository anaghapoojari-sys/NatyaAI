import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib 

# 1. Load your raw dataset
df = pd.read_csv("mudra_dataset.csv")

def preprocess_and_extract_features(row):
    # Reshape row to a (21, 3) matrix of landmarks
    coords = np.array([
        [row[f'x{i}'], row[f'y{i}'], row[f'z{i}']]
        for i in range(21)
    ])
    
    # --- STEP 1: SPATIAL NORMALIZATION ---
    wrist = coords[0]
    translated_coords = coords - wrist  # Move origin to wrist
    
    # Scale normalization using wrist-to-middle-finger-mcp distance
    scale_factor = np.linalg.norm(coords[9] - coords[0])
    if scale_factor == 0:
        scale_factor = 1e-6
    normalized_coords = translated_coords / scale_factor
    
    # --- STEP 2: GEOMETRIC FEATURE EXTRACTION ---
    features = normalized_coords.flatten().tolist()

    # Tip-to-tip distances (Thumb-4, Index-8, Middle-12, Ring-16, Pinky-20)
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dist = np.linalg.norm(normalized_coords[tips[i]] - normalized_coords[tips[j]])
            features.append(dist)
            
    return features

# Process all rows to create feature matrix X
X_features = [preprocess_and_extract_features(row) for _, row in df.iterrows()]
y = df['label'].values  # Assuming your target column is called 'label'

# 3. Data Split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# 4. Train the Engine
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))
joblib.dump(clf, "rf_model.pkl")
print("Model saved successfully!")