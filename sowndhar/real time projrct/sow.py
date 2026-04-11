"""
ai_crop_yield.py
Single-file demo of AI-powered crop yield prediction + simple optimization.

Requirements:
    pip install numpy pandas scikit-learn tensorflow

Run:
    python ai_crop_yield.py

What it does:
1. Generates synthetic farm dataset (soil, weather, NDVI, irrigation, fertilizer).
2. Trains a Keras tabular model to predict yield (tonnes/hectare).
3. Evaluates model performance.
4. For a given farm sample, runs a grid search over irrigation & fertilizer
   levels to recommend settings that maximize predicted yield under constraints.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# ---------------------------
# 1) Synthetic Data Generator
# ---------------------------
def generate_synthetic_dataset(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    rows = []
    for i in range(n_samples):
        # Static soil features
        soil_ph = np.random.normal(6.5, 0.6)                # 4.5 - 8
        organic_matter = np.clip(np.random.normal(3.5, 1.2), 0.1, 10)  # %
        soil_texture = np.random.choice([0,1,2])            # 0:sandy,1:loam,2:clay (one-hot later)
        elevation = np.random.normal(150, 80)               # meters
        slope = abs(np.random.normal(3, 2))                 # degrees

        # Weather summary (season-level aggregates)
        avg_temp = np.random.normal(24, 4)                  # C
        precipitation = np.clip(np.random.normal(450, 180), 0, None)  # mm / season
        sunshine_hours = np.clip(np.random.normal(1800, 300), 600, None)  # yearly / aggregated

        # Remote sensing (NDVI)
        ndvi = np.clip(0.2 + 0.005*organic_matter + 0.001*sunshine_hours + np.random.normal(0,0.05), 0, 1)

        # Management inputs (we'll vary these and let model learn)
        irrigation = np.clip(np.random.normal(200, 120), 0, 800)  # mm applied during season
        fertilizer_n = np.clip(np.random.normal(80, 40), 0, 300)  # kg/ha

        # Interaction effects -> generate yield (tonne/ha)
        # baseline yield influenced by soil & ndvi & weather
        yield_base = (0.8*ndvi + 0.03*organic_matter + 0.01*(7.0 - abs(soil_ph-6.5))
                      + 0.004*(precipitation) + 0.02*(avg_temp-18))
        # management marginal returns
        irr_effect = 0.0009 * irrigation * (1 - np.exp(-0.003*precipitation))  # benefit if precipitation is low
        fert_effect = 0.008 * fertilizer_n * np.exp(-0.002*fertilizer_n)      # diminishing returns
        interaction = 0.002 * ndvi * fertilizer_n

        # penalties for extremes (over-irrigation or too much fertilizer)
        penalty = -0.000002*(irrigation-400)**2 - 0.00001*(fertilizer_n-160)**2

        yield_tph = np.clip( yield_base*5 + irr_effect*2 + fert_effect + interaction + penalty + np.random.normal(0,0.4), 0.1, 20.0)

        rows.append({
            'soil_ph': soil_ph,
            'organic_matter': organic_matter,
            'soil_texture': soil_texture,
            'elevation': elevation,
            'slope': slope,
            'avg_temp': avg_temp,
            'precipitation': precipitation,
            'sunshine_hours': sunshine_hours,
            'ndvi': ndvi,
            'irrigation': irrigation,
            'fertilizer_n': fertilizer_n,
            'yield_tph': yield_tph
        })

    df = pd.DataFrame(rows)
    # One-hot soil_texture
    df = pd.concat([df, pd.get_dummies(df['soil_texture'], prefix='texture')], axis=1).drop(columns=['soil_texture'])
    return df

# ---------------------------
# 2) Preprocess
# ---------------------------
def prepare_data(df, feature_cols, target_col='yield_tph', test_size=0.15, val_size=0.15, random_seed=42):
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1,1)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=relative_val, random_state=random_seed)
    scalerX = StandardScaler().fit(X_train)
    scalery = StandardScaler().fit(y_train)
    X_train_s = scalerX.transform(X_train)
    X_val_s = scalerX.transform(X_val)
    X_test_s = scalerX.transform(X_test)
    y_train_s = scalery.transform(y_train)
    y_val_s = scalery.transform(y_val)
    y_test_s = scalery.transform(y_test)
    return (X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scalerX, scalery, X_test, y_test)

# ---------------------------
# 3) Build & Train Model
# ---------------------------
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss='mse', metrics=['mse'])
    return model

# ---------------------------
# 4) Simple optimizer (grid search)
# ---------------------------
def suggest_management(model, scalerX, scalery, base_sample, irr_range=(0,600), fert_range=(0,240), steps=25, constraints=None):
    """
    base_sample: dict with current farm features (must include all features except 'irrigation' and 'fertilizer_n' if you want to vary them)
    constraints: dict like {'max_irrigation': 400, 'max_fertilizer': 120} (optional)
    returns best setting and DataFrame of top suggestions
    """
    # create grid
    irr_vals = np.linspace(irr_range[0], irr_range[1], steps)
    fert_vals = np.linspace(fert_range[0], fert_range[1], steps)
    candidates = []
    feature_order = base_sample['feature_order']
    for irr in irr_vals:
        for fert in fert_vals:
            if constraints:
                if 'max_irrigation' in constraints and irr > constraints['max_irrigation']:
                    continue
                if 'max_fertilizer' in constraints and fert > constraints['max_fertilizer']:
                    continue
            sample = base_sample['values'].copy()
            # positions for irrigation and fertilizer known
            idx_irr = feature_order.index('irrigation')
            idx_fert = feature_order.index('fertilizer_n')
            sample[idx_irr] = irr
            sample[idx_fert] = fert
            # scale and predict
            sample_s = scalerX.transform([sample])
            pred_s = model.predict(sample_s, verbose=0)
            pred = scalery.inverse_transform(pred_s.reshape(1, -1))[0,0]
            candidates.append({'irrigation': float(irr), 'fertilizer_n': float(fert), 'predicted_yield_tph': float(pred)})
    cand_df = pd.DataFrame(candidates)
    top = cand_df.sort_values('predicted_yield_tph', ascending=False).reset_index(drop=True)
    return top

# ---------------------------
# 5) Main: train & demo
# ---------------------------
def main():
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=6000)
    # feature columns (order matters for optimizer)
    feature_cols = ['soil_ph','organic_matter','elevation','slope','avg_temp','precipitation','sunshine_hours','ndvi',
                    'irrigation','fertilizer_n','texture_0','texture_1','texture_2']
    target_col = 'yield_tph'
    # ensure textures exist (they should)
    for t in ['texture_0','texture_1','texture_2']:
        if t not in df.columns:
            df[t] = 0

    print("Preparing data (scaling, splitting)...")
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scalerX, scalery, X_test_raw, y_test_raw = prepare_data(df, feature_cols, target_col)

    print("Building model...")
    model = build_model(input_dim=X_train_s.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)
    ]

    print("Training model (this may take a bit)...")
    history = model.fit(X_train_s, y_train_s, validation_data=(X_val_s, y_val_s), epochs=120, batch_size=64, callbacks=callbacks, verbose=1)

    # Evaluate
    print("Evaluating on test set...")
    preds_s = model.predict(X_test_s).reshape(-1,1)
    preds = scalery.inverse_transform(preds_s)
    y_true = y_test_raw.reshape(-1,1)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"Test RMSE: {rmse:.3f} t/ha, R²: {r2:.3f}")

    # Example: pick a random farm from test set and optimize management
    idx = np.random.randint(0, X_test_raw.shape[0])
    base_row = X_test_raw[idx].copy()
    # prepare base_sample dict
    base_sample = {'values': base_row.tolist(), 'feature_order': feature_cols}
    # We'll allow irrigation and fertilizer to vary; keep other features fixed
    print("\nExample farm (base features):")
    sample_df = pd.DataFrame([base_row], columns=feature_cols)
    print(sample_df.T)

    # Constraint example: max irrigation 400 mm, max fertilizer 160 kg/ha (user-definable)
    constraints = {'max_irrigation': 400, 'max_fertilizer': 160}

    print("\nSearching for best irrigation & fertilizer (grid search)...")
    top_candidates = suggest_management(model, scalerX, scalery, base_sample, irr_range=(0,600), fert_range=(0,240), steps=30, constraints=constraints)

    print("\nTop 5 recommended settings (constrained):")
    print(top_candidates.head(5).to_string(index=False))

    # Show improvement over base
    # compute base prediction
    base_s = scalerX.transform([base_row])
    base_pred_s = model.predict(base_s, verbose=0)
    base_pred = scalery.inverse_transform(base_pred_s.reshape(1, -1))[0,0]
    best = top_candidates.iloc[0]
    improvement = best['predicted_yield_tph'] - base_pred
    print(f"\nBase predicted yield: {base_pred:.3f} t/ha")
    print(f"Best predicted yield: {best['predicted_yield_tph']:.3f} t/ha (irrigation={best['irrigation']:.1f} mm, fertilizer={best['fertilizer_n']:.1f} kg/ha)")
    print(f"Estimated improvement: {improvement:.3f} t/ha")

    # Save model for later use
    model_path = "crop_yield_model.keras"
    print(f"\nSaving model to '{model_path}' ...")
    model.save(model_path)
    print("Done.")

if __name__ == "__main__":
    main()

