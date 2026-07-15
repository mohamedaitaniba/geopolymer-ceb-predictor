import os
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_geopolymer_models():
    csv_file = r"C:\Users\bloch\Class\CEB_model_ready.csv"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Could not find '{csv_file}'. Please place it in the same directory as this script."
        )

    # 1. Load your clean CSV file
    df = pd.read_csv(r"C:\Users\bloch\Class\CEB_model_ready.csv")

    # Features selected by user
    features = [
        "Clay_pct",
        "Silt_pct",
        "Sand_pct",
        "Silica_Modulus_SiO2_Na2O",
        "Precursor_SiO2_Al2O3_ratio",
    ]

    # Target variables
    target_ucs = "UCS_best_available_MPa"
    target_wa = "Water_Absorption_24h_pct"

    # Enforce numeric types on our inputs
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- PIPELINE A: UCS PREDICTOR ---
    # Filter rows with valid UCS target
    df_ucs = df.dropna(subset=[target_ucs]).copy()
    X_ucs = df_ucs[features]
    y_ucs = df_ucs[target_ucs]

    X_train_ucs, X_test_ucs, y_train_ucs, y_test_ucs = train_test_split(
        X_ucs, y_ucs, test_size=0.15, random_state=42
    )

    ucs_model = Pipeline(
        [
            (
                "imputer",
                KNNImputer(n_neighbors=5),
            ),  # Fills missing values with mathematically closest match
            (
                "regressor",
                RandomForestRegressor(n_estimators=150, random_state=42),
            ),
        ]
    )
    ucs_model.fit(X_train_ucs, y_train_ucs)
    ucs_score = ucs_model.score(X_test_ucs, y_test_ucs)

    # --- PIPELINE B: WATER ABSORPTION PREDICTOR ---
    # Filter rows with valid Water Absorption target
    df_wa = df.dropna(subset=[target_wa]).copy()
    X_wa = df_wa[features]
    y_wa = df_wa[target_wa]

    X_train_wa, X_test_wa, y_train_wa, y_test_wa = train_test_split(
        X_wa, y_wa, test_size=0.15, random_state=42
    )

    wa_model = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=5)),
            (
                "regressor",
                RandomForestRegressor(n_estimators=150, random_state=42),
            ),
        ]
    )
    wa_model.fit(X_train_wa, y_train_wa)
    wa_score = wa_model.score(X_test_wa, y_test_wa)

    print("--- Model Performance ---")
    print(
        f"UCS Model R² validation score: {ucs_score:.2f} (Trained on {len(X_ucs)} mixes)"
    )
    print(
        f"Water Absorption R² validation score: {wa_score:.2f} (Trained on {len(X_wa)} mixes)"
    )

    return ucs_model, wa_model


# Train models on launch
ucs_pipeline, wa_pipeline = train_geopolymer_models()


# 2. Interface Prediction Function
def predict_ceb_properties(clay, silt, sand, silica_mod, si_al_ratio):
    # Prepare single-row input dataframe matching column names used in training
    input_df = pd.DataFrame(
        [[clay, silt, sand, silica_mod, si_al_ratio]],
        columns=[
            "Clay_pct",
            "Silt_pct",
            "Sand_pct",
            "Silica_Modulus_SiO2_Na2O",
            "Precursor_SiO2_Al2O3_ratio",
        ],
    )

    # Make independent predictions
    predicted_ucs = ucs_pipeline.predict(input_df)[0]
    predicted_wa = wa_pipeline.predict(input_df)[0]

    # Normalize prediction display
    return f"🚀 {predicted_ucs:.2f} MPa", f"💧 {predicted_wa:.2f} %"


# 3. Gradio Dashboard Design
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧱 Geopolymer-Stabilized Soil CEB Predictor")
    gr.Markdown(
        "Use this interface to adjust soil compositions and geopolymer chemical ratios to dynamically predict strength and absorption characteristics."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Structural Parameters")
            clay = gr.Slider(
                minimum=0,
                maximum=100,
                value=33.0,
                step=0.1,
                label="Clay Content (%)",
            )
            silt = gr.Slider(
                minimum=0,
                maximum=100,
                value=18.0,
                step=0.1,
                label="Silt Content (%)",
            )
            sand = gr.Slider(
                minimum=0,
                maximum=100,
                value=20.1,
                step=0.1,
                label="Sand Content (%)",
            )

            gr.Markdown("### Activator & Precursor Chemistry")
            silica_mod = gr.Number(
                value=1.5,
                label="Silica Modulus (SiO₂/Na₂O)",
                info="Typically between 0.0 (no activator / pure sodium hydroxide) to 2.5",
            )
            si_al_ratio = gr.Number(
                value=3.5,
                label="Precursor SiO₂/Al₂O₃ Ratio",
                info="Molar ratio of the precursor material (e.g., fly ash, metakaolin, volcanic ash)",
            )

            predict_btn = gr.Button("⚡ Predict Performance", variant="primary")

        with gr.Column():
            gr.Markdown("### Predicted Mix Results")
            output_ucs = gr.Textbox(
                label="Predicted Unconfined Compressive Strength (UCS)",
                placeholder="Awaiting inputs...",
            )
            output_wa = gr.Textbox(
                label="Predicted 24-hour Water Absorption",
                placeholder="Awaiting inputs...",
            )

            gr.Markdown(
                """
            > **Note on Model Resiliency:**
            > Missing inputs in the literature database are dynamically handled using K-Nearest Neighbors imputation to ensure predictions do not crash, even with complex, partially specified soil compositions.
            """
            )

    predict_btn.click(
        fn=predict_ceb_properties,
        inputs=[clay, silt, sand, silica_mod, si_al_ratio],
        outputs=[output_ucs, output_wa],
    )

if __name__ == "__main__":
    demo.launch(share=False)