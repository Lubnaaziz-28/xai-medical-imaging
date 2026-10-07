import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from xai_radar import XaiRadar, export_audit

st.set_page_config(page_title="XAI-Radar", layout="wide")
st.title("XAI-Radar: Multi-Method Explanation Consistency Checker")


def build_demo_model(input_shape=(64, 64, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def load_image(upload):
    img = Image.open(upload).convert("RGB").resize((64, 64))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def normalize_for_display(heatmap):
    h = heatmap - heatmap.min()
    h = h / (h.max() + 1e-8)
    return (h * 255).astype(np.uint8)


with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload medical image", type=["png", "jpg", "jpeg"])
    class_idx = st.number_input("Target class index", min_value=0, value=0, step=1)
    ig_steps = st.slider("Integrated Gradients steps", 10, 200, 50)
    run_btn = st.button("Run XAI-Radar")


col1, col2, col3 = st.columns(3)

if run_btn or "results" in st.session_state:
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        image = load_image(uploaded)
        model = build_demo_model()
        model.predict(np.expand_dims(image, axis=0))

        radar = XaiRadar(model, ig_steps=ig_steps)
        with st.spinner("Running explanations..."):
            result = radar.explain(image, class_idx=int(class_idx))

        st.session_state["results"] = result
        st.session_state["image"] = image

        metrics = result["metrics"]
        score = result["consistency_score"]
        verdict = result["verdict"]

        st.subheader("Trust Verdict")
        if verdict == "GREEN":
            st.success(f"GREEN — Consistent explanations (score: {score:.2f})")
        elif verdict == "YELLOW":
            st.warning(f"YELLOW — Moderate disagreement (score: {score:.2f})")
        else:
            st.error(f"RED — High disagreement (score: {score:.2f})")

        st.subheader("Explanation Heatmaps")
        gradcam_img = Image.fromarray(normalize_for_display(result["gradcam"]))
        shap_img = Image.fromarray(normalize_for_display(result["shap"]))
        ig_img = Image.fromarray(normalize_for_display(result["integrated_gradients"]))

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)
        with col2:
            st.image(shap_img, caption="SHAP", use_column_width=True)
            st.image(ig_img, caption="Integrated Gradients", use_column_width=True)

        st.subheader("Consistency Heatmap")
        stack = np.stack([result["gradcam"], result["shap"], result["integrated_gradients"]], axis=0)
        std_map = np.std(stack, axis=0)
        consistency_map = 1.0 - (std_map / (std_map.max() + 1e-8))
        consistency_img = Image.fromarray(normalize_for_display(consistency_map))
        st.image(consistency_img, caption="Per-pixel consistency (brighter = more consistent)", use_column_width=True)

        st.subheader("Metrics")
        st.json(metrics)

        audit_path = "/tmp/xai_radar_audit.json"
        export_audit(result, audit_path, metadata={"class_idx": int(class_idx)})
        with open(audit_path, "rb") as f:
            st.download_button(
                "Download JSON Audit Log",
                data=f,
                file_name="xai_radar_audit.json",
                mime="application/json",
            )
