import streamlit as st
from streamlit_drawable_canvas import st_canvas
from components.divider import divider
from PIL import Image
import io
from scripts.preprocessing import process_image
import numpy as np

def page ():

    st.title("Predict", text_alignment="center")
    st.markdown(
        "Choose a model and train your _IP_:primary[L]. Draw and label your pictures "
        "and find them preprocessed in the data gallery.",
        text_alignment="center"
    )

    divider()

    st.sidebar.header("Prediction Canvas")

    stroke_width = st.sidebar.slider("Stroke width", 1, 50, 10)
    canvas_size = st.sidebar.slider("Canvas size", 28, 400, 400)
    drawing_mode = st.sidebar.selectbox("Drawing Mode", options=["Freedraw", "Line", "Polygon"])

    divider(sidebar=True)

    canvas_column, prediction_column, evaluation_column = st.columns([1, 1, 1])

    # Canvas
    with canvas_column:
        st.header("Prediction Canvas", help="Hi")

        canvas_data = st_canvas(
            height=canvas_size,
            width=canvas_size,
            background_color="white",
            stroke_color="black",
            stroke_width=stroke_width,
            drawing_mode=drawing_mode.lower(),
        )

        with st.container(border=True, height=115, vertical_alignment="distribute"):

            prediction = st.button("Predict",use_container_width=True)
            file_name = "drawing.png"
            buffer = io.BytesIO()

            if canvas_data.image_data is not None:
                image = Image.fromarray(canvas_data.image_data.astype("uint8"), "RGBA")
                image.save(buffer, format="PNG")

            save = st.download_button("Download",
                                      data=buffer,
                                      file_name=file_name,
                                      use_container_width=True)

            if prediction and st.session_state.model_locked == True:
                processed_image = process_image(canvas_data.image_data)
                processed_image = processed_image[..., np.newaxis]
                processed_image = np.expand_dims(processed_image, axis=0)

    with prediction_column:
        st.header("Prediction")

        with st.container(height=600):

            if prediction and st.session_state.model_locked == True:
                st.markdown(f"Loaded Model: {st.session_state.model_name}")

                st.subheader("Prediction")
                st.image(processed_image)

                pred_probs = st.session_state.model.predict(processed_image)
                pred_class = np.argmax(pred_probs, axis=1)[0]
                pred_label_id = st.session_state.label_id
                st.markdown(f"Predicted class index: {pred_class} ({pred_label_id})")

            else:
                st.markdown("No model loaded.")

    with evaluation_column:
        st.header("Evaluation")
