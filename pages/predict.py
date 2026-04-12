import streamlit as st
from streamlit_drawable_canvas import st_canvas
from components.divider import divider
from PIL import Image
import io
from scripts.preprocessing import process_image, load_dataset_zip
import numpy as np
from collections import defaultdict
import tensorflow as tf

prediction_gallery = "premade_galleries/prediction_gallery.zip"

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

    with prediction_column:
        st.header("Prediction Gallery")

        with st.container(height=435):

            if st.session_state.prediction_gallery_loaded == True:
                st.markdown(
                    f'**Loaded Gallery**: {st.session_state.prediction_gallery_name}</span>',
                    unsafe_allow_html=True
                    )

            if st.session_state.prediction_gallery_loaded == True:
                grouped = defaultdict(list)
                for image, class_id in zip(st.session_state.prediction_samples, st.session_state.prediction_labels):
                    grouped[class_id].append(image)
                for idx, (class_id, samples) in enumerate(grouped.items()):
                    label_name = next(
                        (name for name, cid in st.session_state.prediction_label_id.items() if cid == class_id),
                        "Unknown"
                    )
                    with st.popover(f":primary[Class {class_id}] ({label_name}): {len(samples)} samples",
                                    use_container_width=True):
                        cols = st.columns(10)
                        for i, image in enumerate(samples):
                            with cols[i % 10]:
                                st.image(image)
            else:
                st.markdown("No model loaded.")

        with st.container(height=200):

            mode_options = ["Live", "Batch"]
            current_index = mode_options.index(st.session_state.prediction_mode)

            selected_mode = st.radio(
                "Prediction mode:",
                mode_options,
                index=current_index,
                disabled=st.session_state.prediction_gallery_loaded
            )
            if selected_mode != st.session_state.prediction_mode:
                st.session_state.prediction_mode = selected_mode
                st.rerun()

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

        with st.container(border=True, height=200 if st.session_state.prediction_mode == "Batch" else 125, vertical_alignment="distribute"):

            if st.session_state.prediction_mode == "Live":

                label_column, prediction_button_column = st.columns(2)

                with label_column:
                    label = st.text_input(label="Label your sample", width=300, autocomplete='off')

                with prediction_button_column:

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

            elif st.session_state.prediction_mode == "Batch":
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.file_uploader("Load a prediction gallery:",disabled=st.session_state.prediction_gallery_loaded)
                with col2:

                    if st.session_state.prediction_gallery_loaded == False:
                        load_prediction = st.button("Load _IP_:primary[L] Prediction Gallery")
                    else:

                        prediction = st.button("Predict Gallery",
                                               use_container_width=True,
                                               disabled=st.session_state.model_locked==False)

                        if prediction:

                            X = np.array(st.session_state.prediction_samples)
                            y_true = np.array(st.session_state.prediction_labels)

                            if len(X.shape) == 3:
                                X = X[..., np.newaxis]

                            model = st.session_state.model
                            pred_probs = model.predict(X)
                            y_pred = np.argmax(pred_probs, axis=1)
                            accuracy = np.mean(y_pred == y_true)
                            st.session_state.gallery_predicted = True

                        if st.button("Delete Prediction Gallery", use_container_width=True):

                            @st.dialog(f"Are you sure you want to delete {st.session_state.gallery_name}?")
                            def delete_gallery():
                                st.write("All samples will be removed and will not be recoverable.")
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if st.button("Yes", use_container_width=True):
                                        st.session_state.prediction_samples = []
                                        st.session_state.prediction_labels = []
                                        st.session_state.prediction_label_id = {}
                                        st.session_state.prediction_augmentation_type = []
                                        st.session_state.prediction_gallery_loaded = False
                                        st.session_state.gallery_predicted = False
                                        st.rerun()
                                with col2:
                                    if st.button("Cancel", use_container_width=True):
                                        st.rerun()

                            delete_gallery()

                    if not st.session_state.prediction_gallery_loaded and load_prediction:
                        load_dataset_zip(prediction_gallery,type="prediction")
                        st.session_state.prediction_gallery_name = "IPL"
                        st.session_state.prediction_gallery_loaded = True
                        st.rerun()


    with evaluation_column:
        st.header("Evaluation")

        with st.container(border=True, height=600, vertical_alignment="distribute"):

            if st.session_state.gallery_predicted:

                cm = tf.math.confusion_matrix(y_true, y_pred)
                st.subheader("Confusion Matrix")
                st.dataframe(cm.numpy())
                st.write(accuracy)
