import streamlit as st
from streamlit_drawable_canvas import st_canvas
from components.divider import divider
from components.charts import confusion_matrix_chart
from scripts.modelling import predict_model
import io
from scripts.preprocessing import process_image, load_dataset_zip
import numpy as np
from collections import defaultdict


prediction_gallery = "premade_galleries/prediction_gallery.zip"

def page ():

    st.title("Predict", text_alignment="center")
    st.markdown(
        "Create the data to test your _IP_:primary[L]. Draw and label your pictures  \n"
        "and hit predict when you want your model to guess.",
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
        st.header(f"Prediction Gallery ({len(st.session_state.prediction_samples)})")

        with st.container(height=500):

            if st.session_state.model_locked == True:

                if st.session_state.prediction_gallery_loaded == True:
                    st.markdown(
                        f'**Loaded Gallery**: {st.session_state.prediction_gallery_name}</span>',
                        unsafe_allow_html=True
                        )
                else:
                    st.markdown("**Gallery**: Live Gallery")

                with st.container(height=220,border=True):

                    if len(st.session_state.prediction_samples) > 0:

                        grouped = defaultdict(list)
                        for image, class_id in zip(
                                st.session_state.prediction_samples,
                                st.session_state.prediction_labels
                        ):
                            grouped[class_id].append(image)
                        id_to_label = {v: k for k, v in st.session_state.label_id.items()}
                        for class_name, class_id in st.session_state.label_id.items():
                            samples = grouped.get(class_id, [])
                            if len(samples) == 0:
                                continue
                            with st.popover(
                                    f":primary[Class {class_id}] ({class_name}): {len(samples)} samples",
                                    use_container_width=True
                            ):
                                cols = st.columns(10)
                                for i, image in enumerate(samples):
                                    with cols[i % 10]:
                                        st.image(image)
                    else:
                        st.markdown("No samples predicted yet.")

                with st.container(height=200,border=True):

                    if len(st.session_state.live_predictions) > 0:

                        id_to_label = {v: k for k, v in st.session_state.label_id.items()}

                        for idx in reversed(range(len(st.session_state.live_predictions))):
                            pred = st.session_state.live_predictions[idx]
                            image = st.session_state.prediction_samples[idx]

                            pred_id = int(pred[0])
                            label_name = id_to_label.get(pred_id, "Unknown")

                            col1, col2 = st.columns([1,4])

                            with col1:
                                st.image(image, width=70)

                            with col2:
                                st.markdown(
                                    f"""
                                    <div style="
                                        padding:6px;
                                        border-radius:12px;
                                        background-color:#2e2e33;
                                        border: 1px solid #3a3a40;
                                        margin-bottom:12px;
                                    ">
                                        <div style="font-size:12px; color:#bbbbbb;">
                                            {st.session_state.model_name} thinks ...
                                        </div>
                                        <div style="font-size:18px; font-weight:bold; color:white;">
                                            {label_name}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    else:
                        st.markdown("No samples predicted yet.")

            else:
                st.markdown("No model loaded.")

        with st.container(height=100):

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
                    label = st.selectbox(label="Label your sample", options=st.session_state.label_id.keys())

                with prediction_button_column:

                    prediction = st.button("Predict",
                                           use_container_width=True,
                                           disabled=st.session_state.model_locked==False)

                    file_name = "drawing.png"
                    buffer = io.BytesIO()

                    if prediction:

                        image = process_image(canvas_data.image_data)
                        if label not in st.session_state.label_id:
                            st.error(f"Unknown label: {label}")
                            st.stop()

                        class_id = st.session_state.label_id[label]
                        st.session_state.prediction_samples.append(image)
                        st.session_state.prediction_labels.append(class_id)

                        if canvas_data.image_data is not None:
                            live_y_pred, live_pred_probs = predict_model(
                                st.session_state.model,
                                canvas_data.image_data,
                                mode="live"
                            )

                        st.session_state.live_predictions.append(live_y_pred)
                        st.rerun()

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

                            batch_y_pred, batch_pred_probs = predict_model(
                                st.session_state.model,
                                st.session_state.prediction_samples,
                                mode="batch"
                            )

                            accuracy = np.mean(batch_y_pred == st.session_state.prediction_labels)

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

                        st.session_state.prediction_samples = []
                        st.session_state.prediction_labels = []
                        st.session_state.prediction_label_id = {}
                        st.session_state.prediction_augmentation_type = []
                        st.session_state.live_predictions = []

                        load_dataset_zip(prediction_gallery,type="prediction")
                        st.session_state.prediction_gallery_name = "IPL"
                        st.session_state.prediction_gallery_loaded = True
                        st.rerun()

        if st.session_state.model_locked == False:
            st.warning(
                f"Train {st.session_state.model_name} to be able to make predictions.")

    with evaluation_column:
        st.header("Evaluation")

        with st.container(border=True, height=600, vertical_alignment="distribute"):

            if len(st.session_state.prediction_samples) > 0:

                y_true = np.array(st.session_state.prediction_labels, dtype=np.int32)
                y_pred = np.array([int(p[0]) for p in st.session_state.live_predictions], dtype=np.int32)

                cm = confusion_matrix_chart(y_true, y_pred)

                st.plotly_chart(cm,
                                use_container_width=True,
                                config={
                                    "scrollZoom": False,
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": [
                                        "zoom2d", "pan2d", "select2d", "lasso2d",
                                        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
                                        "hoverCompareCartesian", "hoverClosestCartesian", "toImage",
                                        "toggleSpikelines",
                                        "toggleFullscreen"
                                    ],
                                    "displaylogo": False
                                })