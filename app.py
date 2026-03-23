import streamlit as st
from streamlit_drawable_canvas import st_canvas
from preprocessor import process_image, create_dataset_zip, load_dataset_zip, augment_images
import numpy as np
from collections import defaultdict
from PIL import Image
import io
import zipfile
from collections import Counter
import pandas as pd
import plotly.express as px

# --- Config --- #

scipl1 = "scipl1.zip"
scipl2 = []

colour_map = {
    "non-augmented": "#fffc9e",
    "occluded": "#ffb3ab",
    "rotated": "#99e6b3",
    "translated": "#c9abf5",
    "zoomed": "#ffd4a1"
}

st.set_page_config(page_title="Interactive Pixel Learner",layout="wide")
st.markdown("# _Interactive_ Pixel :primary[Learner] \U0001F916")

for key in ["samples","labels","label_id","next_label_id","canvas_key","gallery_loaded","gallery_name","augmentation_type"]:
    if key not in st.session_state:
        if key in ["samples", "labels","augmentation_type"]:
            st.session_state[key] = []
        elif key in ["label_id"]:
            st.session_state[key] = {}
        elif key == "gallery_loaded":
            st.session_state[key] = False
        else:
            st.session_state[key] = 0

def divider(sidebar=False):
    if sidebar:
        return st.sidebar.markdown("<hr style='margin-top:1px;margin-bottom:1px;'>",unsafe_allow_html=True)
    else:
        return st.markdown("<hr style='margin-top:1px;margin-bottom:1px;'>",unsafe_allow_html=True)

def reindex_classes():
    old_label_id = st.session_state.label_id.copy()
    sorted_labels = sorted(old_label_id.items(), key=lambda x: x[1])
    new_label_id = {}
    for new_id, (label, _) in enumerate(sorted_labels):
        new_label_id[label] = new_id
    st.session_state.label_id = new_label_id
    new_labels = []
    for label in st.session_state.labels:
        for label_name, old_id in old_label_id.items():
            if label == old_id:
                new_labels.append(new_label_id[label_name])
                break
    st.session_state.labels = new_labels

# --- UI --- #

# -- Sidebar -- #
st.sidebar.title("_IP_:primary[L] \U0001F916")

divider(sidebar=True)

# -- Page Box -- #

pages = ["_IP_:primary[L]","Capture","Train","Predict","About"]
if "current_page" not in st.session_state:
    st.session_state.current_page = "Capture"

with st.container(border=False,width=500,height=75,horizontal_alignment="left",vertical_alignment="center"):
    page_columns = st.columns(5,gap="small")
    for i, page_name in enumerate(pages):
        active_page = st.session_state.current_page == page_name
        button_label = f":primary[**{page_name}**]" if active_page else page_name
        if page_columns[i].button(button_label, key=f"btn_{i}",use_container_width=True):
            st.session_state.current_page = page_name
            st.rerun()

divider()

if st.session_state.current_page == "Home":
    st.header("Home")

# -- Capture Page -- #
if st.session_state.current_page == "Capture":

    # - Capture Sidebar - #

    # Canvas Sidebar
    st.sidebar.header("Capture Canvas")

    stroke_width = st.sidebar.slider("Stroke width", 1, 50, 10)
    canvas_size = st.sidebar.slider("Canvas size", 28, 400, 400)
    drawing_mode = st.sidebar.selectbox("Drawing Mode",options=["Freedraw","Line","Polygon"])

    divider(sidebar=True)

    class_names = list(st.session_state.label_id.keys())

    # Gallery Sidebar
    st.sidebar.header(f"Data Gallery ({len(st.session_state.samples)})")

    with st.sidebar.expander("Classes and Samples",expanded=True):

        # Class Selector
        selected_class = st.selectbox(f"**Select class** ({len(st.session_state.label_id)}):", class_names)

        if len(class_names) > 0:

            class_id = st.session_state.label_id[selected_class]
            renamed = st.text_input("Rename Class:",autocomplete="off")

            cols = st.columns([1,1])
            with cols[0]:
                if st.button("Rename", use_container_width=True):
                    if renamed and renamed not in st.session_state.label_id:
                        new_dict = {}
                        for k, v in st.session_state.label_id.items():
                            if k == selected_class:
                                new_dict[renamed] = v
                            else:
                                new_dict[k] = v
                        st.session_state.label_id = new_dict
                        st.rerun()
            with cols[1]:
                if st.button("Delete Class", use_container_width=True):
                    new_samples = []
                    new_labels = []
                    new_aug_types = []

                    for img, label, aug in zip(
                            st.session_state.samples,
                            st.session_state.labels,
                            st.session_state.augmentation_type
                    ):
                        if label != class_id:
                            new_samples.append(img)
                            new_labels.append(label)
                            new_aug_types.append(aug)

                    st.session_state.samples = new_samples
                    st.session_state.labels = new_labels
                    st.session_state.augmentation_type = new_aug_types
                    del st.session_state.label_id[selected_class]

                    reindex_classes()

                    if len(st.session_state.samples) == 0:
                        st.session_state.gallery_loaded = False

                    st.rerun()

        # Sample Selector
        if selected_class:

            divider()

            cols = st.columns(2)
            class_id = st.session_state.label_id[selected_class]
            class_indices = [i for i, label in enumerate(st.session_state.labels) if label == class_id]
            class_images = [
                image for image, label in zip(st.session_state.samples, st.session_state.labels)
                if label == class_id
            ]
            if class_images:
                image_index = st.selectbox(
                    f"**Select sample** ({len(class_images)}):", list(range(len(class_images))),
                    format_func=lambda x: f"Sample {x + 1}"
                )
                with st.container(border=False, height=100):
                    gallery_sidebar_columns = st.columns([1,1.5])
                    with gallery_sidebar_columns[0]:
                        st.image(class_images[image_index], width=100)
                    with gallery_sidebar_columns[1]:

                        # Delete Sample
                        delete = st.button("Delete Sample", use_container_width=True)
                        if delete:
                            original_index = class_indices[image_index]
                            del st.session_state.samples[original_index]
                            del st.session_state.labels[original_index]
                            del st.session_state.augmentation_type[original_index]
                            if class_id not in st.session_state.labels:
                                del st.session_state.label_id[selected_class]
                                selected_class = None
                            if len(st.session_state.labels) == 0:
                                st.session_state.gallery_loaded = False
                            st.rerun()

                        # Save Sample
                        image = class_images[image_index]
                        if image.dtype != np.uint8:
                            image = (image * 255).clip(0, 255).astype(np.uint8)
                        image = Image.fromarray(image)
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        buffer.seek(0)
                        save = st.download_button("Download",
                                                  data=buffer.getvalue(),
                                                  file_name=f"{selected_class}_sample_{image_index+1}.png",
                                                  mime="image/png",
                                                  use_container_width=True)

    with st.sidebar.expander("Augmentation",expanded=True):

            a_p_addition = st.slider("**Percentage Addition:**",min_value=1,max_value=100,value=25,step=1)
            a_per_class = st.checkbox("Per Class",value=True)
            a_strength = st.slider("**Augmentation Strength:**", min_value=1, max_value=100, value=50,step=1)

            divider()
            a_occ_type = st.selectbox("**Occlusion Type:**",options=["Smudge","Noise","Block"],help="Occlusion Type")
            divider()
            a_zoom_type = st.selectbox("**Zoom Type:**", options=["In","Out"], help="Zoom Type")

    divider(sidebar=True)

    # Distribution Sidebar
    st.sidebar.header("Distribution")

    st.sidebar.subheader("Class Distribution")

    st.sidebar.subheader("Augmentation Distribution")
    by_class = st.sidebar.checkbox(f"By Class (Class {st.session_state.label_id[selected_class]})" if st.session_state.samples else "By Class")

    # - Capture Header - #
    st.title("Capture", text_alignment="center")
    st.markdown("Create the data to train the _IP_:primary[L]. Draw and label your pictures  \n"
                "and find them preprocessed in the data gallery.", text_alignment="center")

    divider()

    # - Capture Body - #
    canvas_column, gallery_column, distribution_column = st.columns([1,1,1])

    # Canvas
    with canvas_column:
        st.header("Capture Canvas",help="Hi")

        canvas_data = st_canvas(
            height=canvas_size,
            width=canvas_size,
            background_color="white",
            stroke_color="black",
            stroke_width=stroke_width,
            drawing_mode=drawing_mode.lower(),
        )

        with st.container(border=True,height=115,vertical_alignment="distribute"):
            label_column, capture_button_column = st.columns(2)

            # Label
            with label_column:
                label = st.text_input(label="Label your sample", width=300,autocomplete='off')

            with capture_button_column:

                # Capture Button
                capture = st.button("Capture",use_container_width=True)
                file_name = "drawing.png"
                buffer = io.BytesIO()

                if canvas_data.image_data is not None:
                    image = Image.fromarray(canvas_data.image_data.astype("uint8"), "RGBA")
                    file_name = f"{label}.png" if label else "drawing.png"
                    image.save(buffer, format="PNG")

                # Save Button
                save = st.download_button("Download",
                                          data= buffer,
                                          file_name=file_name,
                                          use_container_width=True)

                if capture and label != "":
                    image = process_image(canvas_data.image_data)
                    label = label.lower().strip()
                    if label not in st.session_state.label_id:
                        existing_ids = set(st.session_state.label_id.values())
                        new_id = 0
                        while new_id in existing_ids:
                            new_id += 1
                        st.session_state.label_id[label] = new_id
                    class_id = st.session_state.label_id[label]
                    st.session_state.samples.append(image)
                    st.session_state.labels.append(class_id)
                    st.session_state.augmentation_type.append("non-augmented")
                    st.rerun()

        if capture and label == "":
            st.warning("Please enter a label before capturing")

    # Gallery
    with gallery_column:
        st.header(f"Data Gallery ({len(st.session_state.samples)})", help="Gallery is ")

        with st.container(height=240):
            if st.session_state.gallery_loaded == True:
                if st.session_state.gallery_name in ["SCIPL-1","SCIPL-2","SCIPL-3"]:
                    st.markdown(f'**Loaded Gallery**: <span style="color: #b0ff9e;">{st.session_state.gallery_name}</span>',unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"**Loaded Gallery**: {st.session_state.gallery_name}",unsafe_allow_html=True)

            if len(st.session_state.samples) == 0:
                st.write("No samples captured yet.")
            else:
                grouped = defaultdict(list)
                for image, class_id in zip(st.session_state.samples, st.session_state.labels):
                    grouped[class_id].append(image)
                for idx, (class_id, samples) in enumerate(grouped.items()):
                    label_name = next(
                        (name for name, cid in st.session_state.label_id.items() if cid == class_id),
                        "Unknown"
                    )
                    with st.expander(f":primary[Class {class_id}] ({label_name}): {len(samples)} samples"):
                        cols = st.columns(10)
                        for i, image in enumerate(samples):
                            with cols[i % 10]:
                                st.image(image)

        with st.container(border=True, height=310, vertical_alignment="distribute"):

            upload_column, download_column = st.columns(2)

            with upload_column:

                # Upload Gallery
                uploaded_file = st.file_uploader("Load Gallery:", type="zip")
                if uploaded_file is not None and not st.session_state.gallery_loaded:
                    load_dataset_zip(uploaded_file)
                    st.session_state.gallery_loaded = True
                    st.session_state.gallery_name = uploaded_file.name
                    st.rerun()

                # Premade Gallery
                premade_gallery = st.selectbox(label="Choose a :primary[premade] gallery:", options=["SCIPL-1", "SCIPL-2","CHIPL","TRIPL"],help="Premade gallery is ")

            with download_column:

                # Augment Gallery
                augmentation = st.radio(label="Choose how to :primary[augment] your gallery:",
                               options=["Occlude","Rotate","Translate","Zoom"],
                               help="Augmentation options",
                               )

                if st.session_state.label_id:
                    if st.button("Augment Gallery", use_container_width=True):
                        dataset_zip = create_dataset_zip()
                        augmented_zip = augment_images(dataset_zip,
                                                       augmentation,
                                                       a_occ_type,
                                                       a_zoom_type,
                                                       a_p_addition,
                                                       a_per_class,
                                                       a_strength)
                        st.session_state.samples = []
                        st.session_state.labels = []
                        st.session_state.augmentation_type = []
                        load_dataset_zip(augmented_zip)

                        st.rerun()

                    # Download Gallery
                    st.download_button(
                        "Download Gallery",
                        data=create_dataset_zip(),
                        file_name="pixel_dataset.zip",
                        mime="application/zip",
                        use_container_width=True,)

                    delete_gallery = st.button("Delete Gallery",use_container_width=True)
                    if delete_gallery:
                        st.session_state.samples = []
                        st.session_state.labels = []
                        st.session_state.label_id = {}
                        st.session_state.augmentation_type = []
                        st.session_state.gallery_loaded = False
                        st.rerun()

                else:
                    load_premade = st.button(label="Load Premade Gallery", use_container_width=True)
                    if load_premade and not st.session_state.gallery_loaded:
                        if premade_gallery == "SCIPL-1":
                            load_dataset_zip(scipl1)
                            st.session_state.gallery_loaded = True
                            st.session_state.gallery_name = "SCIPL-1"
                            st.rerun()

    with distribution_column:
        st.header("Distribution")

        with st.container(height=310, vertical_alignment="top"):
            if st.session_state.labels:
                st.markdown("**Class Distribution**:")
                counts = Counter(st.session_state.labels)
                data = pd.DataFrame({
                    "class": [name for name, cid in st.session_state.label_id.items()],
                    "samples": [counts.get(cid, 0) for cid in st.session_state.label_id.values()]
                })
                aug_keys = list(colour_map.keys())
                fig_bar = px.bar(
                    data,
                    x="class",
                    y="samples",
                    text="samples",
                )
                fig_bar.update_layout(
                    showlegend=False,
                    yaxis_title="Samples",
                    xaxis_title="Class",
                    height=245
                )
                fig_bar.update_traces(marker_color="#fffc9e")
                fig_bar.update_layout(
                    margin=dict(l=20, r=20, t=5, b=20))
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown("No classes created yet.")

        with st.container(height=240,vertical_alignment="top"):
            if st.session_state.labels:
                st.markdown("**Augmentation Distribution (Data Gallery)**:" if st.session_state.gallery_loaded == False
            else f"**Augmentation Distribution ({st.session_state.gallery_name})**:")
                aug_counts = pd.Series(st.session_state.augmentation_type).value_counts().reset_index()
                aug_counts.columns = ["augmentation", "count"]
                fig_pie = px.pie(
                    aug_counts,
                    names="augmentation",
                    values="count",
                    height=170,
                    width=170,
                    color="augmentation",
                    color_discrete_map=colour_map
                )
                fig_pie.update_layout(
                    margin=dict(l=20, r=20, t=10, b=20))
                st.plotly_chart(fig_pie, use_container_width=True,config={'displayModeBar': False, 'displaylogo': False})
            else:
                st.markdown("No augmentations yet.")

    divider()

if st.session_state.current_page == "Train":
    st.title("Train", text_alignment="center")
    st.markdown("Choose a model and train your _IP_:primary[L]. Draw and label your pictures  \n"
                "and find them preprocessed in the data gallery.", text_alignment="center")

    divider()

    model_column, tuning_column, visualisation_column = st.columns([1, 1, 1])

    model = "SCIPL"

    with model_column:
        st.header("Models")

        with st.container(height=340, vertical_alignment="top"):
            if model == "SCIPL":
                st.markdown("**Loaded Model**: Blank SCIPL")

        with st.container(height=240, vertical_alignment="top"):

            first_col, second_col = st.columns([1,1])

            with first_col:
                model = st.selectbox("Select a model:", ["SCIPL", "CHIPL", "TRIPL"],help="None")
                model_type = st.radio("Select model type:",["Blank","Pretrained"])
            with second_col:
                loaded_model = st.file_uploader("Load model")

            train = st.button("Train Model",use_container_width=True)

    with tuning_column:
        st.header("Tuning")

        with st.container(height=600, vertical_alignment="top"):

            st.markdown("**Loaded Model**: Blank SCIPL")

    with visualisation_column:
        st.header("Visualisation")






