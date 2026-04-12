import streamlit as st
import cv2
import io
import zipfile
import numpy as np
from PIL import Image
import random
from collections import defaultdict

def process_image(image):
    image_gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    image_small = cv2.resize(image_gray, (28, 28)) / 255.0
    image_small = image_small.reshape(28, 28)
    return image_small

def augment_images(zip_buffer, method, occlusion_type, zoom_type, addition, per_class, strength):

    addition = float(addition)
    new_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "r") as zin, zipfile.ZipFile(new_buffer, "w") as zout:

        files = sorted(
            [f for f in zin.namelist() if not f.endswith("/")],
            key=lambda x: parse_filename(x)[0]
        )

        if not per_class:
            num_to_augment = max(1, int(len(files) * (addition / 100)))
            files_to_augment = set(random.sample(files, num_to_augment))
        else:
            class_dict = defaultdict(list)
            for f in files:
                class_name = f.split("/")[0]
                class_dict[class_name].append(f)

            files_to_augment = set()
            for class_files in class_dict.values():
                num_to_augment = max(1, int(len(class_files) * (addition / 100)))
                files_to_augment.update(random.sample(class_files, num_to_augment))

        for file in files:
            image_bytes = zin.read(file)
            zout.writestr(file, image_bytes)

            if file not in files_to_augment:
                continue

            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image).astype(np.float32)

            class_name = file.split("/")[0]
            filename = file.split("/")[-1]
            base = filename.split("__")[0]

            if method == "Occlude":
                aug = image_np.copy()
                h, w = aug.shape[:2]

                if occlusion_type == "Smudge":
                    num_smudges = max(1, int(3 * np.sqrt(strength)))
                    for _ in range(num_smudges):
                        occ_w = np.random.randint(w // 16, max(w // 3, 1))
                        occ_h = np.random.randint(h // 16, max(h // 3, 1))
                        x = np.random.randint(0, max(1, w - occ_w))
                        y = np.random.randint(0, max(1, h - occ_h))

                        mask = np.random.rand(occ_h, occ_w)
                        mask = cv2.GaussianBlur(mask, (31, 31), 0)
                        if aug.ndim == 3:
                            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                        aug[y:y+occ_h, x:x+occ_w] *= (1 - mask * 0.5)

                elif occlusion_type == "Noise":
                    noise_frac = 0.005 * strength
                    noise_mask = np.random.rand(*aug.shape[:2]) < noise_frac
                    aug[noise_mask] = np.random.randint(0, 256, np.sum(noise_mask))

                elif occlusion_type == "Block":
                    num_blocks = max(1, int(1 + 5 * (strength / 100)))

                    for _ in range(num_blocks):
                        min_size = max(1, int(0.05 * w))
                        max_size = max(min_size + 1, int(0.2 * w))

                        occ_w = np.random.randint(min_size, max_size)
                        occ_h = np.random.randint(min_size, max_size)

                        x = np.random.randint(0, max(1, w - occ_w))
                        y = np.random.randint(0, max(1, h - occ_h))

                        aug[y:y + occ_h, x:x + occ_w] = 0

                aug = np.clip(aug, 0, 255).astype(np.uint8)
                aug_pil = Image.fromarray(aug)
                aug_type = "occluded"

            elif method == "Rotate":
                max_degree = 5 + (strength / 100) * 85
                angle = random.uniform(-max_degree, max_degree)
                aug_pil = image.rotate(angle, fillcolor=255)
                aug_type = "rotated"

            elif method == "Translate":
                w, h = image.size
                dx = random.randint(-int(0.2*w), int(0.2*w))
                dy = random.randint(-int(0.2*h), int(0.2*h))
                canvas = Image.new(image.mode, (w, h), 255)
                canvas.paste(image, (dx, dy))
                aug_pil = canvas
                aug_type = "translated"

            elif method == "Zoom":
                w, h = image.size
                zoom_factor = 1 + (strength / 100) * 0.5
                new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
                zoomed = image.resize((new_w, new_h))

                left = (new_w - w) // 2
                top = (new_h - h) // 2
                aug_pil = zoomed.crop((left, top, left + w, top + h))
                aug_type = "zoomed"

            aug_bytes = io.BytesIO()
            aug_pil.save(aug_bytes, format="PNG")

            new_filename = f"{class_name}/{base}__aug__{aug_type}.png"
            zout.writestr(new_filename, aug_bytes.getvalue())

    new_buffer.seek(0)
    return new_buffer

def create_dataset_zip():
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as zf:
        for i, (image, label, aug_type) in enumerate(
            zip(
                st.session_state.samples,
                st.session_state.labels,
                st.session_state.augmentation_type
            )
        ):
            class_name = next(
                name for name, cid in st.session_state.label_id.items() if cid == label
            )

            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_bytes = io.BytesIO()
            image_pil.save(image_bytes, format="PNG")

            filename = f"{class_name}/img_{i}__aug__{aug_type}.png"
            zf.writestr(filename, image_bytes.getvalue())

    buffer.seek(0)
    return buffer

def load_dataset_zip(uploaded_file,type="capture"):
    with zipfile.ZipFile(uploaded_file) as z:

        files = [f for f in z.namelist() if f.endswith((".png", ".jpg"))]

        files = sorted(files, key=lambda x: parse_filename(x)[0])

        for file in files:

            class_name = file.split("/")[0]

            if type == "capture":

                if class_name not in st.session_state.label_id:
                    new_id = len(st.session_state.label_id)
                    st.session_state.label_id[class_name] = new_id

                class_id = st.session_state.label_id[class_name]

                image_bytes = z.read(file)
                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                image_array = np.array(image)

                if image_array.max() > 1:
                    image_array = image_array / 255.0

                _, aug_type = parse_filename(file)

                st.session_state.samples.append(image_array)
                st.session_state.labels.append(class_id)
                st.session_state.augmentation_type.append(aug_type)

            else:

                if class_name not in st.session_state.prediction_label_id:
                    new_id = len(st.session_state.prediction_label_id)
                    st.session_state.prediction_label_id[class_name] = new_id

                class_id = st.session_state.prediction_label_id[class_name]

                image_bytes = z.read(file)
                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                image_array = np.array(image)

                if image_array.max() > 1:
                    image_array = image_array / 255.0

                _, aug_type = parse_filename(file)

                st.session_state.prediction_samples.append(image_array)
                st.session_state.prediction_labels.append(class_id)
                st.session_state.prediction_augmentation_type.append(aug_type)



def parse_filename(file):
    filename = file.split("/")[-1]

    if filename.startswith("img_"):
        index_str = filename.split("img_")[1].split("__")[0]

    elif filename.startswith("image_"):
        index_str = filename.split("image_")[1].split(".")[0]

    else:
        index_str = "0"

    try:
        index = int(index_str)
    except ValueError:
        index = 0

    if "__aug__" in filename:
        aug_type = filename.split("__aug__")[1].split(".")[0]
    else:
        aug_type = "non-augmented"

    return index, aug_type