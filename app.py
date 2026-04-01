import streamlit as st
from pages.train import page as train_page
from pages.capture import page as capture_page
from pages.predict import page as predict_page
from components.divider import divider

# --- Config --- #

st.set_page_config(page_title="Interactive Pixel Learner",layout="wide")

st.markdown("""
<style>
/* Multiselect selected items (tags) */
[data-baseweb="tag"] {
    background-color: #333333 !important;
    color: white !important;
}

/* Dropdown highlight */
[data-baseweb="select"] div[role="option"][aria-selected="true"] {
    background-color: #333333 !important;
}

/* Hover color */
[data-baseweb="select"] div[role="option"]:hover {
    background-color: #333333 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Session --- #

DEFAULTS = {
    "current_page": "Capture",
    "samples": [],
    "labels": [],
    "label_id": {},
    "next_label_id": 0,
    "canvas_key": 0,
    "gallery_loaded": False,
    "gallery_name": "",
    "augmentation_type": [],
    "model": None,
    "model_name": "SCIPL",
    "model_type": "Blank",
    "model_locked": False,
    "model_searched": False,
    "model_state_loaded": False,
    "train_cycles": 0,
    "selection_mode": "Manual",
    "search_params": [],
    "search_history": {
        "trial": [],
        "value":[],
        "params": [],
    },
    "history_data": {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": []
    },
    "loss_chart": None,
    "acc_chart": None,
    "search_chart": None
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        if key in ["loss_chart", "acc_chart", "search_chart"]:
            st.session_state[key] = st.empty()
        else:
            st.session_state[key] = default_value

# --- UI --- #

st.markdown("# _Interactive_ Pixel :primary[Learner] \U0001F916")

# -- Sidebar -- #
st.sidebar.title("_IP_:primary[L] \U0001F916")

divider(sidebar=True)

# -- Page Box -- #

pages = ["_IP_:primary[L]","Capture","Train","Predict","About"]

with st.container(border=False,width=500,height=75,horizontal_alignment="left",vertical_alignment="center"):
    page_columns = st.columns(5,gap="small")
    for i, page_name in enumerate(pages):
        active_page = st.session_state.current_page == page_name
        button_label = f":primary[**{page_name}**]" if active_page else page_name
        if page_columns[i].button(button_label, key=f"btn_{i}",use_container_width=True):
            st.session_state.current_page = page_name
            st.rerun()

divider()

# -- Home Page -- #
if st.session_state.current_page == "Home":
    st.header("Home")

# -- Capture Page -- #
if st.session_state.current_page == "Capture":
    capture_page()

# -- Train Page -- #
if st.session_state.current_page == "Train":
    train_page()

if st.session_state.current_page == "Predict":
    predict_page()




