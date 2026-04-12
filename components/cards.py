import streamlit as st

def render_cards(hparams: dict):

    HP_SCHEMA = {
        "Optimisation": [
            ("learning_rate", "Learning Rate"),
            ("batch_size", "Batch Size"),
            ("batch_norm", "Batch Normalisation"),
            ("optimiser", "Optimiser"),
        ],
        "Architecture": [
            ("conv_layers", "Convolutional Layers"),
            ("filters", "Base Filters"),
            ("kernel_size", "Kernel Size"),
            ("dense_units", "Dense Units"),
        ],
        "Regularisation": [
            ("dropout", "Dropout"),
            ("weight_decay", "Weight Decay"),
        ],
        "Fusion": [
            ("fusion_mode", "Fusion Mode"),
            ("fusion_point", "Fusion Point"),
            ("merge_strategy", "Merge Strategy"),
        ]
    }

    for section_name, items in HP_SCHEMA.items():
        filtered = [(label, hparams[key]) for key, label in items if key in hparams]

        if not filtered:
            continue

        st.markdown(f"### {section_name}")
        for i in range(0, len(filtered), 2):
            cols = st.columns(2)

            for col, (label, value) in zip(cols, filtered[i:i+2]):
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            padding:10px;
                            border-radius:10px;
                            background-color:#2e2e33;
                            margin-bottom:8px;
                            transition: all 0.2s ease;
                        ">
                            <div style="font-size:12px; color:white;">{label}</div>
                            <div style="font-size:18px; font-weight:bold; color:white;">
                                {value}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )