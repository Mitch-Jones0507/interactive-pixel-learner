import streamlit as st
from components.divider import divider
from components.charts import search_chart, acc_loss_chart
from components.conceptual_graph import ConceptualGraph
from scripts.modelling import build_model, compile_model, train_model, hyperparameter_search
import numpy as np
import tensorflow as tf
import pandas as pd
import optuna

colour_map = {"scipl": "#b0ff9e", "CHIPL": "#9ecdff", "TRIPL": "#ff9eef"}

class LivePlotCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k in st.session_state.history_data:
            st.session_state.history_data[k].append(logs.get(k))
        df = pd.DataFrame(st.session_state.history_data)
        acc_loss_chart(type="loss")
        acc_loss_chart(type="accuracy")

class LiveSearchCallback:
    def __call__(self, study, trial):
        st.session_state.search_history["trial"].append(trial.number)
        st.session_state.search_history["value"].append(trial.value)
        st.session_state.search_history["params"].append(trial.params)
        search_chart()

def page():

    st.sidebar.header(f"Models")
    st.sidebar.header(f"Tuning")
    st.sidebar.subheader("Hyperparameters")

    patience = st.sidebar.slider("Patience", 1, 20, value=5, step=1)
    min_delta = st.sidebar.slider("Min Delta", 0.01, 1.0, value=0.1, step=0.1)
    search_trials = st.sidebar.slider("Search Trials", 1, 20, value=10,step=1)

    st.title("Train", text_alignment="center")
    st.markdown(
        "Choose a model and train your _IP_:primary[L]. Draw and label your pictures "
        "and find them preprocessed in the data gallery.",
        text_alignment="center"
    )
    divider()

    model_column, tuning_column, visualisation_column = st.columns([1,1,1])

    with visualisation_column:
        st.header(f"Visualisation")
        with st.container(height=275,border=True):
            st.subheader(f"**Search History**:")
            if "search_chart" not in st.session_state:
                st.session_state.search_chart = st.empty()
            if "search_history" in st.session_state and st.session_state.search_history["trial"]:
                search_chart()
            else:
                st.write("No hyperparameter search done.")
                st.session_state.search_chart = st.empty()
        with st.container(height=490,border=True):
            st.subheader(f"**Train History ({st.session_state.train_cycles} Cycles):**")
            if st.session_state.history_data["loss"]:
                acc_loss_chart(type="loss")
                acc_loss_chart(type="accuracy")
            else:
                st.write("No model trained yet.")
                st.session_state.loss_chart = st.empty()
                st.session_state.acc_chart = st.empty()

    with model_column:

        unique_labels = sorted(set(st.session_state.labels))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        st.session_state.labels_mapped = np.array([label_map[l] for l in st.session_state.labels])
        num_classes = len(unique_labels)

        st.header("Models")

        top = st.container(height=350)
        bottom = st.container(height=235)

        if "model" not in st.session_state or st.session_state.model is None:
            st.session_state.model = build_model("SCIPL", num_classes=num_classes)
            st.session_state.model = compile_model(st.session_state.model)
            for k in st.session_state.history_data:
                st.session_state.history_data[k] = []

        with bottom:
            col1, col2 = st.columns(2)
            with col1:
                model_options = ["SCIPL", "CHIPL", "TRIPL"]
                st.selectbox(
                    "Select a :primary[model]:",
                    model_options,
                    key="model_name",
                    disabled=st.session_state.model_locked
                )
                st.radio("Select model type:", ["Blank","Pretrained"], key="model_type",
                         disabled=st.session_state.model_locked)
            with col2:
                st.file_uploader("Load model:", disabled=st.session_state.model_locked)

        with top:
            st.markdown(
                f'**Loaded Model**: {"Searched " if st.session_state.model_searched else "Trained " if st.session_state.model_locked else "Blank " if st.session_state.model_type == "Blank" else "Pretrained "}'
                f'<span style="color: {colour_map[st.session_state.model.name]};">'
                f'{st.session_state.model.name}</span>',
                unsafe_allow_html=True
            )

            divider()

            st.subheader("Conceptual Graph:")

            if st.session_state.model and st.session_state.model_locked and not st.session_state.model_searched:
                graph = ConceptualGraph(st.session_state.model)
                graph.plot_conceptual_graph()
            else:
                st.markdown("No model trained yet.")

        train = st.button(f"Search {st.session_state.model.name}" if st.session_state.selection_mode == "Search" and st.session_state.model_locked == False
                          else f"Train {st.session_state.model.name}" if st.session_state.train_cycles == 0
                            else f"Retrain {st.session_state.model.name}"
                          ,use_container_width=True)

        if train and len(st.session_state.samples) <= 20:
            st.warning(f"Capture more than 20 samples ({len(st.session_state.samples)} samples).")

        if st.session_state.model_locked:
            col1, col2 = st.columns(2)
            with col1:
                delete_model = st.button(f"Delete {st.session_state.model.name}", use_container_width=True)

                if delete_model:
                    st.session_state.train_cycles = 0
                    st.session_state.model = None
                    st.session_state.model_locked = False
                    st.session_state.model_searched = False
                    for k in st.session_state.history_data:
                        st.session_state.history_data[k] = []
                    st.session_state.search_params = []
                    for k in st.session_state.search_history:
                        st.session_state.search_history[k] = []
                    st.session_state.loss_chart.empty()
                    st.session_state.acc_chart.empty()
                    st.session_state.search_chart.empty()
                    st.rerun()
            with col2:
                download_model = st.button(f"Download {st.session_state.model.name}",use_container_width=True)

    with tuning_column:
        st.header("Tuning")
        with st.container(height=740):
            st.markdown(
                f'**Loaded Model**: {"Searched " if st.session_state.model_searched else "Trained " if st.session_state.model_locked else "Blank " if st.session_state.model_type == "Blank" else "Pretrained "}'
                f'<span style="color: {colour_map[st.session_state.model.name]};">'
                f'{st.session_state.model.name}</span>',
                unsafe_allow_html=True
            )
            divider()

            epochs = st.slider("Epochs", 1, 20, 10)
            early_stopping = st.radio("Early Stopping",["Yes","No"])

            divider()

            st.subheader("Hyperparameters",help="Hyperparameters")
            st.selectbox("Choose a selection mode:",["Manual","Search"],
                                                           key="selection_mode",
                                                           disabled=st.session_state.model_locked)

            if st.session_state.selection_mode == "Manual":

                with st.container(height=410):

                    col1, col2 = st.columns([1,1])

                    with col1:
                        lr = st.selectbox("Learning Rate:", [0.0001, 0.001, 0.01, 0.1],
                                          disabled=st.session_state.model_locked)
                        batch_size = st.selectbox("Batch Size:", [8, 32, 64, 128],
                                                  disabled=st.session_state.model_locked)
                        batch_norm = st.selectbox("Batch Normalisation:", ["Yes", "No"],
                                                  disabled=st.session_state.model_locked)
                        conv_layers = st.selectbox("Convolutional Layers:", [1, 2, 3, 4],
                                                   disabled=st.session_state.model_locked)
                        dense_units = st.selectbox("Dense Units:", [32, 64, 128],
                                                   disabled=st.session_state.model_locked)
                    with col2:
                        filters = st.selectbox("Filters:", [16, 32, 64],
                                               disabled=st.session_state.model_locked)
                        dropout = st.selectbox("Dropout Rate:", [0.1, 0.2, 0.3, 0.4, 0.5],
                                               disabled=st.session_state.model_locked)

            else:

                protocol = st.radio("Choose a :primary[searching protocol]:",
                                          ["Grid","Random","Bayesian"],
                                          disabled=st.session_state.model_locked)

                with st.container(height=310):

                    lr_list = st.multiselect("Learning Rates:",
                                   [0.0001,0.001,0.01,0.1],
                                   default=[0.0001,0.001,0.01],
                                   disabled=st.session_state.model_locked)
                    batch_list = st.multiselect("Batch Sizes:",
                                   [8,32,64,128],
                                   default=[32,64],
                                   disabled=st.session_state.model_locked)
                    batch_norm_list = st.multiselect("Batch Normalisation:",
                                   ["Yes", "No"],
                                   default=["Yes", "No"],
                                   disabled=st.session_state.model_locked)
                    conv_layers_list = st.multiselect("Convolutional Layers:",
                                   [1,2,3,4,5],
                                   default=[2,3],
                                   disabled=st.session_state.model_locked)
                    dense_list = st.multiselect("Dense Units:",
                                   [32,64,128],
                                   default=[32,64,128],
                                   disabled=st.session_state.model_locked)
                    filters_list = st.multiselect("Filters:",
                                   [16,32,64],
                                   default=[16,32,64],
                                   disabled=st.session_state.model_locked)
                    dropout_list = st.multiselect("Dropout Rate:",
                                   [0.1, 0.2, 0.3, 0.4, 0.5],
                                   default=[0.2,0.3,0.4],
                                   disabled=st.session_state.model_locked)

    if train and len(st.session_state.samples) > 20:

        X = np.array(st.session_state.samples)[..., np.newaxis]
        y = st.session_state.labels_mapped

        if st.session_state.model_locked == False and st.session_state.selection_mode == "Manual":
            st.session_state.model = build_model(model_name="SCIPL",
                                num_classes=num_classes,
                                input_shape=(28,28,1),
                                conv_layers=conv_layers,
                                base_filters=filters,
                                dense_units=dense_units,
                                batch_norm=batch_norm,
                                dropout=dropout,)
            st.session_state.model = compile_model(st.session_state.model)

        elif st.session_state.model_locked == False and st.session_state.selection_mode == "Search":

            search_space = {
                "lr": lr_list,
                "batch_size": batch_list,
                "batch_norm": [x == "Yes" for x in batch_norm_list],
                "conv_layers": conv_layers_list,
                "dense_units": dense_list,
                "base_filters": filters_list,
                "dropout": dropout_list
            }

            if protocol == "Grid":
                sampler = optuna.samplers.GridSampler(search_space)
            elif protocol == "Random":
                sampler = optuna.samplers.RandomSampler(search_space)
            elif protocol == "Bayesian":
                sampler = optuna.samplers.TPESampler(search_space)

            st.session_state.search_params = hyperparameter_search(X=X,
                                                                   y=y,
                                                                   num_classes=num_classes,
                                                                   search_space=search_space,
                                                                   trials=search_trials,
                                                                   sampler=sampler,
                                                                   callbacks=[LiveSearchCallback()]


            )
            st.session_state.model = build_model(
                model_name="SCIPL",
                num_classes=num_classes,
                conv_layers=st.session_state.search_params["conv_layers"],
                dense_units=st.session_state.search_params["dense_units"],
                base_filters=st.session_state.search_params["base_filters"],
                dropout=st.session_state.search_params["dropout"],
            )
            st.session_state.model = compile_model(st.session_state.model, lr=st.session_state.search_params["lr"])
            st.session_state.model_searched = True
            st.session_state.model_locked = True
            st.rerun()

        st.session_state.train_cycles += 1
        st.session_state.model_searched = False
        st.session_state.model_locked = True

        if st.session_state.selection_mode == "Search":
            batch_size = st.session_state.search_params["batch_size"]

        train_model(model=st.session_state.model,
                    X=X,
                    y=y,
                    epochs=epochs,
                    batch_size=batch_size,
                    live_plot=[LivePlotCallback()],
                    early_stopping=early_stopping,
                    patience=patience,
                    min_delta=min_delta)

        st.rerun()
