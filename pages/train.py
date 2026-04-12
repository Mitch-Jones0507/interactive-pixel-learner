import streamlit as st
from components.divider import divider
from components.charts import search_chart, acc_loss_chart
from components.cards import render_cards
from components.conceptual_graph import ConceptualGraph
from scripts.modelling import build_model, compile_model, prepare_data, train_model, hyperparameter_search
import numpy as np
import tensorflow as tf
import pandas as pd
import optuna

colour_map = {"SCIPL": "#b0ff9e", "CHIPL": "#9ecdff", "TRIPLite": "#ffc4f5", "TRIPL": "#ff9eef"}

class LivePlotCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k in st.session_state.history_data:
            st.session_state.history_data[k].append(logs.get(k))
        df = pd.DataFrame(st.session_state.history_data)
        acc_loss_chart(type="loss",model=st.session_state.model_name)
        acc_loss_chart(type="accuracy",model=st.session_state.model_name)

class LiveSearchCallback:
    def __call__(self, study, trial):
        st.session_state.search_history["trial"].append(trial.number)
        st.session_state.search_history["value"].append(trial.value)
        st.session_state.search_history["params"].append(trial.params)
        search_chart(st.session_state.model_name)

def page():

    st.sidebar.header(f"Models")
    divider(sidebar=True)
    st.sidebar.header(f"Tuning")
    st.sidebar.subheader("Hyperparameters")

    with st.sidebar.expander("Training",expanded=True):
        patience = st.slider("Patience", 1, 20, value=5, step=1)
        min_delta = st.slider("Min Delta", 0.01, 1.0, value=0.1, step=0.1)

    with st.sidebar.expander("Search",expanded=True):
        search_trials = st.slider("Search Trials", 1, 20, value=10,step=1)
        search_epochs = st.slider("Search Epochs", 5, 20, value=10, step=1)
        if search_epochs > 10:
            st.warning("More epochs will greatly increase search time.")
        search_early_stopping = st.radio("Search Early Stopping", ["Yes", "No"])
        search_patience = st.slider("Search Patience", 1, 20, value=5, step=1)
        search_min_delta = st.slider("Search Min Delta", 0.01, 1.0, value=0.1, step=0.1)

    divider(sidebar=True)

    st.title("Train", text_alignment="center")
    st.markdown(
        "Choose a model to train your _IP_:primary[L]. View and control your tuning  \n"
        "parameters and click train when you are ready.",
        text_alignment="center"
    )
    divider()
    model_column, tuning_column, visualisation_column = st.columns([1,1,1])

    with model_column:

        unique_labels = sorted(set(st.session_state.labels))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        st.session_state.labels_mapped = np.array([label_map[l] for l in st.session_state.labels])
        num_classes = len(unique_labels)

        st.header("Models")

        top = st.container(height=500)
        bottom = st.container(border=True)

        if "model" not in st.session_state or st.session_state.model is None:
            st.session_state.model = build_model("SCIPL", num_classes=num_classes)
            st.session_state.model = compile_model(st.session_state.model)
            for k in st.session_state.history_data:
                st.session_state.history_data[k] = []

        with bottom:
            model_options = ["SCIPL", "CHIPL", "TRIPLite", "TRIPL"]

            st.session_state.model_name = st.selectbox(
                "Select an _IP_:primary[L] :primary[model]:",
                model_options,
                index=model_options.index(st.session_state.model_name),
                disabled=st.session_state.model_locked or st.session_state.model_searched,
            )
            st.segmented_control("Select model type:", ["Blank","Pretrained"], key="model_type",
                     disabled=st.session_state.model_locked or st.session_state.model_searched,
                        default="Blank")

        with top:
            st.markdown(
                f'**Loaded Model**: {"Searched " if st.session_state.model_searched else "Trained " if st.session_state.model_locked else "Blank " if st.session_state.model_type == "Blank" else "Pretrained "}'
                f'<span style="color: {colour_map[st.session_state.model_name]};">'
                f'{st.session_state.model_name}</span>',
                unsafe_allow_html=True
            )

            divider()

            st.subheader("Conceptual Framework:")

            if st.session_state.model and st.session_state.model_locked and not st.session_state.model_searched:
                st.markdown("Conceptual graph coming soon.")
            else:
                """
                framework_sample = random.choice(st.session_state.samples)
                framework_sample_json = json.dumps(framework_sample.flatten().tolist())

                with open("components/conceptual.html", "r") as f:
                    html_content = f.read()
                html_content = html_content.replace("SAMPLE_PLACEHOLDER", framework_sample_json)
                components.html(html_content, height=375)"""
                st.markdown("No model trained yet.")

        train = st.button(f"Search {st.session_state.model_name}" if st.session_state.selection_mode == "Search" and st.session_state.model_searched == False
                          else f"Train {st.session_state.model_name}" if st.session_state.train_cycles == 0
                            else f"Retrain {st.session_state.model_name}"
                          ,use_container_width=True,
                          disabled=len(st.session_state.samples)<20)

        if len(st.session_state.samples) <= 20:
            st.warning(f"Capture more than 20 samples to train {st.session_state.model_name} ({len(st.session_state.samples)} samples).")

        if st.session_state.model_locked:
            col1, col2 = st.columns(2)
            with col1:

                delete_model = st.button(f"Delete {st.session_state.model_name}", use_container_width=True)

                if delete_model:

                    @st.dialog(f"Are you sure you want to delete {st.session_state.model_name}?")
                    def delete_model():
                        st.write("This model will not be recoverable.")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Yes",use_container_width=True):
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
                            if st.button("Cancel",use_container_width=True):
                                st.rerun()

                    delete_model()

            with col2:
                download_model = st.button(f"Download {st.session_state.model_name}",use_container_width=True)

    with tuning_column:
        st.header("Tuning")
        with st.container(height=855):
            st.markdown(
                f'**Loaded Model**: {"Searched " if st.session_state.model_searched else "Trained " if st.session_state.model_locked else "Blank " if st.session_state.model_type == "Blank" else "Pretrained "}'
                f'<span style="color: {colour_map[st.session_state.model_name]};">'
                f'{st.session_state.model_name}</span>',
                unsafe_allow_html=True
            )
            divider()

            epochs = st.slider("Training Epochs:", 1, 20, 10)
            early_stopping = st.toggle("Early Stopping:")

            divider()

            st.subheader("Hyperparameters",help="Hyperparameters")
            view_tab, control_tab = st.tabs(["View", "Control"])

            if st.session_state.model_name != "SCIPL":

                if st.session_state.model_name == "TRIPL":
                    fusion_mode = control_tab.segmented_control("Fusion Mode",["Hybrid","Ensemble"],default="Hybrid")

                control_tab.segmented_control("Choose a selection mode:",["Manual","Search"],
                                                               key="selection_mode",
                                                                default = "Manual",
                                                               disabled=st.session_state.model_locked or st.session_state.model_searched)

                if st.session_state.selection_mode == "Manual":

                    schema = {}

                    with control_tab.container(height=490):

                            st.subheader("Optimisation")

                            st.markdown("**Learning Rate**:")
                            lr = st.number_input("", min_value=0.0001, max_value=0.1, value=0.001,
                                                 label_visibility="collapsed",
                                                 disabled=st.session_state.model_locked or st.session_state.model_searched)
                            schema["learning_rate"] = lr

                            st.markdown("**Batch Size**:")
                            batch_size = st.selectbox("", [8, 32, 64, 128], index=1,
                                                      label_visibility="collapsed",
                                                      disabled=st.session_state.model_locked or st.session_state.model_searched)
                            schema["batch_size"] = batch_size

                            st.markdown("**Batch Normalisation**:")
                            batch_norm = st.selectbox(":", ["Yes", "No"], index=1,
                                                      label_visibility="collapsed",
                                                      disabled=st.session_state.model_locked or st.session_state.model_searched)
                            schema["batch_norm"] = batch_norm

                            if st.session_state.model_name != "TRIPLite":

                                st.markdown("**Optimiser**:")
                                optimiser = st.selectbox(":", ["Adam", "SGD"], index=0,
                                                         label_visibility="collapsed",
                                                         disabled=st.session_state.model_locked or st.session_state.model_searched)
                                schema["optimiser"] = optimiser

                                st.subheader("Architecture")

                                st.markdown("**Convolutional Layers**:")
                                conv_layers = st.selectbox("", [1, 2, 3, 4], index = 1,
                                                           label_visibility="collapsed",
                                                           disabled=st.session_state.model_locked or st.session_state.model_searched)
                                schema["conv_layers"] = conv_layers

                                st.markdown("**Dense Units**:")
                                dense_units = st.selectbox("", [32, 64, 128], index = 0,
                                                           label_visibility="collapsed",
                                                           disabled=st.session_state.model_locked or st.session_state.model_searched)
                                schema["dense_units"] = dense_units

                                st.markdown("**Filters**:")
                                filters = st.selectbox("", [8, 16, 32, 64], index = 0,
                                                       label_visibility="collapsed",
                                                       disabled=st.session_state.model_locked or st.session_state.model_searched)
                                schema["filters"] = filters

                                st.markdown("**Kernel Size**:")
                                kernel_size = st.selectbox("", [3, 5, 7], index = 1,
                                                           label_visibility="collapsed",
                                                           disabled=st.session_state.model_locked or st.session_state.model_searched)
                                schema["kernel_size"] = kernel_size

                            st.subheader("Regularisation")

                            st.markdown("**Dropout Rate**:")
                            dropout = st.number_input("", min_value=0.1, max_value=1.0, value=0.3,
                                                 label_visibility="collapsed",
                                                 disabled=st.session_state.model_locked or st.session_state.model_searched)
                            schema["dropout"] = dropout

                            st.markdown("**Weight Decay**:")
                            weight_decay = st.number_input("", min_value=0.0000001, max_value=0.01, value=0.0001,
                                                           label_visibility="collapsed",
                                                           disabled=st.session_state.model_locked or st.session_state.model_searched)
                            schema["weight_decay"] = weight_decay

                            if st.session_state.model_name == "TRIPLite" or "TRIPL":

                                pass

                            if st.session_state.model_name == "TRIPL":

                                st.subheader("Fusion")

                                schema["fusion_mode"] = fusion_mode

                                if fusion_mode == "Hybrid":
                                    st.markdown("**Fusion Point**:")
                                    fusion_point = st.selectbox("", ["Early", "Mid", "Late"],
                                                                label_visibility="collapsed",
                                                                disabled=st.session_state.model_locked or st.session_state.model_searched)
                                    schema["fusion_point"] = fusion_point

                                else:
                                    st.markdown("**Merge Strategy**:")
                                    merge_strategy = st.selectbox("", ["Average", "Concatenate", "Weight"],
                                                                  label_visibility="collapsed",
                                                                  disabled=st.session_state.model_locked or st.session_state.model_searched)
                                    schema["merge_strategy"] = merge_strategy

                else:

                    schema = {}

                    protocol = control_tab.radio("Choose a :primary[searching protocol]:",
                                              ["Grid","Random","Bayesian"],
                                              disabled=st.session_state.model_locked or st.session_state.model_searched)

                    with control_tab.container(height=380):

                        if st.session_state.model_name == "CHIPL":

                            st.subheader("Optimisation")

                            st.markdown("**Learning Rate**:")
                            lr_lower_col, lr_upper_col = st.columns([1,1])
                            with lr_lower_col:
                                lr_lower = st.number_input(
                                    "Lower Bound:",
                                    min_value=0.0001,
                                    max_value=0.01,
                                    value=0.0001,
                                    step=0.0001,
                                    format="%g",
                                    disabled=st.session_state.model_locked or st.session_state.model_searched)
                            with lr_upper_col:
                                lr_upper = st.number_input(
                                    "Upper Bound:",
                                    min_value=0.01,
                                    max_value=0.1,
                                    value=0.01,
                                    step=0.001,
                                    format="%g",
                                    disabled=st.session_state.model_locked or st.session_state.model_searched)
                            schema["learning_rate"] = [lr_lower, lr_upper]

                            st.markdown("**Batch Sizes**:")
                            batch_list = st.multiselect("",
                                           [8,32,64,128],
                                           default=[32,64],
                                           disabled=st.session_state.model_locked or st.session_state.model_searched,
                                            label_visibility="collapsed")
                            schema["batch_size"] = batch_list

                            st.markdown("**Batch Normalisation**:")
                            batch_norm_list = st.multiselect("Batch Normalisation:",
                                           ["Yes", "No"],
                                           default=["Yes", "No"],
                                           disabled=st.session_state.model_locked or st.session_state.model_searched,
                                            label_visibility="collapsed")
                            schema["batch_norm"] = batch_norm_list

                            st.markdown("**Optimiser**:")
                            optimiser_list = st.multiselect("",
                                                       ["Adam", "SGD"],
                                                       default=["Adam"],
                                                        disabled=st.session_state.model_locked or st.session_state.model_searched,
                                                       label_visibility="collapsed")
                            schema["optimiser"] = optimiser_list

                            st.subheader("Architecture")

                            st.markdown("**Convolutional Layers**:")
                            conv_layers_list = st.multiselect("",
                                           [1,2,3,4,5],
                                           default=[2,3],
                                           disabled=st.session_state.model_locked or st.session_state.model_searched,
                                            label_visibility="collapsed")
                            schema["conv_layers"] = conv_layers_list

                            st.markdown("**Dense Units**:")
                            dense_list = st.multiselect("",
                                           [32,64,128],
                                           default=[32,64,128],
                                           disabled=st.session_state.model_locked or st.session_state.model_searched,
                                            label_visibility="collapsed")
                            schema["dense_units"] = dense_list

                            st.markdown("**Filters**:")
                            filters_list = st.multiselect("",
                                           [16,32,64],
                                           default=[16,32,64],
                                           disabled=st.session_state.model_locked or st.session_state.model_searched,
                                            label_visibility="collapsed")
                            schema["filters"] = filters_list

                            st.markdown("**Kernel Size**:")
                            kernel_list = st.multiselect(
                                "",
                                [3, 5, 7],
                                default=[3],
                                disabled=st.session_state.model_locked or st.session_state.model_searched,
                                label_visibility="collapsed"
                            )
                            schema["kernel_size"] = kernel_list

                            st.subheader("Regularisation")

                            st.markdown("**Dropout Rate**:")
                            dr_lower_col, dr_upper_col = st.columns([1,1])

                            if "dr_lower" not in st.session_state:
                                st.session_state.dr_lower = 0.1
                            if "dr_upper" not in st.session_state:
                                st.session_state.dr_upper = 0.5

                            with dr_lower_col:
                                dr_lower = st.number_input(
                                    "Lower Bound:",
                                    min_value=0.0,
                                    max_value=st.session_state.dr_upper,
                                    value=st.session_state.dr_lower,
                                    step=0.01,
                                    format="%g",
                                    key="dr_lower",
                                    disabled=st.session_state.model_locked or st.session_state.model_searched
                                )
                            with dr_upper_col:
                                dr_upper = st.number_input(
                                    "Upper Bound:",
                                    min_value=st.session_state.dr_lower,
                                    max_value=1.0,
                                    value=st.session_state.dr_upper,
                                    step=0.01,
                                    format="%g",
                                    key="dr_upper",
                                    disabled=st.session_state.model_locked or st.session_state.model_searched
                                )
                            schema["dropout"] = [dr_lower, dr_upper]

                            st.markdown("**Weight Decay (L2)**:")
                            wd_lower_col, wd_upper_col = st.columns([1, 1])

                            if "wd_lower" not in st.session_state:
                                st.session_state.wd_lower = 1e-6
                            if "wd_upper" not in st.session_state:
                                st.session_state.wd_upper = 1e-3

                            with wd_lower_col:
                                wd_lower = st.number_input(
                                    "Lower Bound:",
                                    min_value=1e-7,
                                    max_value=st.session_state.wd_upper,
                                    value=st.session_state.wd_lower,
                                    format="%g",
                                    key="wd_lower",
                                    disabled=st.session_state.model_locked or st.session_state.model_searched
                                )

                            with wd_upper_col:
                                wd_upper = st.number_input(
                                    "Upper Bound:",
                                    min_value=st.session_state.wd_lower,
                                    max_value=1e-1,
                                    value=st.session_state.wd_upper,
                                    format="%g",
                                    key="wd_upper",
                                    disabled=st.session_state.model_locked or st.session_state.model_searched
                                )
                                schema["weight_decay"] = [wd_lower, wd_upper]
            else:

                schema = {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "batch_norm": "No",
                    "optimiser": "Adam",
                    "conv_layers": 2,
                    "dense_units": 32,
                    "filters": 8,
                    "kernel_size": 3,
                    "dropout": 0.3,
                    "weight_decay": 0.0001,
                }

                with control_tab.container(border=True):
                    st.markdown(f"{st.session_state.model_name} can not be hyperparametrically controlled.")

            with view_tab.container(height=540, border=True):
                if st.session_state.model_searched == False:
                    render_cards(hparams=schema)
                else:
                    render_cards(hparams=st.session_state.search_params)

    with visualisation_column:
        st.header(f"Visualisation")

        if st.session_state.model_name != "SCIPL":
            with st.container(height=275,border=True):
                st.subheader(f"**Search History**:")
                if "search_chart" not in st.session_state:
                    st.session_state.search_chart = st.empty()
                if "search_history" in st.session_state and st.session_state.search_history["trial"]:
                    search_chart(model=st.session_state.model_name)
                else:
                    st.write("No hyperparameter search done.")
                    st.session_state.search_chart = st.empty()
        with st.container(height=490,border=True):
            st.subheader(f"**Train History ({st.session_state.train_cycles} Cycles):**")
            if st.session_state.history_data["loss"]:
                acc_loss_chart(type="loss",model=st.session_state.model_name)
                acc_loss_chart(type="accuracy",model=st.session_state.model_name)
            else:
                st.write("No model trained yet.")
                st.session_state.loss_chart = st.empty()
                st.session_state.acc_chart = st.empty()

    if train and len(st.session_state.samples) > 20:

        if st.session_state.model_locked == False and st.session_state.model_searched == False:
            X = np.array(st.session_state.samples)[..., np.newaxis]
            y = st.session_state.labels_mapped
            st.session_state.split_data["X_train"], st.session_state.split_data["y_train"], st.session_state.split_data["X_test"], st.session_state.split_data["y_test"] = prepare_data(X,y)

        if st.session_state.model_locked == False and st.session_state.model_searched == False and st.session_state.selection_mode == "Manual":
            st.session_state.model = build_model(model_name=st.session_state.model_name,
                                num_classes=num_classes,
                                input_shape=(28,28,1),
                                conv_layers=schema["conv_layers"],
                                base_filters=schema["filters"],
                                dense_units=schema["dense_units"],
                                kernel_size=schema["kernel_size"],
                                batch_norm=schema["batch_norm"],
                                dropout=schema["dropout"],
                                weight_decay=schema["weight_decay"])

            st.session_state.model = compile_model(st.session_state.model,
                                                   lr=schema["learning_rate"],
                                                   optimiser_name=schema["optimiser"])

        elif st.session_state.model_locked == False and st.session_state.model_searched == False and st.session_state.selection_mode == "Search":

            search_space = {
                "learning_rate": {"low": lr_lower, "high": lr_upper, "log": True},
                "batch_size": batch_list,
                "batch_norm": [x == "Yes" for x in batch_norm_list],
                "optimiser": optimiser_list,
                "conv_layers": conv_layers_list,
                "dense_units": dense_list,
                "base_filters": filters_list,
                "kernel_size": kernel_list,
                "dropout_rate": {"low": dr_lower, "high": dr_upper, "log": True},
                "weight_decay": {"low": wd_lower, "high": wd_upper, "log": True},
            }

            if protocol == "Grid":

                grid_dict = {
                    "learning_rate": np.logspace(np.log10(lr_lower), np.log10(lr_upper), num=5).tolist(),
                    "dropout_rate": np.linspace(dr_lower, dr_upper, num=5).tolist(),
                    "weight_decay": np.logspace(np.log10(wd_lower), np.log10(wd_upper), num=5).tolist(),
                    "batch_size": batch_list,
                    "batch_norm": [x == "Yes" for x in batch_norm_list],
                    "optimiser": optimiser_list,
                    "conv_layers": conv_layers_list,
                    "dense_units": dense_list,
                    "base_filters": filters_list,
                    "kernel_size": kernel_list,
                }
                sampler = optuna.samplers.GridSampler(grid_dict)
            elif protocol == "Random":
                sampler = optuna.samplers.RandomSampler()
            elif protocol == "Bayesian":
                sampler = optuna.samplers.TPESampler()

            st.session_state.search_params = hyperparameter_search(X_train=st.session_state.split_data["X_train"],
                                                                   y_train=st.session_state.split_data["y_train"],
                                                                   X_test=st.session_state.split_data["X_test"],
                                                                   y_test=st.session_state.split_data["y_test"],
                                                                   num_classes=num_classes,
                                                                   search_space=search_space,
                                                                   trials=search_trials,
                                                                   epochs=search_epochs,
                                                                   sampler=sampler,
                                                                   callbacks=[LiveSearchCallback()],
                                                                   early_stopping=search_early_stopping,
                                                                   patience=search_patience,
                                                                   min_delta=search_min_delta,


            )

            st.session_state.model = build_model(
                model_name=st.session_state.model_name,
                num_classes=num_classes,
                conv_layers=st.session_state.search_params["conv_layers"],
                dense_units=st.session_state.search_params["dense_units"],
                base_filters=st.session_state.search_params["base_filters"],
                kernel_size=st.session_state.search_params["kernel_size"],
                dropout=st.session_state.search_params["dropout_rate"],
                batch_norm=st.session_state.search_params["batch_norm"],
                weight_decay=st.session_state.search_params["weight_decay"]
            )

            st.session_state.model = compile_model(st.session_state.model,
                                                   lr=st.session_state.search_params["learning_rate"],
                                                   optimiser_name=st.session_state.search_params["optimiser"])

            st.session_state.model_searched = True
            st.rerun()

        st.session_state.train_cycles += 1
        st.session_state.model_searched = False
        st.session_state.model_locked = True

        if st.session_state.selection_mode == "Search":
            batch_size = st.session_state.search_params["batch_size"]
        else:
            batch_size = schema["batch_size"]

        train_model(model=st.session_state.model,
                    X_train=st.session_state.split_data["X_train"],
                    y_train=st.session_state.split_data["y_train"],
                    X_test=st.session_state.split_data["X_test"],
                    y_test=st.session_state.split_data["y_test"],
                    epochs=epochs,
                    batch_size=batch_size,
                    live_plot=[LivePlotCallback()],
                    early_stopping=early_stopping,
                    patience=patience,
                    min_delta=min_delta)

        st.rerun()

    divider()
