import streamlit as st
import numpy as np
import tensorflow as tf
from scripts.preprocessing import process_image
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import optuna
import random
from models.scipl import build_scipl
from models.chipl import build_chipl

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def build_model(model_name, num_classes, input_shape=(28,28,1),
                conv_layers=2, base_filters=8, dense_units=32,
                kernel_size=3,dropout=0.3, batch_norm="Yes",
                weight_decay=0.0001):

    if model_name == "SCIPL":
        return build_scipl(num_classes)

    elif model_name == "CHIPL":
        return build_chipl(num_classes, input_shape, conv_layers,
                           base_filters, dense_units, kernel_size,
                           dropout, batch_norm, weight_decay)

    elif model_name == "TRIPL":
        return build_scipl(num_classes)

def compile_model(model, lr=0.001, optimiser_name="Adam"):
    if optimiser_name == "Adam":
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=lr,
        )
    elif optimiser_name == "SGD":
        optimiser = tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.9,
        )
    model.compile(
        optimizer=optimiser,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def prepare_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32,
                live_plot=None,early_stopping=False,patience=5,min_delta=0.01):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
    )

    callbacks = []
    if live_plot is not None:
        callbacks.extend(live_plot)
    if early_stopping == True:
        callbacks.append(early_stop)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

def hyperparameter_search(X_train, y_train, X_test, y_test, num_classes,
                          search_space, trials=10, epochs=10, sampler=None,
                          callbacks=None, early_stopping=None, patience=5,
                          min_delta=0.01):

    is_grid = isinstance(sampler, optuna.samplers.GridSampler)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):

        set_seed(42)

        params = {
            "batch_size": trial.suggest_categorical("batch_size", search_space["batch_size"]),
            "batch_norm": trial.suggest_categorical("batch_norm", search_space["batch_norm"]),
            "optimiser": trial.suggest_categorical("optimiser", search_space["optimiser"]),
            "conv_layers": trial.suggest_categorical("conv_layers", search_space["conv_layers"]),
            "dense_units": trial.suggest_categorical("dense_units", search_space["dense_units"]),
            "base_filters": trial.suggest_categorical("base_filters", search_space["base_filters"]),
            "kernel_size": trial.suggest_categorical("kernel_size", search_space["kernel_size"]),
        }

        if is_grid:
            lr_categories = np.logspace(
                np.log10(search_space["learning_rate"]["low"]),
                np.log10(search_space["learning_rate"]["high"]),
                num=5
            ).tolist()
            params["learning_rate"] = trial.suggest_categorical("learning_rate", lr_categories)

            dr_categories = np.linspace(
                search_space["dropout_rate"]["low"],
                search_space["dropout_rate"]["high"],
                num=5
            ).tolist()
            params["dropout_rate"] = trial.suggest_categorical("dropout_rate", dr_categories)

            wd_categories = np.logspace(
                np.log10(search_space["weight_decay"]["low"]),
                np.log10(search_space["weight_decay"]["high"]),
                num=5
            ).tolist()
            params["weight_decay"] = trial.suggest_categorical("weight_decay", wd_categories)
        else:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"]["low"],
                search_space["learning_rate"]["high"],
                log=search_space["learning_rate"]["log"]
            )
            params["dropout_rate"] = trial.suggest_float(
                "dropout_rate",
                search_space["dropout_rate"]["low"],
                search_space["dropout_rate"]["high"],
                log=False
            )
            params["weight_decay"] = trial.suggest_float(
                "weight_decay",
                search_space["weight_decay"]["low"],
                search_space["weight_decay"]["high"],
                log=search_space["weight_decay"]["log"]
            )

        model = build_model(
            model_name=st.session_state.model_name,
            num_classes=num_classes,
            input_shape=(28, 28, 1),
            conv_layers=params["conv_layers"],
            base_filters=params["base_filters"],
            dense_units=params["dense_units"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout_rate"],
            batch_norm=params["batch_norm"],
            weight_decay=params["weight_decay"],
        )

        model = compile_model(
            model,
            lr=params["learning_rate"],
            optimiser_name=params["optimiser"],
        )

        _, history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            batch_size=params["batch_size"],
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta
        )

        val_acc = history.history["val_accuracy"]
        return np.mean(val_acc[-3:])

    study.optimize(objective, n_trials=trials, callbacks=callbacks)
    return study.best_params


def predict_model(model, data, mode="batch"):

    if mode == "batch":
        X = np.array(data)

        if len(X.shape) == 3:
            X = X[..., np.newaxis]

    elif mode == "live":
        image = process_image(data)
        X = np.expand_dims(image, axis=0)

    else:
        raise ValueError("mode must be 'batch' or 'live'")

    pred_probs = model.predict(X)
    y_pred = np.argmax(pred_probs, axis=1)

    return y_pred, pred_probs