import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import optuna
from models.scipl import build_scipl
from models.chipl import build_chipl

def build_model(model_name, num_classes, input_shape=(28,28,1),
                conv_layers=3, base_filters=64, dense_units=64,
                dropout=0.1, batch_norm=False):
    if model_name in ["SCIPL", "TRIPL"]:
        return build_scipl(num_classes, input_shape, conv_layers,
                           base_filters, dense_units, dropout, batch_norm)
    elif model_name == "CHIPL":
        return build_chipl(num_classes)

def compile_model(model, lr=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(model, X, y, epochs=10, batch_size=32, live_plot=None,early_stopping=None,patience=5,min_delta=0.01):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
    )

    callbacks = []
    if live_plot is not None:
        callbacks.extend(live_plot)
    if early_stopping == "Yes":
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

def hyperparameter_search(X, y, num_classes, search_space, trials=10, sampler=None, callbacks=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = {
            "lr": trial.suggest_categorical("lr", search_space["lr"]),
            "batch_size": trial.suggest_categorical("batch_size", search_space["batch_size"]),
            "batch_norm": trial.suggest_categorical("batch_norm", search_space["batch_norm"]),
            "conv_layers": trial.suggest_categorical("conv_layers", search_space["conv_layers"]),
            "dense_units": trial.suggest_categorical("dense_units", search_space["dense_units"]),
            "base_filters": trial.suggest_categorical("base_filters", search_space["base_filters"]),
            "dropout": trial.suggest_categorical("dropout", search_space["dropout"])
        }

        model = build_model(
            model_name="SCIPL",
            num_classes=num_classes,
            input_shape=(28,28,1),
            conv_layers=params["conv_layers"],
            base_filters=params["base_filters"],
            dense_units=params["dense_units"],
            dropout=params["dropout"],
            batch_norm=params["batch_norm"]
        )

        model = compile_model(model, lr=params["lr"])

        history = train_model(model=model,
                              X=X,
                              y=y,
                              epochs=5,
                              batch_size=params["batch_size"],
                              )[1]

        return max(history.history["val_accuracy"])

    study.optimize(objective, n_trials=trials, callbacks=callbacks)
    return study.best_params