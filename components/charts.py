import streamlit as st
import plotly.express as px
import pandas as pd
import tensorflow as tf

def search_chart(model):

    history = st.session_state.search_history

    df = pd.DataFrame({
        "trial": history["trial"],
        "value": history["value"]
    })

    params_df = pd.DataFrame(history["params"])
    df = pd.concat([df, params_df], axis=1)

    fig = px.line(
        df,
        x="trial",
        y="value",
        markers=True,
        hover_data=df.columns,
        labels={
            "trial": "trial",
            "value": "val accuracy"
        },
        color_discrete_sequence=["#9ecdff"] if model == "CHIPL" else ["#ff9eef"]
    )
    fig.update_layout(
        margin=dict(t=0, b=30, l=40, r=10),
    )
    st.session_state.search_chart.plotly_chart(
        fig,
        use_container_width=True,
        height=200,
        config={"displayModeBar": False}
    )

def acc_loss_chart(type, model):

    history = st.session_state.history_data
    df = pd.DataFrame(history)

    if type == "loss":
        data = df[["loss", "val_loss"]].reset_index().melt(
            id_vars="index",
            var_name="metric",
            value_name="value"
        )
        x_title = "epoch"
        y_title = "loss"
    else:
        data = df[["accuracy", "val_accuracy"]].reset_index().melt(
            id_vars="index",
            var_name="metric",
            value_name="value"
        )
        x_title = "epoch"
        y_title = "accuracy"

    fig = px.line(
        data,
        x="index",
        y="value",
        color="metric",
        markers=False,
        color_discrete_map={
            "loss": "#b0ff9e",
            "val_loss": "#d8ffc9",
            "accuracy": "#b0ff9e",
            "val_accuracy": "#d8ffc9"
        } if model == "SCIPL" else {
            "loss": "#9ecdff",
            "val_loss": "#d1e7ff",
            "accuracy": "#9ecdff",
            "val_accuracy": "#d1e7ff"
        } if model == "CHIPL" else {
            "loss": "#ff9eef",
            "val_loss": "#fcc5f3",
            "accuracy": "#ff9eef",
            "val_accuracy": "#fcc5f3"
        }
    )

    fig.update_layout(
        margin=dict(t=0, b=30, l=40, r=10),
        showlegend=False,
        xaxis_title=x_title,
        yaxis_title=y_title
    )

    if type == "loss":
        st.session_state.loss_chart.plotly_chart(
            fig,
            use_container_width=True,
            height=200,
            config={"displayModeBar": False}
        )
    else:
        st.session_state.acc_chart.plotly_chart(
            fig,
            use_container_width=True,
            height=200,
            config={"displayModeBar": False}
        )

def confusion_matrix_chart(y_true, y_pred):

    cm = tf.math.confusion_matrix(
        labels=y_true,
        predictions=y_pred,
        num_classes=len(st.session_state.label_id)
    ).numpy()

    labels = list(st.session_state.label_id.keys())

    colorscale = [
        [0.0, "#2e2e33"],
        [0.5, "#6b6b52"],
        [1.0, "#fffc9e"]
    ]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale=colorscale
    )

    fig.update_layout(
        paper_bgcolor="#2a2a2e",
        plot_bgcolor="#2a2a2e",

        font=dict(color="#ffffff", family="Inter"),

        xaxis=dict(title="Predicted", showline=False),
        yaxis=dict(title="True", showline=False),

        margin=dict(l=10, r=10, t=0, b=10),
    )

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(constrain="domain")

    fig.update_coloraxes(showscale=False)

    fig.update_coloraxes(showscale=False)

    return fig