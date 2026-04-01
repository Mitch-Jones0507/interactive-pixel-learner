import streamlit as st
import plotly.express as px
import pandas as pd

def search_chart():

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
        color_discrete_sequence=["#b0ff9e"]
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

def acc_loss_chart(type):

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