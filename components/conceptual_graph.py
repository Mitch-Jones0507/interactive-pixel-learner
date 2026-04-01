import tensorflow as tf
import plotly.graph_objects as go
import streamlit as st

class ConceptualGraph:
    def __init__(self, model):
        self.model = model
        self.attributes = []

    def extract_model_attributes(self):
        for layer in self.model.layers:
            layer_info = {
                "Layer Name": layer.name,
                "Layer Type": layer.__class__.__name__,
                "Input Shape": getattr(layer, "input_shape", None),
                "Output Shape": getattr(layer, "output_shape", None),
                "Trainable": layer.trainable,
                "Params": layer.count_params(),
                "Activation": getattr(layer, "activation", None).__name__ if hasattr(layer, "activation") else None
            }
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_info.update({
                    "Filters": layer.filters,
                    "Kernel Size": layer.kernel_size,
                    "Strides": layer.strides,
                    "Padding": layer.padding,
                    "Dilation Rate": layer.dilation_rate
                })
            elif isinstance(layer, tf.keras.layers.Dense):
                layer_info.update({
                    "Units": layer.units,
                    "Use Bias": layer.use_bias
                })
            elif isinstance(layer, (tf.keras.layers.MaxPooling2D, tf.keras.layers.AveragePooling2D)):
                layer_info.update({
                    "Pool Size": layer.pool_size,
                    "Strides": layer.strides,
                    "Padding": layer.padding
                })
            elif isinstance(layer, tf.keras.layers.Dropout):
                layer_info.update({
                    "Dropout Rate": layer.rate
                })
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer_info.update({
                    "Momentum": layer.momentum,
                    "Epsilon": layer.epsilon
                })
            layer_info["Weights Shapes"] = [w.shape for w in layer.get_weights()]

            self.attributes.append(layer_info)
        return self.attributes

    def plot_conceptual_graph(self):
        if not self.attributes:
            self.extract_model_attributes()

        fig = go.Figure()

        y_gap_layer = 0.5  # gap between layers
        y_gap_attr = 0.3  # gap between attributes inside box
        x0, x1 = 0.05, 0.95  # horizontal limits
        x_center = (x0 + x1) / 2

        y_pos = -1  # starting y-coordinate

        layers_to_plot = [{"Layer Name": f"Input Layer ({len(st.session_state.samples)} samples)", "Layer Type": ""}] + self.attributes

        for layer in reversed(layers_to_plot):
            if layer["Layer Type"]:
                text_lines = [f"<b>{layer['Layer Name']} ({layer['Layer Type']})</b>"]
            else:
                text_lines = [f"<b>{layer['Layer Name']}</b>"]
            for k, v in layer.items():
                if k not in ["Layer Name", "Layer Type"]:
                    text_lines.append(f"{k}: {v}")

            box_height = y_gap_attr * len(text_lines) + y_gap_layer
            y1 = y_pos + 0.1
            y0 = y_pos - (box_height - 0.1)

            # Draw box
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                line=dict(color="black", width=0),
                fillcolor="#1f1f1f",
                layer="below"
            )

            # Add text
            y_text = y1 - y_gap_attr
            for line in reversed(text_lines):
                fig.add_trace(go.Scatter(
                    x=[x_center],
                    y=[y_text],
                    text=[line],
                    mode="text",
                    hoverinfo='none',
                    showlegend=False
                ))
                y_text -= y_gap_attr

            y_pos = y0 - y_gap_layer

        fig.update_layout(
            dragmode=False,
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange='reversed'),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
            height=50 * (-y_pos),
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": False,
                "displayModeBar": True,
                "modeBarButtonsToRemove": [
                    "zoom2d", "pan2d", "select2d", "lasso2d",
                    "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
                    "hoverCompareCartesian", "hoverClosestCartesian", "toImage",
                    "toggleSpikelines"
                ],
                "displaylogo": False
            }
        )