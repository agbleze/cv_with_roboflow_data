# -*- coding: utf-8 -*-
"""Dash MNIST t-SNE

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nSGnnlt_dwNFDNxCMIuDxk-AsujIdncP
"""

#!pip install --q dash==2.0.0 jupyter-dash==0.4.0;

from keras.datasets import mnist
import dash

X, y = mnist.load_data()[0]

images = []
labels = []

# create 100 of each digit
for num in range(10):
  imgs = []
  labels_inner = []
  for img, label in zip(X, y):
    if len(imgs) >= 100: break
    if label == num:
      imgs.append(img)
      labels_inner.append(num)
  images += imgs
  labels += labels_inner

len(images)
len(labels)

import io
import base64
import pickle
import gzip

import numpy as np

#from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go

from PIL import Image

from sklearn.manifold import TSNE

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

# Flatten image matrices from (28,28) to (784,)
flattenend_images = [i.flatten() for i in images]

# t-SNE Outputs a 3 dimensional point for each image
tsne = TSNE(
    random_state = 123,
    n_components=3,
    verbose=0,
    perplexity=40,
    n_iter=300) \
    .fit_transform(flattenend_images)

# Color for each digit
color_map = {
    0: "#E52B50",
    1: "#9F2B68",
    2: "#3B7A57",
    3: "#3DDC84",
    4: "#FFBF00",
    5: "#915C83",
    6: "#008000",
    7: "#7FFFD4",
    8: "#E9D66B",
    9: "#007FFF",
}
colors = [color_map[l] for l in labels]

fig = go.Figure(data=[go.Scatter3d(
    x=tsne[:, 0],
    y=tsne[:, 1],
    z=tsne[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=colors,
    )
)])

fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-10,10]),
        yaxis=dict(range=[-10,10]),
        zaxis=dict(range=[-10,10]),
    )
)

#app = JupyterDash(__name__)
app = dash.Dash(__name__)
app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
    ],
)

@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = images[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P("MNIST Digit " + str(labels[num]), style={'font-weight': 'bold'})
        ])
    ]

    return True, bbox, children

if __name__ == "__main__":
    app.run_server(port=8090, debug=False) #(mode='inline', debug=True)