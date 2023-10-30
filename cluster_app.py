
from dash import dcc
import dash_bootstrap_components as dbc
import dash_trich_components as dt
import dash
from clustimage import Clustimage
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import lru_cache
from typing import Union, List, NamedTuple
import abc
import cv2
from zipfile import ZipFile
from glob import glob
from dash import Dash, dcc, html, Input, Output, State, callback, callback_context
import os
import datetime
from PIL import Image



app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.MINTY, dbc.icons.BOOTSTRAP,
                                      dbc.icons.FONT_AWESOME
                                      ]
                )


def create_upload_button():
    upload_button = dcc.Upload(
            id='upload-image',
            children=html.Div([
                html.P('Select image zip file')
            ]),
            style={
                'width': '100%',
                'height': '50px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        )
    return upload_button

img_page = dbc.Row([
                    dbc.Col([html.Div(id='output-image-upload'),
                             html.Br(),
                             html.Img(id='bar-graph-matplotlib')
                            ], 
                            width=12
                            )
                ])



    # dbc.Row([
    #     dbc.Col([
    #         dcc.Graph(id='bar-graph-plotly', figure={})
    #     ], width=12, md=6),
    #     dbc.Col([
    #         dag.AgGrid(
    #             id='grid',
    #             rowData=df.to_dict("records"),
    #             columnDefs=[{"field": i} for i in df.columns],
    #             columnSize="sizeToFit",
    #         )
    #     ], width=12, md=6),
    # ], className='mt-4')


main_page = html.Div([
                    dbc.Row(children=[
                                    dbc.Col(width="auto",
                                            children=[html.Div("Sidebar content"),
                                                      create_upload_button(), #dcc.Upload()
                                                      dbc.Button("Cluster image", id="id_cluster_img", size="md", color="dark"),
                                                      html.Br(), html.Br(),
                                                      dbc.Button("Split Data", id="id_split_data", color="dark", size="md"),
                                                     # html.Button( ),
                                                    ],
                                            style={"backgroundColor": '#5ebbcb',
                                                   "height": "100em"
                                                   }
                                            ),
                                    dbc.Col(children=[img_page#html.Div(id="main_page_content")
                                                      ]
                                            )
                                ],
                            style={"height": "100%"}
                            ),
    #dt.SideBar([dt.SideBarItem(dcc.Upload(html.Button('Upload Zip File of images')))]),
    #html.Div(id="main_page_content")
])

app.layout = main_page  #html.Div("Image clustering app")

@callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
             # State('upload-image', 'last_modified')
              )
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        #contents_test = "valid.zip"
        folder_name = list_of_names[0].split(".")[0]
        with ZipFile(list_of_names[0], "r") as file:
                extract_folder = "extract_folder"
                file.extractall(extract_folder)
                
        img_folder = os.path.join(extract_folder, folder_name)
                
        img = glob(f"{img_folder}/*.jpg")[0]
        pil_image = Image.open(img)
        children = [html.H5(list_of_names[0]),
                    html.Br(),
                   html.Img(src=pil_image)
                    ]
        return children

if __name__ == "__main__":
    app.run(port=8010, debug=True, use_reloader=True)

