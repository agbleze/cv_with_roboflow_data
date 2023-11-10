# #%%
# import dash
# #import dash_html_components as html
# from dash import html
# import dash_uploader as du

# app = dash.Dash(__name__)

# # 1) configure the upload folder
# du.configure_upload(app, folder="data", use_upload_id=False)

# # 2) Use the Upload component
# app.layout = html.Div([
#     du.Upload(),
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)
# # %%




from dash import Dash, dcc, html, Input, Output, State, callback

import datetime
from glob import glob
import os
import cv2
from zipfile import ZipFile
from PIL import Image


# filename = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/valid.zip"
# with zipfile.ZipFile(filename, "r") as zipfile:
#     zipfile.extractall()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def parse_contents(contents, filename):#, date):
    # import zipfile
    # from glob import glob
    # import os

    # filename = "/Users/lin/Documents/python_venvs/cv_with_roboflow_data/valid.zip"
    with zipfile.ZipFile(contents, "r") as zipfile:
        extract_folder = "extract_folder"
        zipfile.extractall(extract_folder)
        
    img_folder = os.path.join(extract_folder, filename)
        
    img = glob(f"{img_folder}/*.jpg")[0]

    return html.Div([
        html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.H5(img),
        #html.Img(img)
        # html.Hr(),
        # html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
    ])

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
        # children = [
        #     parse_contents(c, n) for c, n in
        #     zip(list_of_contents, list_of_names)]
        # return children

if __name__ == '__main__':
    app.run(debug=False)
    
    
    