
from dash import html, Output, Input, State, callback_context
from dash import dcc
import dash_bootstrap_components as dbc
import dash_trich_components as dt
import dash





app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.MINTY, dbc.icons.BOOTSTRAP,
                                      dbc.icons.FONT_AWESOME
                                      ]
                )


def create_upload_button():
    upload_button = dcc.Upload(
            id='upload-data',
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
                                    dbc.Col(children=[html.Div(id="main_page_content")
                                                      ]
                                            )
                                ],
                            style={"height": "100%"}
                            ),
    #dt.SideBar([dt.SideBarItem(dcc.Upload(html.Button('Upload Zip File of images')))]),
    #html.Div(id="main_page_content")
])

app.layout = main_page  #html.Div("Image clustering app")


if __name__ == "__main__":
    app.run(port=8010, debug=True, use_reloader=True)

