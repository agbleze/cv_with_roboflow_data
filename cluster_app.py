
from dash import html, Output, Input, State, callback_context
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_trich_components as dt
import dash





app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.MINTY, dbc.icons.BOOTSTRAP,
                                      dbc.icons.FONT_AWESOME
                                      ]
                )


app.layout = html.Div()


if __name__ == "__main__":
    app.run(port=8010, debug=True, use_reloader=True)

