"""requirements.txt: dash, dash-bootstrap-components, plotly, numpy, scipy.

Pulsed FRC Digital Twin -- interactive control room for an idealized 0D pulse.
Run with ``python app.py`` and open http://127.0.0.1:8050.
"""
from __future__ import annotations

import numpy as np
import dash
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physics import PulseParameters, simulate_frc_pulse


CYAN, PANEL, PAPER = "#21d4fd", "#111827", "#080c16"
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="FRC Digital Twin")
server = app.server
app.index_string = """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
body { background:#080c16; } .app-shell { max-width:1600px; padding:24px; }
.title { color:#21d4fd; letter-spacing:.13em; font-weight:800; text-shadow:0 0 18px #0b7691; }
.subtitle { color:#8394aa; letter-spacing:.04em; margin-bottom:22px; }
.panel { background:#111827; border:1px solid #26374e; box-shadow:0 0 18px #07131f; }
.panel-title,.control-label { color:#9eeeff; letter-spacing:.08em; font-size:.78rem; font-weight:700; }
.control-label { display:block; margin-bottom:8px; } .status { padding:14px; font-size:1.05rem; font-weight:800; letter-spacing:.06em; border:1px solid; }
.green { color:#45f0a5; border-color:#45f0a5; box-shadow:0 0 14px #176044; } .yellow { color:#ffd43b; border-color:#ffd43b; }
.red { color:#ff6b6b; border-color:#ff6b6b; box-shadow:0 0 14px #6c2020; } .warning { margin-top:12px; color:#ffcf70; font-size:.82rem; }
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""


def control(label: str, component: object) -> html.Div:
    return html.Div([html.Label(label, className="control-label"), component], className="mb-3")


app.layout = dbc.Container(fluid=True, className="app-shell", children=[
    html.H1("PULSED FRC  //  DIGITAL TWIN", className="title"),
    html.P("0D coupled energy balance • magnetic compression • direct energy conversion", className="subtitle"),
    dbc.Row([
        dbc.Col(width=3, children=dbc.Card(className="panel h-100", body=True, children=[
            html.H4("MASTER CONTROL", className="panel-title"),
            control("EXTERNAL FIELD  [T]", dcc.Slider(2, 20, 0.5, value=8, id="field", marks={2: "2", 20: "20"})),
            control("INITIAL DENSITY  [10²⁰ m⁻³]", dcc.Slider(0.5, 12, 0.5, value=3, id="density", marks={0.5: "0.5", 12: "12"})),
            control("INITIAL TEMPERATURE  [keV]", dcc.Slider(5, 140, 5, value=45, id="temperature", marks={5: "5", 140: "140"})),
            control("COMPRESSION RATIO", dcc.Slider(1, 10, 0.25, value=4, id="compression", marks={1: "1×", 10: "10×"})),
            control("PULSE DURATION  [ms]", dcc.Slider(2, 30, 1, value=8, id="duration", marks={2: "2", 30: "30"})),
            control("FUEL MIX", dcc.Dropdown([{"label": "D–He3 (aneutronic)", "value": "D-He3"}, {"label": "D–T (neutronic)", "value": "D-T"}], "D-He3", id="fuel", clearable=False)),
            html.Hr(), html.Div(id="status", className="status"), html.Div(id="fuel-warning", className="warning")
        ])),
        dbc.Col(width=9, children=[
            dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="flux-map", config={"displayModeBar": False})), className="panel"), width=7),
                     dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="energy", config={"displayModeBar": False})), className="panel"), width=5)]),
            dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="telemetry", config={"displayModeBar": False})), className="panel mt-3"), width=12))
        ])
    ])
])


def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL, font={"color": "#c8d5e6"},
                      margin={"l": 55, "r": 50, "t": 42, "b": 45}, legend={"orientation": "h", "y": 1.12})
    fig.update_xaxes(gridcolor="#273449", zerolinecolor="#273449")
    fig.update_yaxes(gridcolor="#273449", zerolinecolor="#273449")
    return fig


@app.callback(Output("flux-map", "figure"), Output("energy", "figure"), Output("telemetry", "figure"),
              Output("status", "children"), Output("status", "className"), Output("fuel-warning", "children"),
              Input("field", "value"), Input("density", "value"), Input("temperature", "value"),
              Input("compression", "value"), Input("duration", "value"), Input("fuel", "value"))
def update_twin(field: float, density: float, temperature: float, compression: float, duration: float, fuel: str):
    """One callback performs the ODE once, then fans its telemetry into all plots."""
    result = simulate_frc_pulse(PulseParameters(b_ext=field, density=density * 1e20,
        temperature_kev=temperature, compression_ratio=compression, duration_ms=duration, fuel=fuel))
    tms = result["time_s"] * 1e3
    idx = int(np.argmax(result["net_w_m3"]))

    # Analytic FRC-like flux function: closed surfaces dynamically contract.
    rmax = 0.42 / compression ** (1 / 3)
    zmax = 0.82 / compression ** (1 / 3)
    r = np.linspace(0, rmax * 1.35, 120); z = np.linspace(-zmax * 1.35, zmax * 1.35, 160)
    rr, zz = np.meshgrid(r, z)
    rho2 = (rr / rmax) ** 2 + (zz / zmax) ** 2
    psi = field * rr**2 * (1 - rho2) * np.exp(-0.35 * rho2)
    flux = go.Figure(go.Contour(x=r, y=z, z=psi, colorscale="Turbo", contours={"showlabels": False}, colorbar={"title": "ψ [Wb/rad]"}))
    flux.update_layout(title="LIVE MAGNETIC DIAGNOSTICS — FLUX SURFACES")
    flux.update_xaxes(title="R [m]"); flux.update_yaxes(title="Z [m]", scaleanchor="x", scaleratio=1); style(flux, 355)

    energy = go.Figure()
    for name, value, color in [("Fusion heat", result["fusion_w_m3"][idx], "#ff4d6d"), ("Bremsstrahlung", -result["brems_w_m3"][idx], "#4dabf7"), ("Synchrotron", -result["sync_w_m3"][idx], "#74c0fc"), ("Net thermal", result["net_w_m3"][idx], "#45f0a5")]:
        energy.add_bar(name=name, x=["Peak pulse"], y=[value / 1e6], marker_color=color)
    energy.update_layout(title="INSTANTANEOUS ENERGY BALANCE", barmode="relative", yaxis_title="MW / m³"); style(energy, 355)

    telemetry = make_subplots(specs=[[{"secondary_y": True}]])
    telemetry.add_scatter(x=tms, y=result["temperature_kev"], name="Temperature [keV]", line={"color": "#ff8c42"})
    telemetry.add_scatter(x=tms, y=result["density_m3"] / 1e20, name="Density [10²⁰ m⁻³]", line={"color": CYAN})
    telemetry.add_scatter(x=tms, y=result["net_w_m3"] * result["volume_m3"] / 1e6, name="Net thermal [MW]", line={"color": "#45f0a5"}, secondary_y=True)
    telemetry.add_scatter(x=tms, y=result["emf_v"] / 1e3, name="Induced voltage [kV]", line={"color": "#d0bfff", "dash": "dot"}, secondary_y=True)
    telemetry.update_layout(title="PULSE TELEMETRY", xaxis_title="Time [ms]"); telemetry.update_yaxes(title_text="Temperature / density", secondary_y=False); telemetry.update_yaxes(title_text="Power [MW] / voltage [kV]", secondary_y=True); style(telemetry, 350)
    state = result["status"]; status_class = "status red" if result["unstable"] else ("status green" if "IGNITION" in state else "status yellow")
    warning = "⚠ D–T produces neutron power. Direct Energy Conversion is disabled." if fuel == "D-T" else "✓ Aneutronic D–He3: inductive direct conversion enabled."
    return flux, energy, telemetry, f"REACTOR STATUS: {state}", status_class, warning


if __name__ == "__main__":
    app.run(debug=True)
