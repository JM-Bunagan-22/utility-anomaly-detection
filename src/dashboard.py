"""
Dash dashboard visualizing daily consumption with flagged anomalies.

Run: python src/dashboard.py
Then open http://localhost:8050
"""

import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FLAGGED_PATH = os.path.join(DATA_DIR, "flagged_consumption.csv")

df = pd.read_csv(FLAGGED_PATH, parse_dates=["date"])

app = Dash(__name__)
app.title = "Utility Consumption Anomaly Detection"

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["date"], y=df["total_kwh"],
    mode="lines", name="Daily kWh",
    line=dict(color="#2b6cb0", width=1.5),
))

flagged = df[df["flagged_by_both"]]
fig.add_trace(go.Scatter(
    x=flagged["date"], y=flagged["total_kwh"],
    mode="markers", name="Flagged anomaly",
    marker=dict(color="#e53e3e", size=9, symbol="circle-open", line=dict(width=2)),
))

fig.update_layout(
    title="Daily Household Power Consumption — Flagged Anomalies",
    xaxis_title="Date",
    yaxis_title="Total kWh",
    template="plotly_white",
    hovermode="x unified",
)

summary_cards = html.Div([
    html.Div([
        html.H3(f"{len(df)}"), html.P("Days analyzed"),
    ], className="card"),
    html.Div([
        html.H3(f"{int(df['flag_zscore'].sum())}"), html.P("Flagged (z-score)"),
    ], className="card"),
    html.Div([
        html.H3(f"{int(df['flag_isoforest'].sum())}"), html.P("Flagged (Isolation Forest)"),
    ], className="card"),
    html.Div([
        html.H3(f"{int(df['flagged_by_both'].sum())}"), html.P("High-confidence flags"),
    ], className="card"),
], style={"display": "flex", "gap": "20px", "marginBottom": "20px"})

app.layout = html.Div([
    html.H1("⚡ Utility Consumption Anomaly Detection"),
    html.P("Flagging suspicious consumption days using rolling z-score and Isolation Forest, based on public household power data."),
    summary_cards,
    dcc.Graph(figure=fig),
    html.H3("Flagged Days (high confidence)"),
    dcc.Graph(
        figure=go.Figure(
            data=[go.Table(
                header=dict(values=["Date", "Total kWh", "Z-score", "IsoForest Score"]),
                cells=dict(values=[
                    flagged["date"].dt.strftime("%Y-%m-%d"),
                    flagged["total_kwh"].round(2),
                    flagged["zscore"].round(2),
                    flagged["isoforest_score"].round(3),
                ]),
            )]
        )
    ),
], style={"fontFamily": "Arial, sans-serif", "margin": "40px"})

if __name__ == "__main__":
    app.run(debug=True)
