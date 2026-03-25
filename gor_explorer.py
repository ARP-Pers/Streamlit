"""
PRB GOR Explorer
Explore spatial and categorical drivers of initial GOR (and other variables)
across the Powder River Basin.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PRB GOR Explorer", layout="wide")

# Continuous variable metadata: name -> (colorscale, label, colorscale)
CONT_VARS = {
    "GORi":   ("RdYlGn_r", "GORi (scf/bbl)",   "RdYlGn_r"),
    "TVD_FT": ("Viridis",  "TVD (ft)",          "Viridis"),
    "PLL":    ("Purples",  "Perf Lateral (ft)", "Purples"),
    "Year":   ("Plasma",   "First Prod Year",   "Plasma"),
}
CAT_VARS = ["Formation", "Operator"]
ALL_COLOR_OPTS = list(CONT_VARS.keys()) + CAT_VARS

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GORi"] = pd.to_numeric(
        df["GORi"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df["FirstProdDate"] = pd.to_datetime(df["FirstProdDate"], errors="coerce")
    df["Year"] = df["FirstProdDate"].dt.year
    df["PLL"] = pd.to_numeric(df["PLL"], errors="coerce")
    df["TVD_FT"] = pd.to_numeric(df["TVD_FT"], errors="coerce")
    return df


import os
_here = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_here, "PRB.csv")
df_raw = load_data(DATA_PATH)


def _label(var: str) -> str:
    return CONT_VARS.get(var, (None, var, None))[1]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("PRB GOR Explorer")
    st.markdown("---")

    gor_max_99 = int(df_raw["GORi"].quantile(0.99))
    gor_cap = st.slider(
        "Cap GORi color scale at (scf/bbl)",
        min_value=500,
        max_value=int(df_raw["GORi"].max()),
        value=gor_max_99,
        step=500,
        help="Winsorize extreme GOR values for color scaling only",
    )

    formations = st.multiselect(
        "Formation",
        sorted(df_raw["Formation"].dropna().unique()),
        default=sorted(df_raw["Formation"].dropna().unique()),
    )

    operators_all = sorted(df_raw["Operator"].dropna().unique())
    operators = st.multiselect("Operator", operators_all, default=operators_all)

    year_min, year_max = int(df_raw["Year"].min()), int(df_raw["Year"].max())
    year_range = st.slider("First Prod Year", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.markdown("**Spatial filter**")

    _lat_min = round(float(df_raw["Latitude"].min()), 4)
    _lat_max = round(float(df_raw["Latitude"].max()), 4)
    _lon_min = round(float(df_raw["Longitude"].min()), 4)
    _lon_max = round(float(df_raw["Longitude"].max()), 4)

    lat_range = st.slider(
        "Latitude",
        min_value=_lat_min, max_value=_lat_max,
        value=(_lat_min, _lat_max), step=0.01,
    )
    lon_range = st.slider(
        "Longitude",
        min_value=_lon_min, max_value=_lon_max,
        value=(_lon_min, _lon_max), step=0.01,
    )

    st.markdown("---")
    st.caption(f"Dataset: {len(df_raw):,} wells")

# ── Global filter ─────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["Formation"].isin(formations)
    & df_raw["Operator"].isin(operators)
    & df_raw["Year"].between(*year_range)
    & df_raw["GORi"].notna()
    & df_raw["Latitude"].between(*lat_range)
    & df_raw["Longitude"].between(*lon_range)
].reset_index(drop=True)

df["GORi_plot"] = df["GORi"].clip(upper=gor_cap)

st.markdown(
    f"### {len(df):,} wells &nbsp;|&nbsp; "
    f"median GORi = **{df['GORi'].median():,.0f}** scf/bbl &nbsp;|&nbsp; "
    f"median TVD = **{df['TVD_FT'].median():,.0f}** ft"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_pivot, tab_scatter, tab_stats = st.tabs(
    ["🗺 Map", "📊 Pivot Analysis", "🔍 Scatter", "📋 Stats Table"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – MAP
# ─────────────────────────────────────────────────────────────────────────────
with tab_map:
    col_color = st.radio(
        "Color wells by",
        ["GORi", "Formation", "Operator", "Year"],
        horizontal=True,
    )

    if col_color == "GORi":
        fig_map = px.scatter_map(
            df,
            lat="Latitude",
            lon="Longitude",
            color="GORi_plot",
            color_continuous_scale="RdYlGn_r",
            hover_name="API",
            hover_data={
                "Formation": True,
                "Operator": True,
                "Year": True,
                "TVD_FT": True,
                "PLL": True,
                "GORi": True,
                "GORi_plot": False,
                "Latitude": False,
                "Longitude": False,
            },
            labels={"GORi_plot": f"GORi (capped {gor_cap:,})"},
            zoom=7,
            height=650,
        )
        fig_map.update_coloraxes(colorbar_title="GORi")
    else:
        fig_map = px.scatter_map(
            df,
            lat="Latitude",
            lon="Longitude",
            color=col_color,
            hover_name="API",
            hover_data={
                "Formation": True,
                "Operator": True,
                "Year": True,
                "TVD_FT": True,
                "PLL": True,
                "GORi": True,
                "Latitude": False,
                "Longitude": False,
            },
            zoom=7,
            height=650,
        )

    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – PIVOT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_pivot:
    st.markdown("#### GOR distribution by category")

    c1, c2 = st.columns([1, 3])
    with c1:
        group_by = st.selectbox(
            "Group by",
            ["Formation", "Operator", "Year", "Formation × Operator"],
        )
        stat = st.selectbox("Aggregate stat", ["Median", "Mean", "P75", "P90"])
        chart_type = st.radio("Chart type", ["Box plot", "Bar (aggregate)", "Violin"])
        min_wells = st.number_input("Min wells per group", value=5, min_value=1)

    stat_fn = {
        "Median": lambda x: x.median(),
        "Mean": lambda x: x.mean(),
        "P75": lambda x: x.quantile(0.75),
        "P90": lambda x: x.quantile(0.90),
    }[stat]

    with c2:
        if group_by == "Formation × Operator":
            df["_group"] = df["Formation"] + " | " + df["Operator"]
            grp_col = "_group"
        else:
            grp_col = group_by

        # Filter out small groups
        counts = df.groupby(grp_col)["GORi"].count()
        valid_groups = counts[counts >= min_wells].index
        df_grp = df[df[grp_col].isin(valid_groups)].copy()

        if chart_type == "Box plot":
            order = (
                df_grp.groupby(grp_col)["GORi"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            fig_piv = px.box(
                df_grp,
                x=grp_col,
                y="GORi",
                category_orders={grp_col: order},
                color=grp_col,
                log_y=True,
                height=520,
                labels={"GORi": "GORi (scf/bbl)"},
            )
            fig_piv.update_layout(showlegend=False, xaxis_tickangle=-45)

        elif chart_type == "Violin":
            order = (
                df_grp.groupby(grp_col)["GORi"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            fig_piv = px.violin(
                df_grp,
                x=grp_col,
                y="GORi",
                category_orders={grp_col: order},
                color=grp_col,
                log_y=True,
                box=True,
                height=520,
                labels={"GORi": "GORi (scf/bbl)"},
            )
            fig_piv.update_layout(showlegend=False, xaxis_tickangle=-45)

        else:  # Bar aggregate
            agg = (
                df_grp.groupby(grp_col)["GORi"]
                .agg(func=stat_fn)
                .reset_index()
                .sort_values("GORi", ascending=False)
            )
            agg.columns = [grp_col, "GORi_stat"]
            fig_piv = px.bar(
                agg,
                x=grp_col,
                y="GORi_stat",
                color="GORi_stat",
                color_continuous_scale="RdYlGn_r",
                height=520,
                labels={"GORi_stat": f"{stat} GORi (scf/bbl)"},
            )
            fig_piv.update_layout(showlegend=False, xaxis_tickangle=-45)

        st.plotly_chart(fig_piv, width='stretch')

    # Pivot table
    st.markdown("#### Pivot table")
    row_var = st.selectbox("Row", ["Formation", "Operator", "Year"], key="pt_row")
    col_var = st.selectbox(
        "Column",
        ["Formation", "Operator", "Year"],
        index=2,
        key="pt_col",
    )
    agg_var = st.selectbox(
        "Value (GORi)",
        ["Median", "Mean", "Count", "P75", "P90"],
        key="pt_agg",
    )

    agg_map = {
        "Median": ("GORi", "median"),
        "Mean": ("GORi", "mean"),
        "Count": ("GORi", "count"),
        "P75": ("GORi", lambda x: x.quantile(0.75)),
        "P90": ("GORi", lambda x: x.quantile(0.90)),
    }

    fn_label, fn = agg_map[agg_var]
    pivot_df = df.pivot_table(
        values="GORi", index=row_var, columns=col_var, aggfunc=fn
    )
    pivot_df = pivot_df.round(0)

    # Sort rows by median GORi
    row_order = df.groupby(row_var)["GORi"].median().sort_values(ascending=False).index
    pivot_df = pivot_df.reindex(row_order.intersection(pivot_df.index))

    # Heatmap
    fig_heat = px.imshow(
        pivot_df,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        height=max(300, 30 * len(pivot_df)),
        labels={"color": f"{agg_var} GORi"},
    )
    fig_heat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_heat, width='stretch')

    with st.expander("Show raw pivot table"):
        st.dataframe(pivot_df.style.format("{:,.0f}", na_rep="–"), width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – SCATTER
# ─────────────────────────────────────────────────────────────────────────────
with tab_scatter:
    st.markdown("#### GORi vs continuous variables")

    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("X axis", ["TVD_FT", "PLL", "Year"], key="sc_x")
        color_var = st.selectbox(
            "Color", ["Formation", "Operator", "Year", "None"], key="sc_c"
        )
    with c2:
        log_y = st.checkbox("Log scale GORi", value=True)
        show_trend = st.checkbox("Show trendline (OLS)", value=True)

    color_arg = None if color_var == "None" else color_var

    fig_sc = px.scatter(
        df,
        x=x_var,
        y="GORi",
        color=color_arg,
        opacity=0.5,
        log_y=log_y,
        hover_data=["API", "Formation", "Operator", "Year", "TVD_FT", "PLL", "GORi"],
        height=550,
        labels={"GORi": "GORi (scf/bbl)", x_var: x_var},
    )

    if show_trend:
        valid = df[[x_var, "GORi"]].dropna()
        y_fit = np.log(valid["GORi"]) if log_y else valid["GORi"]
        try:
            coeffs = np.polyfit(valid[x_var], y_fit, 1)
            x_line = np.linspace(valid[x_var].min(), valid[x_var].max(), 200)
            y_line = np.polyval(coeffs, x_line)
            if log_y:
                y_line = np.exp(y_line)
            fig_sc.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name="Trend (linear fit)",
                )
            )
        except Exception:
            pass

    fig_sc.update_layout(legend=dict(itemsizing="constant"))
    st.plotly_chart(fig_sc, width='stretch')

    # Correlation summary
    numeric_cols = ["GORi", "TVD_FT", "PLL", "Year"]
    corr = df[numeric_cols].corr()[["GORi"]].drop("GORi").round(3)
    corr.columns = ["Pearson r with GORi"]
    corr["|r|"] = corr["Pearson r with GORi"].abs()
    corr = corr.sort_values("|r|", ascending=False).drop(columns="|r|")
    st.markdown("**Pearson correlation with GORi** (all filtered wells):")
    st.dataframe(corr, width='content')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – STATS TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab_stats:
    st.markdown("#### Summary statistics per group")

    group_col = st.selectbox(
        "Group by", ["Formation", "Operator", "Year"], key="st_grp"
    )

    summary = (
        df.groupby(group_col)["GORi"]
        .agg(
            Count="count",
            Median="median",
            Mean="mean",
            Std="std",
            P25=lambda x: x.quantile(0.25),
            P75=lambda x: x.quantile(0.75),
            P90=lambda x: x.quantile(0.90),
            Max="max",
        )
        .round(0)
        .sort_values("Median", ascending=False)
        .reset_index()
    )

    st.dataframe(
        summary.style.background_gradient(subset=["Median", "P90"], cmap="RdYlGn_r"),
        width='stretch',
        height=600,
    )

    csv_out = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_out,
        file_name=f"GOR_stats_by_{group_col}.csv",
        mime="text/csv",
    )
