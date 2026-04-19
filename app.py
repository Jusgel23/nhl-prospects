"""
NHL Prospect Prediction — Streamlit frontend.

Run with:
    streamlit run app.py

Pick up to two prospects from the sidebar to render side-by-side
scouting cards modeled on the Andy & Rono prospect-card template.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.database import (
    load_outcomes,
    load_players,
    load_predictions,
    load_seasons,
)
from src.models.features import build_feature_matrix
from src.comparables.similarity import build_comparable_index


# ── constants ────────────────────────────────────────────────────────────────

PLACEHOLDER = "—"

# Status taxonomy — matches the legend at the bottom of the template
STATUS_STYLES = {
    "Superstar":             {"color": "#7C1E3A", "icon": "★"},
    "Star Producer":         {"color": "#C0392B", "icon": "★"},
    "Fringe Star":           {"color": "#E67E22", "icon": "★"},
    "Average Producer":      {"color": "#2874A6", "icon": "●"},
    "Replacement Producer":  {"color": "#5D6D7E", "icon": "●"},
    "100 Gamer":             {"color": "#B7950B", "icon": "×"},
    "Bust":                  {"color": "#78281F", "icon": "✕"},
    "Developing":            {"color": "#1F618D", "icon": "◐"},
}

# Accent color for the top bar of each card — mapped from status
CARD_ACCENTS = {
    "Superstar":            "#B22222",
    "Star Producer":        "#C0392B",
    "Fringe Star":          "#E67E22",
    "Average Producer":     "#2E86C1",
    "Replacement Producer": "#5D6D7E",
    "100 Gamer":            "#B7950B",
    "Bust":                 "#78281F",
    "Developing":           "#C0392B",
}

# Outcome icon for comparables rows (maps to the legend)
OUTCOME_ICONS = {
    "Elite (Star NHLer)":   ("★", "#D4AC0D"),   # gold star
    "NHLer (Role Player)":  ("◐", "#27AE60"),   # green half-circle
    "Did Not Reach NHL":    ("—", "#7B7D7D"),   # grey dash
    "Unknown":              ("?", "#95A5A6"),
}

# Minimal flag map — add as needed
FLAG_EMOJI = {
    "CAN": "🇨🇦", "USA": "🇺🇸", "SWE": "🇸🇪", "FIN": "🇫🇮",
    "RUS": "🇷🇺", "CZE": "🇨🇿", "SVK": "🇸🇰", "GER": "🇩🇪",
    "SUI": "🇨🇭", "DEN": "🇩🇰", "BLR": "🇧🇾", "LAT": "🇱🇻",
    "NOR": "🇳🇴", "AUT": "🇦🇹", "FRA": "🇫🇷",
}


# ── data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all():
    players = load_players()
    seasons = load_seasons()
    outcomes = load_outcomes()
    try:
        predictions = load_predictions()
    except Exception:
        predictions = pd.DataFrame()
    return players, seasons, outcomes, predictions


@st.cache_resource(show_spinner="Building comparables index…")
def get_comp_index(_players: pd.DataFrame, _seasons: pd.DataFrame,
                   _outcomes: pd.DataFrame):
    if _outcomes.empty:
        return None
    hist = build_feature_matrix(_players, _seasons, _outcomes)
    labeled = hist[hist.get("is_nhler").notna()] if "is_nhler" in hist.columns else pd.DataFrame()
    if labeled.empty or len(labeled) < 10:
        return None
    return build_comparable_index(labeled)


@st.cache_data(show_spinner=False)
def get_feature_row(_players: pd.DataFrame, _seasons: pd.DataFrame,
                    _outcomes: pd.DataFrame, player_id: str) -> pd.Series | None:
    """Single-row feature vector for one prospect (used to feed comparables)."""
    sub_players = _players[_players["player_id"] == player_id]
    sub_seasons = _seasons[_seasons["player_id"] == player_id]
    if sub_players.empty or sub_seasons.empty:
        return None
    outcomes = _outcomes[_outcomes["player_id"] == player_id] if not _outcomes.empty else None
    feats = build_feature_matrix(sub_players, sub_seasons, outcomes)
    if feats.empty:
        return None
    return feats.iloc[0]


# ── derivations ──────────────────────────────────────────────────────────────

def calc_age(dob: str) -> float | None:
    try:
        b = datetime.strptime(dob, "%Y-%m-%d").date()
        d = date.today()
        yrs = (d - b).days / 365.25
        return round(yrs, 1)
    except Exception:
        return None


def height_display(cm: float | None) -> str:
    if not cm or pd.isna(cm):
        return PLACEHOLDER
    inches_total = round(cm / 2.54)
    ft, inch = divmod(inches_total, 12)
    return f"{ft}'{inch}\""


def weight_display(kg: float | None) -> str:
    if not kg or pd.isna(kg):
        return PLACEHOLDER
    return f"{round(kg * 2.20462)}"


def format_born(dob: str | None) -> str:
    if not dob:
        return PLACEHOLDER
    try:
        return datetime.strptime(dob, "%Y-%m-%d").strftime("%b %d, %Y")
    except Exception:
        return dob


def status_for(player_id: str, outcomes: pd.DataFrame,
               predictions: pd.DataFrame) -> str:
    """Pick a status bucket from realized career first, else from the model."""
    out = outcomes[outcomes["player_id"] == player_id]
    if not out.empty:
        o = out.iloc[0]
        nhl_gp = o.get("nhl_gp") or 0
        nhl_pts = o.get("nhl_points") or 0
        if nhl_gp and nhl_gp > 0:
            ppg = nhl_pts / nhl_gp if nhl_gp else 0
            is_star = bool(o.get("is_star", 0))
            is_nhler = bool(o.get("is_nhler", 0))
            if is_star and ppg >= 0.9:    return "Superstar"
            if is_star:                    return "Star Producer"
            if is_nhler and nhl_gp >= 500: return "Average Producer"
            if is_nhler:                   return "Replacement Producer"
            if nhl_gp >= 100:              return "100 Gamer"
            return "Bust"

    pred = predictions[predictions["player_id"] == player_id] if not predictions.empty else pd.DataFrame()
    if not pred.empty:
        p = pred.iloc[0]
        star = p.get("star_probability") or 0
        nhler = p.get("nhler_probability") or 0
        if star >= 0.50: return "Superstar"
        if star >= 0.30: return "Star Producer"
        if star >= 0.15: return "Fringe Star"
        if nhler >= 0.55: return "Average Producer"
        if nhler >= 0.35: return "Replacement Producer"
        if nhler >= 0.20: return "100 Gamer"
        if nhler < 0.10:  return "Bust"
    return "Developing"


def get_dev_stages(seasons_for_player: pd.DataFrame,
                   draft_year: int | None) -> pd.DataFrame:
    """
    Return a DataFrame indexed by stage label (D-1..D3) with season year
    and NHLe points-per-82-game value for the bar chart.
    Uses `draft_label` if present, else derives from `season` & `draft_year`.
    """
    df = seasons_for_player.copy()
    if df.empty:
        return df

    if "draft_label" not in df.columns or df["draft_label"].isna().all():
        if draft_year:
            yr = df["season"].str.split("-").str[0].astype(int)
            df["draft_label"] = (yr - int(draft_year)).apply(
                lambda d: f"D{int(d):+d}" if pd.notna(d) else None
            )
        else:
            df["draft_label"] = None

    df["season_yr"] = df["season"].str.split("-").str[0].astype(int) + 1
    if df["draft_label"].isna().all():
        return pd.DataFrame(columns=["draft_label", "season_yr", "nhle_ppg", "nhle82"])

    # One row per stage (max NHLe PPG if multiple leagues that season)
    agg = (
        df.groupby("draft_label", dropna=True)
          .agg(season_yr=("season_yr", "max"),
               nhle_ppg=("nhle_ppg", "max"))
          .reset_index()
    )
    agg["nhle82"] = (agg["nhle_ppg"] * 82).round().astype("Int64")
    return agg


def get_stage_probs(seasons_for_player: pd.DataFrame,
                    outcomes_row: pd.Series | None,
                    predictions_row: pd.Series | None) -> dict[str, dict]:
    """
    Build {stage: {"star": prob_or_None, "nhler": prob_or_None}} for D0..D3.
    - Realized historical career → 0.99 / 0.01 based on actual outcome at every
      stage the player actually skated.
    - Current prospect → single model probability at their most-recent stage
      present in the seasons data.
    - All other stages → None (rendered as placeholder).
    """
    stages = ["D+0", "D+1", "D+2", "D+3"]
    result = {s: {"star": None, "nhler": None} for s in stages}

    seasons_stages = set()
    if "draft_label" in seasons_for_player.columns:
        seasons_stages = {
            s for s in seasons_for_player["draft_label"].dropna().unique()
            if s in stages
        }

    if outcomes_row is not None and (outcomes_row.get("nhl_gp") or 0) > 0:
        is_star = bool(outcomes_row.get("is_star", 0))
        is_nhler = bool(outcomes_row.get("is_nhler", 0))
        for s in stages:
            if s in seasons_stages or s in {"D+0", "D+1", "D+2", "D+3"}:
                result[s]["star"] = 0.99 if is_star else 0.01
                result[s]["nhler"] = 0.99 if is_nhler else 0.01
        return result

    if predictions_row is not None:
        # Attach model prob to the latest stage we have data for, else D+0
        latest = None
        ordered = [s for s in stages if s in seasons_stages]
        if ordered:
            ordered.sort()
            latest = ordered[-1]
        else:
            latest = "D+0"

        star = predictions_row.get("star_probability")
        nhler = predictions_row.get("nhler_probability")
        if pd.notna(star):
            result[latest]["star"] = float(star)
        if pd.notna(nhler):
            result[latest]["nhler"] = float(nhler)

    return result


# ── rendering ────────────────────────────────────────────────────────────────

def nhle_bar_chart(stages_df: pd.DataFrame, accent: str) -> go.Figure:
    """Horizontal bar chart of NHLe-82 by development stage (D-1..D+3)."""
    order = ["D-1", "D+0", "D+1", "D+2", "D+3"]
    rows = []
    for label in order:
        match = stages_df[stages_df["draft_label"] == label]
        if match.empty:
            rows.append({"label": label, "year": None, "value": None})
        else:
            r = match.iloc[0]
            rows.append({
                "label": label,
                "year": int(r["season_yr"]) if pd.notna(r["season_yr"]) else None,
                "value": int(r["nhle82"]) if pd.notna(r["nhle82"]) else None,
            })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_bar(
        y=df["label"],
        x=df["value"].fillna(0),
        orientation="h",
        marker=dict(color=accent),
        text=[f"{v}" if pd.notna(v) else PLACEHOLDER for v in df["value"]],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="white", size=13),
        hovertemplate="%{y}: %{x} NHLe pts/82<extra></extra>",
    )
    # Year annotations on left
    for i, r in df.iterrows():
        fig.add_annotation(
            x=0, y=r["label"],
            text=str(r["year"]) if r["year"] else PLACEHOLDER,
            showarrow=False, xanchor="right", xshift=-6,
            font=dict(size=11, color="#555"),
        )
    fig.update_layout(
        height=200, margin=dict(l=60, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0, max(100, (df["value"].max() or 0) * 1.2)]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def prob_row_html(probs: dict[str, dict], kind: str, title: str) -> str:
    """Render D0-D3 probability circles horizontally."""
    stages = ["D+0", "D+1", "D+2", "D+3"]
    short = {"D+0": "D0", "D+1": "D1", "D+2": "D2", "D+3": "D3"}

    circles = []
    for s in stages:
        val = probs[s][kind]
        if val is None:
            txt, bg, fg, border = PLACEHOLDER, "#EAECEE", "#909497", "#BFC9CA"
        else:
            pct = int(round(val * 100))
            txt = f"{pct}%"
            # Blue ramp: light → deep based on value
            if pct >= 70:
                bg, fg, border = "#1F4E79", "white", "#1F4E79"
            elif pct >= 40:
                bg, fg, border = "#2E86C1", "white", "#2E86C1"
            elif pct >= 20:
                bg, fg, border = "#85C1E9", "#1B2631", "#85C1E9"
            else:
                bg, fg, border = "#D6EAF8", "#1B2631", "#AED6F1"
        circles.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;">'
            f'  <div style="width:44px;height:44px;border-radius:50%;'
            f'       background:{bg};color:{fg};border:2px solid {border};'
            f'       display:flex;align-items:center;justify-content:center;'
            f'       font-size:12px;font-weight:600;">{txt}</div>'
            f'  <div style="font-size:10px;color:#566573;margin-top:4px;">{short[s]}</div>'
            f'</div>'
        )
    return (
        f'<div style="background:#F4F6F7;padding:10px 12px;border-radius:4px;">'
        f'  <div style="font-size:12px;font-weight:700;color:#1B2631;margin-bottom:8px;">{title}</div>'
        f'  <div style="display:flex;gap:12px;justify-content:flex-start;">{"".join(circles)}</div>'
        f'</div>'
    )


def comps_table_html(title: str, comps: pd.DataFrame) -> str:
    rows_html = []
    if comps is None or comps.empty:
        rows_html.append(
            f'<tr><td colspan="2" style="padding:8px;color:#909497;font-style:italic;">'
            f'No comparables available</td></tr>'
        )
    else:
        for _, r in comps.iterrows():
            icon, color = OUTCOME_ICONS.get(r.get("outcome_label", "Unknown"),
                                             OUTCOME_ICONS["Unknown"])
            rows_html.append(
                f'<tr>'
                f'  <td style="padding:6px 8px;border-bottom:1px solid #EAECEE;">{r["name"]}</td>'
                f'  <td style="padding:6px 8px;border-bottom:1px solid #EAECEE;'
                f'      text-align:right;color:{color};font-weight:700;">{icon}</td>'
                f'</tr>'
            )
    return (
        f'<div style="background:white;border:1px solid #D5DBDB;border-radius:4px;overflow:hidden;">'
        f'  <div style="background:#1B2631;color:white;padding:6px 10px;font-weight:700;font-size:12px;">{title}</div>'
        f'  <table style="width:100%;border-collapse:collapse;font-size:12px;">'
        f'    {"".join(rows_html)}'
        f'  </table>'
        f'</div>'
    )


def render_card(player_row: pd.Series, seasons_sub: pd.DataFrame,
                outcome_row: pd.Series | None, pred_row: pd.Series | None,
                status: str, dy_comps: pd.DataFrame, full_comps: pd.DataFrame):
    accent = CARD_ACCENTS.get(status, "#1F618D")
    style = STATUS_STYLES.get(status, STATUS_STYLES["Developing"])

    # ── Accent bar ───────────────────────────────────────────────
    st.markdown(
        f'<div style="height:6px;background:{accent};border-radius:2px;margin-bottom:8px;"></div>',
        unsafe_allow_html=True,
    )

    # ── Header: name + flag + status + logo placeholder ─────────
    flag = FLAG_EMOJI.get((player_row.get("nationality") or "").upper(), "🏒")
    name = player_row.get("name", "Unknown")
    pos = (player_row.get("position") or "").upper()[:1] or "—"

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(
            f'<div style="font-size:22px;font-weight:800;color:#1B2631;line-height:1.1;">{name}</div>'
            f'<div style="margin-top:2px;font-size:13px;color:#566573;">'
            f'{flag} <span style="color:{style["color"]};font-weight:700;">{status}</span>'
            f' &nbsp;|&nbsp; <b>{pos}</b></div>',
            unsafe_allow_html=True,
        )
    with header_col2:
        st.markdown(
            '<div style="text-align:right;font-size:11px;color:#909497;font-style:italic;">'
            'drafted by</div>'
            '<div style="text-align:right;font-size:28px;">🏒</div>',
            unsafe_allow_html=True,
        )

    # ── Bio stats ────────────────────────────────────────────────
    gp = int(seasons_sub["gp"].sum()) if not seasons_sub.empty else 0
    assists = int(seasons_sub["assists"].sum()) if not seasons_sub.empty else 0
    points = int(seasons_sub["points"].sum()) if not seasons_sub.empty else 0
    ppg = round(points / gp, 2) if gp else 0

    dy = int(outcome_row["draft_year"]) if outcome_row is not None and pd.notna(outcome_row.get("draft_year")) else PLACEHOLDER
    rnd = int(outcome_row["draft_round"]) if outcome_row is not None and pd.notna(outcome_row.get("draft_round")) else PLACEHOLDER
    pick = int(outcome_row["draft_pick"]) if outcome_row is not None and pd.notna(outcome_row.get("draft_pick")) else PLACEHOLDER

    bio = (
        f'<div style="font-size:12px;color:#2C3E50;line-height:1.8;margin-top:6px;">'
        f'<b>GP:</b> {gp} | <b>A:</b> {assists} | <b>Pts:</b> {points} | <b>PPG:</b> {ppg}<br>'
        f'<b>Born:</b> {format_born(player_row.get("dob"))} | '
        f'<b>Age:</b> {calc_age(player_row.get("dob")) or PLACEHOLDER}<br>'
        f'<b>H:</b> {height_display(player_row.get("height_cm"))} | '
        f'<b>W:</b> {weight_display(player_row.get("weight_kg"))} | '
        f'<b>Shoots:</b> {player_row.get("shoots") or PLACEHOLDER}<br>'
        f'<b>DY:</b> {dy} | <b>Round:</b> {rnd} | <b>Pick:</b> {pick}'
        f'</div>'
    )
    st.markdown(bio, unsafe_allow_html=True)

    # ── NHLe bar chart ───────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:14px;font-size:12px;font-weight:700;color:#1B2631;">NHLe (pts / 82 games)</div>',
        unsafe_allow_html=True,
    )
    draft_year = int(outcome_row["draft_year"]) if outcome_row is not None and pd.notna(outcome_row.get("draft_year")) else None
    stages_df = get_dev_stages(seasons_sub, draft_year)
    if not stages_df.empty:
        st.plotly_chart(nhle_bar_chart(stages_df, accent),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"nhle_{player_row['player_id']}")
    else:
        st.caption("No season data available.")

    # ── Probability rows ─────────────────────────────────────────
    probs = get_stage_probs(seasons_sub, outcome_row, pred_row)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(prob_row_html(probs, "star", "Star Probabilities"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(prob_row_html(probs, "nhler", "NHLer Probabilities"),
                    unsafe_allow_html=True)

    # ── Comparables ──────────────────────────────────────────────
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(comps_table_html("DY Comps", dy_comps), unsafe_allow_html=True)
    with d2:
        st.markdown(comps_table_html("Full Comps", full_comps), unsafe_allow_html=True)


def render_legend():
    items = []
    for label, s in STATUS_STYLES.items():
        items.append(
            f'<span style="margin-right:14px;font-size:11px;">'
            f'<b style="color:{s["color"]};">{s["icon"]}</b> '
            f'<span style="color:#2C3E50;text-transform:uppercase;letter-spacing:0.5px;">{label}</span>'
            f'</span>'
        )
    st.markdown(
        f'<div style="margin-top:18px;padding:10px 12px;background:#1B2631;color:white;'
        f' border-radius:4px;">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="NHL Prospect Cards",
        page_icon="🏒",
        layout="wide",
    )
    st.title("🏒 NHL Prospect Cards")
    st.caption("Pick up to two prospects to compare.")

    players, seasons, outcomes, predictions = load_all()

    if players.empty:
        st.error("No player data found. Run `python main.py collect` first.")
        return

    # Sidebar filters & picker
    st.sidebar.header("Pick prospects")

    draft_years = sorted(
        [int(y) for y in outcomes["draft_year"].dropna().unique()]
    ) if not outcomes.empty else []
    years_opt = ["All"] + [str(y) for y in draft_years]
    year_filter = st.sidebar.selectbox("Draft year filter", years_opt, index=0)

    positions = sorted(p for p in players["position"].dropna().unique() if p)
    pos_filter = st.sidebar.multiselect("Position", positions, default=[])

    # Apply filters to pick list
    pick_df = players.copy()
    if pos_filter:
        pick_df = pick_df[pick_df["position"].isin(pos_filter)]
    if year_filter != "All" and not outcomes.empty:
        ids_in_year = outcomes[outcomes["draft_year"] == int(year_filter)]["player_id"]
        pick_df = pick_df[pick_df["player_id"].isin(ids_in_year)]

    pick_df = pick_df.sort_values("name")
    name_to_id = dict(zip(pick_df["name"], pick_df["player_id"]))
    name_opts = ["—"] + list(pick_df["name"])

    player_a_name = st.sidebar.selectbox("Player A", name_opts, index=min(1, len(name_opts) - 1))
    player_b_name = st.sidebar.selectbox("Player B (optional)", name_opts, index=0)

    comp_index = get_comp_index(players, seasons, outcomes)
    if comp_index is None:
        st.info("Comparables index unavailable (no historical outcomes loaded). "
                "Run `python main.py train` to enable comps.")

    chosen_ids = [name_to_id.get(n) for n in [player_a_name, player_b_name] if n and n != "—"]
    if not chosen_ids:
        st.warning("Pick a player from the sidebar to render a card.")
        return

    cols = st.columns(len(chosen_ids))
    for col, pid in zip(cols, chosen_ids):
        with col:
            player_row = players[players["player_id"] == pid].iloc[0]
            seasons_sub = seasons[seasons["player_id"] == pid].copy()

            out_match = outcomes[outcomes["player_id"] == pid] if not outcomes.empty else pd.DataFrame()
            outcome_row = out_match.iloc[0] if not out_match.empty else None

            pred_match = predictions[predictions["player_id"] == pid] if not predictions.empty else pd.DataFrame()
            pred_row = pred_match.iloc[0] if not pred_match.empty else None

            status = status_for(pid, outcomes, predictions)

            dy_comps = pd.DataFrame()
            full_comps = pd.DataFrame()
            if comp_index is not None:
                feat_row = get_feature_row(players, seasons, outcomes, pid)
                if feat_row is not None:
                    full_comps = comp_index.find_comparables(feat_row, n=5,
                                                              same_draft_year_only=False)
                    if outcome_row is not None and pd.notna(outcome_row.get("draft_year")):
                        feat_row_dy = feat_row.copy()
                        feat_row_dy["draft_year"] = outcome_row["draft_year"]
                        dy_comps = comp_index.find_comparables(feat_row_dy, n=5,
                                                                same_draft_year_only=True)

            render_card(player_row, seasons_sub, outcome_row, pred_row,
                        status, dy_comps, full_comps)

    render_legend()


if __name__ == "__main__":
    main()
