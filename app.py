"""
NHL Prospect Prediction — Streamlit frontend.

Run with:
    streamlit run app.py

Board view: sortable table of prospects on the left, detail card on the right
when a row is selected. Sidebar filters the board by draft year, position,
and free-text name search.
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
from src.data.images import headshot_url, nhl_id_for
from src.models.features import build_feature_matrix
from src.comparables.similarity import build_comparable_index


# ── constants ────────────────────────────────────────────────────────────────

PLACEHOLDER = "—"

# Status taxonomy — matches the legend at the bottom of the board
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

# Accent color for the top bar of the detail card — mapped from status
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
    "Elite (Star NHLer)":   ("★", "#D4AC0D"),
    "NHLer (Role Player)":  ("◐", "#27AE60"),
    "Did Not Reach NHL":    ("—", "#7B7D7D"),
    "Unknown":              ("?", "#95A5A6"),
}

# Minimal flag map — add as needed
FLAG_EMOJI = {
    "CAN": "🇨🇦", "USA": "🇺🇸", "SWE": "🇸🇪", "FIN": "🇫🇮",
    "RUS": "🇷🇺", "CZE": "🇨🇿", "SVK": "🇸🇰", "GER": "🇩🇪",
    "SUI": "🇨🇭", "DEN": "🇩🇰", "BLR": "🇧🇾", "LAT": "🇱🇻",
    "NOR": "🇳🇴", "AUT": "🇦🇹", "FRA": "🇫🇷", "CZ": "🇨🇿",
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

def calc_age(dob) -> float | None:
    if not dob or pd.isna(dob):
        return None
    try:
        b = datetime.strptime(str(dob), "%Y-%m-%d").date()
        return round((date.today() - b).days / 365.25, 1)
    except Exception:
        return None


def height_display(cm) -> str:
    if not cm or pd.isna(cm):
        return PLACEHOLDER
    inches_total = round(float(cm) / 2.54)
    ft, inch = divmod(inches_total, 12)
    return f"{ft}'{inch}\""


def weight_display(kg) -> str:
    if not kg or pd.isna(kg):
        return PLACEHOLDER
    return f"{round(float(kg) * 2.20462)}"


def format_born(dob) -> str:
    if not dob or pd.isna(dob):
        return PLACEHOLDER
    try:
        return datetime.strptime(str(dob), "%Y-%m-%d").strftime("%b %d, %Y")
    except Exception:
        return str(dob)


def status_for(player_id: str, outcomes: pd.DataFrame,
               predictions: pd.DataFrame) -> str:
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
    df = seasons_for_player.copy()
    if df.empty:
        return df

    if "draft_label" not in df.columns or df["draft_label"].isna().all():
        if draft_year:
            yr = df["season"].astype(str).str.split("-").str[0].astype(int)
            df["draft_label"] = (yr - int(draft_year)).apply(
                lambda d: f"D{int(d):+d}" if pd.notna(d) else None
            )
        else:
            df["draft_label"] = None

    df["season_yr"] = df["season"].astype(str).str.split("-").str[0].astype(int) + 1
    if df["draft_label"].isna().all():
        return pd.DataFrame(columns=["draft_label", "season_yr", "nhle_ppg", "nhle82"])

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
        latest = None
        ordered = sorted(s for s in stages if s in seasons_stages)
        latest = ordered[-1] if ordered else "D+0"

        star = predictions_row.get("star_probability")
        nhler = predictions_row.get("nhler_probability")
        if pd.notna(star):
            result[latest]["star"] = float(star)
        if pd.notna(nhler):
            result[latest]["nhler"] = float(nhler)

    return result


# ── chart helpers (unchanged) ────────────────────────────────────────────────

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
        f'</table>'
        f'</div>'
    )


# ── board DataFrame ──────────────────────────────────────────────────────────

def _status_icon_label(status: str) -> str:
    s = STATUS_STYLES.get(status, STATUS_STYLES["Developing"])
    return f"{s['icon']} {status}"


def _draft_pick_display(row) -> str:
    dy = row.get("draft_year")
    rnd = row.get("draft_round")
    pick = row.get("draft_pick")
    if pd.isna(dy) or dy in (None, 0):
        return PLACEHOLDER
    parts = [str(int(dy))]
    if pd.notna(rnd) and rnd not in (None, 0):
        parts.append(f"R{int(rnd)}")
    if pd.notna(pick) and pick not in (None, 0):
        parts.append(f"#{int(pick)}")
    return " ".join(parts)


def _best_nhle_ppg(player_id: str, seasons: pd.DataFrame) -> float:
    sub = seasons[seasons["player_id"] == player_id]
    if sub.empty or "nhle_ppg" not in sub.columns:
        return 0.0
    best = sub["nhle_ppg"].max()
    return float(best) if pd.notna(best) else 0.0


@st.cache_data(show_spinner="Building prospect board…")
def build_board(_players: pd.DataFrame, _seasons: pd.DataFrame,
                _outcomes: pd.DataFrame, _predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per player — the sortable board view."""
    if _players.empty:
        return pd.DataFrame()

    df = _players[["player_id", "name", "position", "nationality", "dob",
                    "draft_year", "draft_round", "draft_pick", "draft_team"]].copy()

    # Join predictions (most recent model output per player)
    if not _predictions.empty:
        pred_cols = ["player_id", "nhler_probability", "star_probability",
                     "projected_career_pts", "rank_score"]
        pred_cols = [c for c in pred_cols if c in _predictions.columns]
        df = df.merge(_predictions[pred_cols], on="player_id", how="left")
    for col in ("nhler_probability", "star_probability",
                "projected_career_pts", "rank_score"):
        if col not in df.columns:
            df[col] = pd.NA

    # Headshot URL via NHL ID map
    df["headshot_url"] = df["player_id"].astype(str).apply(headshot_url)

    # Best NHLe across career
    df["nhle_ppg"] = df["player_id"].apply(lambda pid: _best_nhle_ppg(pid, _seasons))

    # Status bucket
    df["status"] = df["player_id"].apply(
        lambda pid: status_for(pid, _outcomes, _predictions)
    )

    # Display helpers
    df["draft_display"] = df.apply(_draft_pick_display, axis=1)
    df["status_display"] = df["status"].apply(_status_icon_label)
    df["nhler_pct"] = (df["nhler_probability"] * 100).round().astype("Int64")
    df["star_pct"] = (df["star_probability"] * 100).round().astype("Int64")
    df["proj_pts"] = df["projected_career_pts"].round().astype("Int64")

    # Rank: prefer model rank_score; fall back to nhler_probability desc
    sort_key = "rank_score" if df["rank_score"].notna().any() else "nhler_probability"
    df = df.sort_values(
        [sort_key, "nhler_probability", "star_probability"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def filter_board(board: pd.DataFrame, *,
                 draft_year: str | None = None,
                 positions: list[str] | None = None,
                 search: str | None = None,
                 include_settled: bool = False) -> pd.DataFrame:
    """Apply sidebar filters to the board DataFrame."""
    df = board.copy()

    # By default drop veterans whose careers are settled via nhl_outcomes
    # (so the "board" reads like a scouting board, not a roster ledger).
    # Users can tick a box to include them.
    if not include_settled:
        active_statuses = {"Developing", "100 Gamer", "Fringe Star",
                           "Star Producer", "Superstar",
                           "Bust", "Average Producer",
                           "Replacement Producer"}
        # Heuristic: if a player's draft_year is >= 2022 OR they lack one,
        # treat them as a current prospect.
        if "draft_year" in df.columns:
            mask = (df["draft_year"] >= 2022) | df["draft_year"].isna()
            df = df[mask]

    if draft_year and draft_year != "All":
        try:
            df = df[df["draft_year"] == int(draft_year)]
        except Exception:
            pass

    if positions:
        df = df[df["position"].isin(positions)]

    if search:
        q = search.strip().lower()
        if q:
            df = df[df["name"].astype(str).str.lower().str.contains(q, na=False)]

    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ── detail drawer (composable card pieces) ───────────────────────────────────

def render_card_header(player_row: pd.Series, status: str, accent: str):
    style = STATUS_STYLES.get(status, STATUS_STYLES["Developing"])
    flag = FLAG_EMOJI.get((player_row.get("nationality") or "").upper(), "🏒")
    name = player_row.get("name", "Unknown")
    pos = (player_row.get("position") or "").upper()[:1] or "—"

    # Accent bar
    st.markdown(
        f'<div style="height:6px;background:{accent};border-radius:2px;margin-bottom:8px;"></div>',
        unsafe_allow_html=True,
    )

    headshot = headshot_url(str(player_row.get("player_id", "")))
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image(headshot, width=90)
    with c2:
        st.markdown(
            f'<div style="font-size:22px;font-weight:800;color:#1B2631;line-height:1.1;">{name}</div>'
            f'<div style="margin-top:4px;font-size:13px;color:#566573;">'
            f'{flag} <span style="color:{style["color"]};font-weight:700;">'
            f'{style["icon"]} {status}</span>'
            f' &nbsp;|&nbsp; <b>{pos}</b></div>',
            unsafe_allow_html=True,
        )


def render_card_bio(player_row: pd.Series, seasons_sub: pd.DataFrame,
                    outcome_row: pd.Series | None):
    gp = int(seasons_sub["gp"].sum()) if not seasons_sub.empty else 0
    assists = int(seasons_sub["assists"].sum()) if not seasons_sub.empty else 0
    points = int(seasons_sub["points"].sum()) if not seasons_sub.empty else 0
    ppg = round(points / gp, 2) if gp else 0

    def _iget(field):
        # Prefer players-table value; fall back to outcomes if present
        v = player_row.get(field)
        if (v is None or pd.isna(v)) and outcome_row is not None:
            v = outcome_row.get(field)
        return v if pd.notna(v) and v not in (None, 0) else None

    dy_val = _iget("draft_year")
    rnd_val = _iget("draft_round")
    pick_val = _iget("draft_pick")
    team_val = player_row.get("draft_team")

    dy = int(dy_val) if dy_val else PLACEHOLDER
    rnd = int(rnd_val) if rnd_val else PLACEHOLDER
    pick = int(pick_val) if pick_val else PLACEHOLDER
    team = team_val if team_val and pd.notna(team_val) else PLACEHOLDER

    bio = (
        f'<div style="font-size:12px;color:#2C3E50;line-height:1.8;margin-top:10px;">'
        f'<b>GP:</b> {gp} | <b>A:</b> {assists} | <b>Pts:</b> {points} | <b>PPG:</b> {ppg}<br>'
        f'<b>Born:</b> {format_born(player_row.get("dob"))} | '
        f'<b>Age:</b> {calc_age(player_row.get("dob")) or PLACEHOLDER}<br>'
        f'<b>H:</b> {height_display(player_row.get("height_cm"))} | '
        f'<b>W:</b> {weight_display(player_row.get("weight_kg"))} | '
        f'<b>Shoots:</b> {player_row.get("shoots") or PLACEHOLDER}<br>'
        f'<b>DY:</b> {dy} | <b>Round:</b> {rnd} | <b>Pick:</b> {pick} | <b>Team:</b> {team}'
        f'</div>'
    )
    st.markdown(bio, unsafe_allow_html=True)


def render_card_charts(player_row: pd.Series, seasons_sub: pd.DataFrame,
                        outcome_row: pd.Series | None,
                        pred_row: pd.Series | None, accent: str):
    st.markdown(
        '<div style="margin-top:14px;font-size:12px;font-weight:700;color:#1B2631;">NHLe (pts / 82 games)</div>',
        unsafe_allow_html=True,
    )
    draft_year = None
    if outcome_row is not None and pd.notna(outcome_row.get("draft_year")):
        draft_year = int(outcome_row["draft_year"])
    elif pd.notna(player_row.get("draft_year")):
        draft_year = int(player_row["draft_year"])

    stages_df = get_dev_stages(seasons_sub, draft_year)
    if not stages_df.empty:
        st.plotly_chart(
            nhle_bar_chart(stages_df, accent),
            use_container_width=True,
            config={"displayModeBar": False},
            key=f"nhle_{player_row['player_id']}",
        )
    else:
        st.caption("No season data available.")

    probs = get_stage_probs(seasons_sub, outcome_row, pred_row)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(prob_row_html(probs, "star", "Star Probabilities"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(prob_row_html(probs, "nhler", "NHLer Probabilities"),
                    unsafe_allow_html=True)


def render_card_comps(dy_comps: pd.DataFrame, full_comps: pd.DataFrame):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(comps_table_html("DY Comps", dy_comps), unsafe_allow_html=True)
    with d2:
        st.markdown(comps_table_html("Full Comps", full_comps), unsafe_allow_html=True)


def render_drawer(player_id: str, players: pd.DataFrame, seasons: pd.DataFrame,
                  outcomes: pd.DataFrame, predictions: pd.DataFrame,
                  comp_index):
    """Render the full detail card for a single selected player."""
    match = players[players["player_id"] == player_id]
    if match.empty:
        st.info("Selected player not found.")
        return
    player_row = match.iloc[0]
    seasons_sub = seasons[seasons["player_id"] == player_id].copy()

    out_match = outcomes[outcomes["player_id"] == player_id] if not outcomes.empty else pd.DataFrame()
    outcome_row = out_match.iloc[0] if not out_match.empty else None

    pred_match = predictions[predictions["player_id"] == player_id] if not predictions.empty else pd.DataFrame()
    pred_row = pred_match.iloc[0] if not pred_match.empty else None

    status = status_for(player_id, outcomes, predictions)
    accent = CARD_ACCENTS.get(status, "#1F618D")

    # Comparables (optional — only if index is built)
    dy_comps = full_comps = pd.DataFrame()
    if comp_index is not None:
        feat_row = get_feature_row(players, seasons, outcomes, player_id)
        if feat_row is not None:
            full_comps = comp_index.find_comparables(feat_row, n=5,
                                                     same_draft_year_only=False)
            target_dy = player_row.get("draft_year")
            if (target_dy is None or pd.isna(target_dy)) and outcome_row is not None:
                target_dy = outcome_row.get("draft_year")
            if pd.notna(target_dy):
                feat_row_dy = feat_row.copy()
                feat_row_dy["draft_year"] = target_dy
                dy_comps = comp_index.find_comparables(feat_row_dy, n=5,
                                                       same_draft_year_only=True)

    render_card_header(player_row, status, accent)
    render_card_bio(player_row, seasons_sub, outcome_row)
    render_card_charts(player_row, seasons_sub, outcome_row, pred_row, accent)
    render_card_comps(dy_comps, full_comps)


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

BOARD_COLS_VISIBLE = [
    "rank", "headshot_url", "name", "position", "draft_display",
    "nhler_pct", "star_pct", "proj_pts", "nhle_ppg", "status_display",
]


def main():
    st.set_page_config(
        page_title="Prospect Board",
        page_icon="🏒",
        layout="wide",
    )
    st.title("🏒 NHL Prospect Board")
    st.caption("Click a row to view the full scouting card on the right.")

    players, seasons, outcomes, predictions = load_all()
    if players.empty:
        st.error("No player data found. Run `python main.py collect` first.")
        return

    # ── Sidebar filters ──────────────────────────────────────────
    st.sidebar.header("Filters")

    all_years = sorted(
        {int(y) for y in players["draft_year"].dropna().unique() if y >= 2022},
        reverse=True,
    )
    years_opt = ["All"] + [str(y) for y in all_years]
    year_filter = st.sidebar.selectbox("Draft year", years_opt, index=0)

    all_positions = [
        p for p in ["F", "D", "G"] if p in set(players["position"].dropna())
    ]
    pos_filter = st.sidebar.multiselect("Position", all_positions, default=[])

    search = st.sidebar.text_input("Search by name", "")

    include_settled = st.sidebar.checkbox(
        "Include veterans with settled careers", value=False,
        help="Off by default — board focuses on current draft-eligible prospects.",
    )

    comp_index = get_comp_index(players, seasons, outcomes)
    if comp_index is None:
        st.sidebar.caption("⚠️ Comparables unavailable (no trained model)")

    # ── Build + filter board ─────────────────────────────────────
    board = build_board(players, seasons, outcomes, predictions)
    view = filter_board(
        board,
        draft_year=year_filter,
        positions=pos_filter or None,
        search=search or None,
        include_settled=include_settled,
    )

    # Summary chip
    st.markdown(
        f'<div style="margin:4px 0 12px 0;color:#566573;font-size:13px;">'
        f'Showing <b>{len(view)}</b> prospects · '
        f'Click any row to open the scouting card.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Two-column layout: board + detail drawer ─────────────────
    board_col, drawer_col = st.columns([3, 2], gap="large")

    with board_col:
        if view.empty:
            st.info("No prospects match the current filters.")
            selected_pid = None
        else:
            event = st.dataframe(
                view[BOARD_COLS_VISIBLE],
                hide_index=True,
                use_container_width=True,
                height=min(650, 60 + 36 * min(len(view), 18)),
                selection_mode="single-row",
                on_select="rerun",
                column_config={
                    "rank":          st.column_config.NumberColumn("#", width="small"),
                    "headshot_url":  st.column_config.ImageColumn(" ", width="small"),
                    "name":          st.column_config.TextColumn("Name", width="medium"),
                    "position":      st.column_config.TextColumn("Pos", width="small"),
                    "draft_display": st.column_config.TextColumn("Draft", width="small"),
                    "nhler_pct":     st.column_config.ProgressColumn(
                        "NHLer %", format="%d%%", min_value=0, max_value=100),
                    "star_pct":      st.column_config.ProgressColumn(
                        "Star %", format="%d%%", min_value=0, max_value=100),
                    "proj_pts":      st.column_config.NumberColumn("Proj Pts", width="small"),
                    "nhle_ppg":      st.column_config.NumberColumn(
                        "NHLe", format="%.2f", width="small"),
                    "status_display":st.column_config.TextColumn("Status", width="medium"),
                },
                key="board_table",
            )
            sel = event.selection.rows if event and event.selection else []
            if sel:
                selected_pid = view.iloc[sel[0]]["player_id"]
            else:
                selected_pid = view.iloc[0]["player_id"]  # default to #1

    with drawer_col:
        if view.empty:
            pass
        elif selected_pid:
            render_drawer(selected_pid, players, seasons, outcomes,
                          predictions, comp_index)
        else:
            st.info("Click a row on the left to open the scouting card.")

    render_legend()


if __name__ == "__main__":
    main()
