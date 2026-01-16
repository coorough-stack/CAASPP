# app.py — single-file Streamlit app for CAASPP one-pagers
from __future__ import annotations
import os, io
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from textwrap import wrap

def draw_bottom_text(ax, blurb: str, yby_lines: list[str], *, fontsize=11):
    """Render 'What this means' and 'Year-by-Year' with robust spacing."""
    ax.axis('off')
    LINE = 0.052    # vertical step in axes coords (tweak 0.048–0.056 to taste)
    GAP  = 0.018    # extra gap between blocks
    y = 0.96

    # Heading 1
    ax.text(0.0, y, "What this means", transform=ax.transAxes,
            ha='left', va='top', fontsize=fontsize, fontweight='bold')
    y -= LINE

    # Wrapped blurb
    for t in wrap(blurb or "", width=90):
        ax.text(0.0, y, t, transform=ax.transAxes,
                ha='left', va='top', fontsize=fontsize)
        y -= LINE

    y -= GAP

    # Heading 2
    ax.text(0.0, y, "Year-by-Year Performance", transform=ax.transAxes,
            ha='left', va='top', fontsize=fontsize, fontweight='bold')
    y -= LINE

    # Lines
    for t in (yby_lines or []):
        ax.text(0.0, y, t, transform=ax.transAxes,
                ha='left', va='top', fontsize=fontsize)
        y -= LINE

# ---- Flexible reader for raw CAASPP exports ----
def _assemble_student_name(df: pd.DataFrame) -> pd.Series:
    if "Student" in df.columns:
        return df["Student"].astype(str)
    if {"First Name", "Last Name"}.issubset(df.columns):
        return (df["First Name"].astype(str).str.strip() + " " +
                df["Last Name"].astype(str).str.strip()).str.strip()
    if "Name" in df.columns:  # try "Last, First"
        parts = df["Name"].astype(str).str.split(",", n=1, expand=True)
        if parts.shape[1] == 2:
            return (parts[1].str.strip() + " " + parts[0].str.strip()).str.strip()
        return df["Name"].astype(str)
    # fallback
    return pd.Series(["Student"] * len(df), index=df.index)

def read_scores_flex(uploaded_file, subject: str, swap_sbac_parts: bool = False) -> pd.DataFrame:
    """
    Accepts either:
      A) already-normalized scores (Score_30..Score_80), or
      B) raw CAASPP export like the one you posted (with TESTID/Part/Grade1/S/S)
    Returns a DataFrame with: Student ID, Student, Current Grade, Score_30..Score_80
    """
    # try utf-8 then latin-1
    try:
        raw = pd.read_csv(uploaded_file, encoding="utf-8", engine="python")
    except Exception:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, encoding="latin-1", engine="python")

    raw.columns = [c.strip().replace("\ufeff", "") for c in raw.columns]

    # Case A: already normalized (has Score_* columns) -> just ensure name & current grade
    if any(str(c).startswith("Score_") for c in raw.columns):
        if "Student" not in raw.columns:
            raw["Student"] = _assemble_student_name(raw)
        if "Current Grade" not in raw.columns and "Grade" in raw.columns:
            raw["Current Grade"] = raw["Grade"]
        return raw

    # Case B: raw CAASPP export (TESTID/Part/Grade1/S/S)
    needed = {"TESTID", "Part", "Grade1", "S/S"}
    if not needed.issubset(set(raw.columns)):
        # If it’s some other layout, leave as-is (app will show column prompts)
        return raw

    # Keep SBAC only (drop ELPAC/CAST/CAA/etc.)
    sbac = raw[raw["TESTID"].astype(str).str.upper() == "SBAC"].copy()
    sbac = sbac[pd.notna(sbac["S/S"]) & pd.notna(sbac["Grade1"])]

    # Subject selection via Part mapping (default: Part 1 = ELA, Part 2 = Math)
    # If your district uses the opposite, tick the swap box (see sidebar toggle below).
    part_math, part_ela = (1.0, 2.0) if swap_sbac_parts else (2.0, 1.0)
    target_part = part_math if subject == "Math" else part_ela
    sbac = sbac[sbac["Part"] == target_part].copy()

    # Keep only grades 30..80 step 10 and take the latest test per student/grade
    sbac["Grade1"] = sbac["Grade1"].astype(float)
    sbac = sbac[sbac["Grade1"].isin([30, 40, 50, 60, 70, 80])]
    if "Date Taken" in sbac.columns:
        sbac["_date"] = pd.to_datetime(sbac["Date Taken"], errors="coerce")
        sbac = sbac.sort_values("_date").drop_duplicates(["Student ID", "Grade1"], keep="last")

    # Pivot to wide: Student ID × Score_{Grade1} = S/S
    sbac["ScoreCol"] = "Score_" + sbac["Grade1"].astype(int).astype(str)
    wide = sbac.pivot_table(index=["Student ID"], columns="ScoreCol", values="S/S", aggfunc="first").reset_index()

    # Attach Student and (if present) Current Grade
    raw["Student"] = _assemble_student_name(raw)
    if "Current Grade" not in raw.columns and "Grade" in raw.columns:
        raw["Current Grade"] = raw["Grade"]

    id_cols = ["Student ID", "Student"] + (["Current Grade"] if "Current Grade" in raw.columns else [])
    base = raw.drop_duplicates("Student ID")[id_cols]
    out = base.merge(wide, on="Student ID", how="right")

    # Ensure all expected Score_* exist
    for g in (30, 40, 50, 60, 70, 80):
        col = f"Score_{g}"
        if col not in out.columns:
            out[col] = np.nan

    # Make sure Student ID is string for safer joins later
    out["Student ID"] = out["Student ID"].astype("string")
    return out

def normalize_current_grade(df: pd.DataFrame, col: str = "Current Grade") -> pd.DataFrame:
    """Coerce 'Current Grade' to integers 3..8.
       Accepts values like '6', '6th', 'Grade 6', or 30/40/… (→ 3..8)."""
    if col not in df.columns:
        return df

    s = df[col].astype(str)

    # pull first number we see (e.g., "6th" -> 6, "Grade 7" -> 7)
    num = pd.to_numeric(s.str.extract(r"(\d+)", expand=False), errors="coerce")

    # if values look like 30/40/… (Grade1 style), convert to 3..8
    num = np.where(num >= 30, np.round(num / 10.0), num)

    # clamp to 3..8 and set back
    out = pd.Series(num, index=df.index).round()
    df[col] = pd.to_numeric(out, errors="coerce").clip(lower=3, upper=8).astype("Int64")
    return df

def _parse_grade_any(x):
    """Return an integer 3..8 if possible from values like '6', '6th', 'Grade 7', 30/40/… etc."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    # pull first number
    import re
    m = re.search(r"\d+", s)
    if not m:
        return pd.NA
    n = pd.to_numeric(m.group(), errors="coerce")
    if pd.isna(n):
        return pd.NA
    # Grade1-style 30/40/... -> 3..8
    if 30 <= n <= 80:
        n = int(round(n / 10.0))
    return int(n) if 3 <= int(n) <= 8 else pd.NA

def ensure_current_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees df['Current Grade'] exists as Int64 with values 3..8 by deriving from:
    - Current Grade (any format), or
    - Grade / Grade Level / Grade1 (30/40/... ok), or
    - LatestGradeTested, or
    - The latest non-null Score_* column present.
    """
    df = df.copy()

    # 1) start with existing Current Grade if present
    cg = None
    if "Current Grade" in df.columns:
        cg = df["Current Grade"].apply(_parse_grade_any)

    # 2) fallbacks from common export columns
    for col in ["Grade", "Grade Level", "Grade1"]:
        if cg is None or cg.isna().all():
            if col in df.columns:
                cg = pd.Series([_parse_grade_any(v) for v in df[col]], index=df.index)

    # 3) fallback from LatestGradeTested (already 3..8 usually)
    if cg is None or cg.isna().all():
        if "LatestGradeTested" in df.columns:
            cg = df["LatestGradeTested"].apply(_parse_grade_any)

    # 4) last resort: infer from which Score_* exists (take max present → “current”)
    if cg is None or cg.isna().all():
        def _infer_from_scores(row):
            present = [g for g in range(3,9) if pd.notna(row.get(f"Score_{g*10}", pd.NA))]
            return pd.NA if not present else max(present)
        cg = df.apply(_infer_from_scores, axis=1)

    # finalize
    cg = pd.to_numeric(cg, errors="coerce").astype("Int64")
    df["Current Grade"] = cg
    return df

def detect_section_columns(df: pd.DataFrame) -> dict:
    """Detects Student ID and Section columns in a sections-enrollment CSV."""
    def _find(possibles):
        # exact
        for p in possibles:
            for c in df.columns:
                if c.strip().lower() == p:
                    return c
        # contains
        for p in possibles:
            for c in df.columns:
                if p in c.strip().lower():
                    return c
        return None

    sid = _find(["student id","studentid","id","stu#","stu #","local id","state id"])
    section = _find(["section","section id","section code","course section","period","class","class id"])
    return {"student_id": sid, "section": section}


def apply_section_filter(_df: pd.DataFrame, selected_sections: list) -> pd.DataFrame:
    """Keep rows where student's SectionsList intersects selected_sections."""
    if not selected_sections or "SectionsList" not in _df.columns:
        return _df
    sel = set(map(str, selected_sections))
    return _df[_df["SectionsList"].apply(lambda lst: bool(set(map(str, (lst or []))).intersection(sel)))]

# ---- Design constants ----
ISR_COLORS = {1: "#F35031", 2: "#F2C94F", 3: "#5AC923", 4: "#00B4EB"}
BAND_ALPHA = 0.18
TREND_LINE = "#3A3A3A"
POINT_EDGE = "black"

# Thresholds (grades 3–8)
MATH_THRESHOLDS = {
    3: (2381, 2436, 2501),
    4: (2411, 2485, 2549),
    5: (2455, 2528, 2579),
    6: (2473, 2552, 2610),
    7: (2484, 2567, 2635),
    8: (2504, 2586, 2653),
}
ELA_THRESHOLDS = {
    3: (2367, 2432, 2490),
    4: (2416, 2473, 2533),
    5: (2442, 2502, 2582),
    6: (2457, 2531, 2618),
    7: (2479, 2552, 2649),
    8: (2487, 2567, 2668),
}
THRESHOLDS = {"Math": MATH_THRESHOLDS, "ELA": ELA_THRESHOLDS}

SCORE_COLS = [f"Score_{g*10}" for g in range(3, 9)]  # Score_30..Score_80
STUDENT_ID_COL = "Student ID"
STUDENT_NAME_COL = "Student"
CURRENT_GRADE_COL = "Current Grade"

# ---- Utilities ----
def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def thresholds_for(subject: str, grade: int) -> Tuple[int, int, int]:
    table = THRESHOLDS[subject]
    grade = int(grade) if pd.notna(grade) else 8
    if grade not in table:
        grade = max(3, min(8, grade))
    return table[grade]

def level_name(level: int) -> str:
    return {1: "Not Met", 2: "Nearly Met", 3: "Met", 4: "Exceeded"}.get(level, "—")

def score_to_level(score: float, subject: str, grade: int) -> int:
    if pd.isna(score):
        return 0
    L2, L3, L4 = thresholds_for(subject, grade)
    if score < L2:
        return 1
    if score < L3:
        return 2
    if score < L4:
        return 3
    return 4

def points_to_next_level(score: float, subject: str, grade: int) -> int:
    if pd.isna(score):
        return np.nan
    L2, L3, L4 = thresholds_for(subject, grade)
    lvl = score_to_level(score, subject, grade)
    if lvl == 1:
        return int(L2 - score)
    if lvl == 2:
        return int(L3 - score)
    if lvl == 3:
        return int(L4 - score)
    return 0

def _latest_score_and_grade(row: pd.Series) -> Tuple[float, Optional[int]]:
    for g in range(8, 2, -1):
        col = f"Score_{g*10}"
        if col in row.index and pd.notna(row[col]):
            return row[col], g
    return (np.nan, None)

def ensure_name_column(df: pd.DataFrame) -> pd.DataFrame:
    if STUDENT_NAME_COL in df.columns:
        return df
    last = None
    first = None
    for c in df.columns:
        lc = c.lower()
        if "last" in lc and "name" in lc:
            last = c
        if "first" in lc and "name" in lc:
            first = c
    if last is not None and first is not None:
        df[STUDENT_NAME_COL] = (df[first].astype(str).str.strip() + " " +
                                df[last].astype(str).str.strip()).str.strip()
    else:
        df[STUDENT_NAME_COL] = "Student"
    return df

# ---- Augmentation (percentile, latest, etc.) ----
def attach_latest_and_percentiles(df: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Add LatestScore, LatestGradeTested, SchoolPercentile(+ordinal),
    and compute LatestLevel / PtsToNextLevel using the GRADE TESTED."""
    df = df.copy()
    df = ensure_name_column(df)

    # latest score + the grade it was tested in
    lat_scores, lat_grades = [], []
    for _, r in df.iterrows():
        s, g = _latest_score_and_grade(r)
        lat_scores.append(s)
        lat_grades.append(g)
    df["LatestScore"] = lat_scores
    df["LatestGradeTested"] = lat_grades

    if CURRENT_GRADE_COL not in df.columns:
        df[CURRENT_GRADE_COL] = df["LatestGradeTested"]

    # Percentile within CURRENT grade (for communication)
    df["SchoolPercentile"] = np.nan
    for g, sub in df.groupby(df[CURRENT_GRADE_COL]):
        if sub["LatestScore"].notna().sum() == 0:
            continue
        pct = sub["LatestScore"].rank(method="average", pct=True)
        df.loc[sub.index, "SchoolPercentile"] = (pct * 100).round(0).astype("Int64")

    df["SchoolPercentileOrdinal"] = df["SchoolPercentile"].apply(
        lambda x: f"{ordinal(int(x))} percentile" if pd.notna(x) else ""
    )

    # Use grade tested first; fall back to current grade; else 8
    def _grade_for_level(g_tested, g_current):
        if pd.notna(g_tested):
            return int(g_tested)
        if pd.notna(g_current):
            return int(g_current)
        return 8

    df["LatestLevel"] = [
        score_to_level(s, subject, _grade_for_level(g, cg))
        for s, g, cg in zip(df["LatestScore"], df["LatestGradeTested"], df[CURRENT_GRADE_COL])
    ]
    df["PtsToNextLevel"] = [
        points_to_next_level(s, subject, _grade_for_level(g, cg))
        for s, g, cg in zip(df["LatestScore"], df["LatestGradeTested"], df[CURRENT_GRADE_COL])
    ]
    return df

# ---- Sorting / filtering ----
def filter_by_grade_and_level(df: pd.DataFrame,
                              grades: Optional[List[int]],
                              levels: Optional[List[int]],
                              hide_lvl4: bool) -> pd.DataFrame:
    out = df.copy()
    if grades:
        out = out[out[CURRENT_GRADE_COL].isin(grades)]
    if levels:
        out = out[out["LatestLevel"].isin(levels)]
    if hide_lvl4:
        out = out[out["LatestLevel"] != 4]
    return out

def sort_df(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "As in file":
        return df.sort_values("_file_order", kind="stable")
    if how == "Teacher > Class > Student":
        cols = [c for c in ["Teacher","Class",STUDENT_NAME_COL] if c in df.columns]
        if cols:
            return df.sort_values(cols, kind="stable")
        else:
            return df.sort_values([STUDENT_NAME_COL], kind="stable")
    if how == "Current Grade (desc)":
        return df.sort_values([CURRENT_GRADE_COL, STUDENT_NAME_COL], ascending=[False, True], kind="stable")
    if how == "Percentile (desc)":
        return df.sort_values(["SchoolPercentile", STUDENT_NAME_COL], ascending=[False, True], kind="stable")
    if how == "Closest to Next Level (asc)":
        prox = df["PtsToNextLevel"].fillna(9999).replace(0, 9999)
        return df.assign(_prox=prox).sort_values(["_prox", STUDENT_NAME_COL], kind="stable").drop(columns="_prox")
    return df

# ---- Column detection & ID normalization ----
def detect_sid_column(df: pd.DataFrame) -> Optional[str]:
    lc = [c.strip().lower() for c in df.columns]
    for target in ["student id","studentid","id","stu#","stu #"]:
        for i, c in enumerate(lc):
            if c == target:
                return df.columns[i]
    # fallback: exact match "Student ID"
    for c in df.columns:
        if c.strip().lower() == "student id":
            return c
    return None

def detect_roster_columns(roster: pd.DataFrame) -> Dict[str, Optional[str]]:
    def find_one(possibles):
        # exact
        for p in possibles:
            for c in roster.columns:
                if c.strip().lower() == p:
                    return c
        # contains
        for p in possibles:
            for c in roster.columns:
                if p in c.strip().lower():
                    return c
        return None
    sid = find_one(["student id","studentid","id","stu#","stu #"])
    teacher = find_one(["teacher","instructor","advisor"])
    klass = find_one(["class","period","section","course"])
    return {"student_id": sid, "teacher": teacher, "class": klass}

def sid_norm_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.normalize("NFKC")
         .str.replace(r"\s+", "", regex=True)
         .str.replace("\u00A0", "", regex=False)
         .str.replace(r"[^\w]", "", regex=True)
         .str.upper()
    )

# ---- In-app HTML helpers ----
def level_pill_html(level: int, text: Optional[str] = None) -> str:
    style_base = (
        "display:inline-block;font-size:13px;line-height:1;"
        "padding:2px 8px;border-radius:9999px;font-weight:700;"
        "box-shadow:inset 0 0 0 1px rgba(0,0,0,.15)"
    )
    if not level:
        return f'<span style="{style_base};background:#eee;color:#333;">N/A</span>'
    col = ISR_COLORS[level]
    label = text or f"Level {level} ({level_name(level)})"
    return f'<span style="{style_base};background:{col};color:#000;">{label}</span>'

def level_key_inline(subject: str, grade: int) -> str:
    L2, L3, L4 = thresholds_for(subject, grade)
    l2_hi = L3 - 1
    l3_hi = L4 - 1
    parts = [
        (f"<b>L1</b> < {L2}", 1),
        (f"<b>L2</b> {L2}–{l2_hi}", 2),
        (f"<b>L3</b> {L3}–{l3_hi}", 3),
        (f"<b>L4</b> ≥ {L4}", 4),
    ]
    spans = []
    for text, lvl in parts:
        spans.append(
            f'<span style="background:{ISR_COLORS[lvl]}33; padding:4px 8px; border-radius:6px; '
            f'display:inline-block; margin-right:8px; box-shadow: inset 0 0 0 1px rgba(0,0,0,.08)">{text}</span>'
        )
    return f'<div style="font-size:12px;margin-top:6px;">Level key (G{grade}): ' + "  •  ".join(spans) + "</div>"

# ---- PDF key (Matplotlib) ----
def _hex_with_alpha(hex_color: str, alpha: float):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)

def draw_level_key(ax, subject: str, grade: int):
    ax.clear()
    ax.axis("off")

    L2, L3, L4 = thresholds_for(subject, int(grade))
    l2_hi = L3 - 1
    l3_hi = L4 - 1
    labels = [
        (r"$\mathbf{L1}$" + f" < {L2}", 1),
        (r"$\mathbf{L2}$" + f" {L2}–{l2_hi}", 2),
        (r"$\mathbf{L3}$" + f" {L3}–{l3_hi}", 3),
        (r"$\mathbf{L4}$" + f" ≥ {L4}", 4),
    ]

    ax.text(0.02, 0.50, f"Level key (G{grade}):",
            transform=ax.transAxes, ha="left", va="center", fontsize=9)

    start_x   = 0.19
    gap       = 0.018

    def _clean_len(s: str) -> int:
        return len(s.replace(r"$\mathbf{", "").replace("}$", "").replace("$", ""))

    lengths = [_clean_len(t) for t, _ in labels]
    total_len = sum(lengths)
    total_gaps = (len(labels) - 1) * (2 * gap)
    end_x = 0.98
    avail = max(0.05, end_x - start_x - total_gaps)

    char_w = avail / max(1, total_len)
    char_w = min(0.011, max(0.0075, char_w))

    x = start_x
    for i, (text, lvl) in enumerate(labels):
        ax.text(
            x, 0.50, text,
            transform=ax.transAxes, ha="left", va="center", fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.28,rounding_size=0.18",
                facecolor=_hex_with_alpha(ISR_COLORS[lvl], 0.25),
                edgecolor="black", linewidth=0.5
            ),
        )
        x += char_w * _clean_len(text)
        if i < len(labels) - 1:
            x += gap
            ax.text(x, 0.50, "•", transform=ax.transAxes, ha="left", va="center",
                    fontsize=11, color="#666666")
            x += gap

# ---- Visual builders ----
def build_trend_figure(row: pd.Series, subject: str, dpi: int = 160, ax=None):
    pts = []
    for g in range(3, 9):
        col = f"Score_{g*10}"
        if col in row.index and pd.notna(row[col]):
            pts.append((g, float(row[col])))
    grades = [g for g, _ in pts]
    scores = [s for _, s in pts]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=dpi)
        created_fig = True
    else:
        ax.clear()
        fig = ax.figure

    ax.set_title(f"CAASPP {subject} Scores by Grade", pad=8, fontsize=12, weight="bold")

    if grades:
        data_min, data_max = min(scores), max(scores)
        pad = max(60, (data_max - data_min) * 0.25)
        ylo, yhi = data_min - pad, data_max + pad

        grade_gap = 0.08
        w = 1.0 - grade_gap
        for g in range(min(grades), max(grades) + 1):
            L2, L3, L4 = thresholds_for(subject, g)
            x0 = g - w / 2
            for y0, y1, col in [(ylo, L2, ISR_COLORS[1]),
                                (L2, L3, ISR_COLORS[2]),
                                (L3, L4, ISR_COLORS[3]),
                                (L4, yhi, ISR_COLORS[4])]:
                ax.add_patch(Rectangle((x0, y0), w, y1 - y0,
                                       facecolor=col, alpha=BAND_ALPHA,
                                       edgecolor=None, zorder=0))
    else:
        ylo, yhi = (2300, 2700)

    if len(pts) >= 2:
        ax.plot(grades, scores, color=TREND_LINE, linewidth=1.8, zorder=1)

    # slope-aware labels with simple de-collision
    label_positions = []
    fig = ax.figure
    pix_per_pt = fig.dpi / 72.0
    y0 = ax.get_ylim()[0]
    p0 = ax.transData.transform((0, y0))
    p1 = ax.transData.transform((0, y0 + 1))
    pixels_per_data_unit_y = abs(p1[1] - p0[1]) if abs(p1[1] - p0[1]) > 0 else 1.0
    data_per_point = 1.0 / (pixels_per_data_unit_y * pix_per_pt)
    offset_candidates_pts = [10, 14, 18, 22, 26]
    near_x = 0.5
    near_y = 20

    for i, (g, s) in enumerate(pts):
        lvl = score_to_level(s, subject, g)
        ax.scatter([g], [s], s=56, c=[ISR_COLORS[lvl]], edgecolors=POINT_EDGE, linewidths=1.2, zorder=3)

        label = f"Latest (G{g}): {int(round(s))}" if i == len(pts) - 1 else f"G{g}: {int(round(s))}"

        if len(pts) == 1:
            sign = +1
        elif i == 0:
            sign = -1 if (pts[i+1][1] - s) > 0 else +1
        elif i == len(pts) - 1:
            sign = -1 if (s - pts[i-1][1]) > 0 else +1
        else:
            sign = -1 if (pts[i+1][1] - pts[i-1][1]) > 0 else +1

        ylim = ax.get_ylim()
        chosen = offset_candidates_pts[-1]
        y_label = s + sign * chosen * data_per_point
        for cand in offset_candidates_pts:
            y_try = s + sign * cand * data_per_point
            if not (ylim[0] + 8 <= y_try <= ylim[1] - 8):
                continue
            ok = True
            for (xj, yj) in label_positions:
                if abs(g - xj) < near_x and abs(y_try - yj) < near_y:
                    ok = False
                    break
            if ok:
                chosen = cand
                y_label = y_try
                break

        va = "bottom" if sign > 0 else "top"
        ax.annotate(label, (g, s), xytext=(0, sign * chosen), textcoords="offset points",
                    ha="center", va=va, fontsize=9, color="black", zorder=4)
        label_positions.append((g, y_label))

    if grades:
        ax.set_xlim(min(grades) - 0.6, max(grades) + 0.6)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(range(3, 9))
    ax.set_xlabel("Grade Tested")
    ax.set_ylabel("Scale score")
    ax.grid(axis="y", color="#DDDDDD", alpha=0.7, linestyle="--", linewidth=0.8)

    if created_fig:
        fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.16)
        return fig
    else:
        return ax

def build_growth_figure(row: pd.Series, subject: str, dpi: int = 160, ax=None):
    seq = [(g, float(row[f"Score_{g*10}"])) for g in range(3,9)
           if f"Score_{g*10}" in row.index and pd.notna(row[f"Score_{g*10}"])]
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 2.2), dpi=dpi)
        created_fig = True
    else:
        ax.clear()
        fig = ax.figure

    if len(seq) < 2:
        ax.text(0.5, 0.5, "No year-over-year growth data yet.", ha="center", va="center")
        ax.axis("off")
        if created_fig:
            fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.28)
            fig.tight_layout()
            return fig
        return ax

    labels, deltas = [], []
    for i in range(1, len(seq)):
        g0, s0 = seq[i-1]; g1, s1 = seq[i]
        labels.append(f"{g0}→{g1}")
        deltas.append(int(round(s1 - s0)))

    x = np.arange(len(deltas))
    colors = ["#2e7d32" if d >= 0 else "#c62828" for d in deltas]
    bars = ax.bar(x, deltas, color=colors, alpha=0.70)
    ax.axhline(0, color="#444", linewidth=1)

    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, (h/2 if h!=0 else 0.2),
                f"{'+' if deltas[i]>=0 else ''}{deltas[i]}", ha="center", va="center",
                fontsize=9, color="black")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Points")
    ax.set_title("Year-over-Year Growth", fontsize=11, pad=6)
    ax.grid(axis="y", color="#E0E0E0", linestyle="--", alpha=0.7)

    if created_fig:
        fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.28)
        fig.tight_layout()
        return fig
    else:
        return ax

# ---- App UI ----
st.set_page_config(page_title="CAASPP Student One-Pagers", layout="wide")

# Sidebar
st.sidebar.header("One-Pagers • CAASPP")
subject = st.sidebar.radio("Subject", ["Math","ELA"], horizontal=True, key="subject_radio_single")
school_name = st.sidebar.text_input("School Name", value="CK Price", key="school_name_single")
scores_up = st.sidebar.file_uploader("Load Student Scores CSV", type=["csv"], key="scores_csv_single")

st.sidebar.markdown("---")
grades_filter = st.sidebar.multiselect("Filter by Current Grade", [3,4,5,6,7,8], default=[], key="grades_filter_single")
levels_filter = st.sidebar.multiselect("Filter by Level", [1,2,3,4], default=[], key="levels_filter_single")
hide_lvl4 = st.sidebar.checkbox("Hide Level 4", value=False, key="hide_lvl4_single")

sort_choice = st.sidebar.selectbox("Sort by", [
    "As in file",
    "Teacher > Class > Student",
    "Current Grade (desc)",
    "Percentile (desc)",
    "Closest to Next Level (asc)",
], key="sort_choice_single")

st.sidebar.markdown("---")
pick_mode = st.sidebar.radio("Selection", ["All filtered students", "First N only"], horizontal=False, key="pick_mode_single")
first_n = st.sidebar.number_input("N", min_value=1, max_value=200, value=10, step=1, disabled=(pick_mode=='All filtered students'), key="first_n_single")
st.sidebar.markdown("---")
if st.sidebar.button("Shutdown server", type="primary", key="shutdown_btn_single"):
    st.sidebar.warning("Server shutting down…")
    os._exit(0)

# SBAC part mapping toggle
swap_parts = st.sidebar.checkbox(
    "SBAC Part mapping: Part 1=Math, Part 2=ELA (default is Part 1=ELA, Part 2=Math)",
    value=False,
    key="swap_parts_single"
)


# Scores CSV required
if not scores_up:
    st.title(f"{school_name} — {subject} CAASPP Test Scores")
    st.info("Upload your Student Scores CSV to begin.")
    st.stop()

# Read scores (flexible) + ensure clean Current Grade
df = read_scores_flex(scores_up, subject, swap_sbac_parts=swap_parts)
df["_file_order"] = np.arange(len(df))

# First pass to compute LatestGradeTested/etc.
df = attach_latest_and_percentiles(df, subject)

# Coerce/derive Current Grade to 3..8, then recompute levels/percentiles using the clean grade
df = ensure_current_grade(df)
df = attach_latest_and_percentiles(df, subject)

# Detect Student ID column (scores)
sid_col = detect_sid_column(df)
if not sid_col:
    with st.expander("Select Student ID column in Scores CSV (auto-detect failed)"):
        sid_col = st.selectbox("Student ID column (scores)", options=[None] + df.columns.tolist(), key="scores_sid_col_single")
        if not sid_col:
            st.error("Please select the Student ID column to proceed.")
            st.stop()

# Optional roster upload (for sorting only)
st.subheader("Optional: Roster for distribution (Teacher/Class)")
roster_up = st.file_uploader("Upload roster CSV (Teacher/Class mapping)", type=["csv"], key="roster_csv_single")

if roster_up is not None:
    try:
        roster_df = pd.read_csv(roster_up, dtype={STUDENT_ID_COL: "string"})
    except Exception:
        roster_up.seek(0)
        roster_df = pd.read_csv(roster_up, dtype={STUDENT_ID_COL: "string"}, encoding="latin-1")

    rc = detect_roster_columns(roster_df)
    cols = [None] + roster_df.columns.tolist()
    def _idx(v):
        try: return cols.index(v)
        except Exception: return 0

    c1, c2, c3 = st.columns(3)
    with c1:
        teacher_col = st.selectbox("Teacher column (roster)", options=cols, index=_idx(rc.get("teacher")), key="roster_teacher_col_single")
    with c2:
        class_col   = st.selectbox("Class/Period column (roster)", options=cols, index=_idx(rc.get("class")), key="roster_class_col_single")
    with c3:
        sid_roster  = st.selectbox("Student ID column (roster)", options=cols, index=_idx(rc.get("student_id")), key="roster_sid_col_single")

    if teacher_col and class_col and sid_roster:
        left  = pd.DataFrame({"__sid": sid_norm_series(df[sid_col])})
        right = roster_df[[sid_roster, teacher_col, class_col]].rename(
            columns={sid_roster:"__sid", teacher_col:"__teacher", class_col:"__class"}
        )
        right["__sid"] = sid_norm_series(right["__sid"])
        merged = left.merge(right, on="__sid", how="left")
        df["Teacher"] = merged["__teacher"].astype("string")
        df["Class"]   = merged["__class"].astype("string")
        matched = df["Teacher"].notna().sum()
        st.success(f"Roster matched to students: {matched}/{len(df)}")
        with st.expander("Preview roster merge (first 20)"):
            name_cols = [c for c in [STUDENT_NAME_COL] if c in df.columns]
            if not name_cols:
                guess = [c for c in df.columns if c.lower().startswith("student") and "id" not in c.lower()]
                name_cols = guess[:1]
            show_cols = [c for c in [sid_col] + name_cols + ['Teacher','Class'] if c in df.columns]
            st.dataframe(df.loc[:, show_cols].head(20))
    else:
        st.info("Select Teacher, Class, and Student ID columns to link the roster.")

# ---------- Optional: Sections enrollment (Student -> Section) ----------
st.subheader("Optional: Sections enrollment (Student → Section)")
sections_up = st.file_uploader("Upload sections CSV (each row = a student in a section)", type=["csv"], key="sections_csv_v1")

# Defaults (columns will appear later even if no file)
selected_sections = []

if sections_up is not None:
    try:
        sections_df = pd.read_csv(sections_up, dtype={"Student ID":"string"})
    except Exception:
        sections_up.seek(0)
        sections_df = pd.read_csv(sections_up, dtype={"Student ID":"string"}, encoding="latin-1")

    sc = detect_section_columns(sections_df)
    sec_cols = [None] + sections_df.columns.tolist()
    def _idx(v):
        try: return sec_cols.index(v)
        except Exception: return 0

    s1, s2 = st.columns(2)
    with s1:
        sid_sections = st.selectbox("Student ID column (sections)", options=sec_cols, index=_idx(sc.get("student_id")), key="sections_sid_col_v1")
    with s2:
        section_col  = st.selectbox("Section column", options=sec_cols, index=_idx(sc.get("section")), key="sections_section_col_v1")

    if sid_sections and section_col:
        # Normalize IDs and clean section codes as strings
        sec = sections_df[[sid_sections, section_col]].rename(columns={sid_sections:"__sid", section_col:"__section"})
        sec["__sid"] = sid_norm_series(sec["__sid"])
        sec["__section"] = sec["__section"].astype(str).str.strip()

        # Aggregate to a per-student list of sections
        agg = (sec[sec["__section"]!=""]
               .groupby("__sid")["__section"]
               .apply(lambda s: sorted(set(s.tolist())))
               .reset_index()
               .rename(columns={"__section":"__sections"}))

        # Attach to df (one row per student), storing both list and a printable string
        left_ids = pd.DataFrame({"__sid": sid_norm_series(df[sid_col])})
        merged_sections = left_ids.merge(agg, on="__sid", how="left")
        df["SectionsList"] = merged_sections["__sections"]
        df["Sections"] = merged_sections["__sections"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")

        # Sidebar multiselect for section filter (populate from attached data)
        all_sections = sorted({sec for lst in df["SectionsList"].dropna() for sec in lst})
        selected_sections = st.sidebar.multiselect("Filter by Section", options=all_sections, default=[], key="filter_sections_v1")

        # Diagnostics
        matched = df["Sections"].ne("").sum()
        st.success(f"Sections attached to {matched}/{len(df)} students.")
        with st.expander("Preview sections (first 20)"):
            name_col = STUDENT_NAME_COL if STUDENT_NAME_COL in df.columns else (df.columns[0] if len(df.columns)>0 else "Student")
            show = [c for c in [sid_col, name_col, "Sections"] if c in df.columns or c in ["Sections"]]
            st.dataframe(df.loc[:, show].head(20))
    else:
        st.info("Select the Student ID and Section columns to link sections.")

# Augment for subject
# Augment + normalize grades, then re-augment so percentiles/levels use clean grades
df = attach_latest_and_percentiles(df, subject)   # creates LatestGradeTested, etc.
df = normalize_current_grade(df, "Current Grade") # force 3..8
df = attach_latest_and_percentiles(df, subject)   # recompute percentiles/levels with clean grade


# Filters / sort
view = filter_by_grade_and_level(df, grades_filter, levels_filter, hide_lvl4)
# NEW: filter by selected sections (if any)
view = apply_section_filter(view, selected_sections)
view = sort_df(view, sort_choice)
if pick_mode == "First N only":
    view = view.head(int(first_n))
view = sort_df(view, sort_choice)
if pick_mode == "First N only":
    view = view.head(int(first_n))

# ---------- Export: PDF (one page per student) ----------
st.markdown("### Export")
def build_pdf_bytes(rows: pd.DataFrame, subject: str, title: str) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for _, row in rows.iterrows():
            fig = plt.figure(figsize=(8.5, 11), dpi=160, constrained_layout=False)
            # 22 rows grid
            gs = fig.add_gridspec(22, 6)
            fig.subplots_adjust(left=0.085, right=0.99, top=0.975, bottom=0.06, hspace=0.28)

            ax_title = fig.add_subplot(gs[0:1, :])
            ax_meta  = fig.add_subplot(gs[1:3, :])
            ax_trend = fig.add_subplot(gs[3:13, :])
            ax_key   = fig.add_subplot(gs[13:15, :])
            ax_growth= fig.add_subplot(gs[15:18, :])
            ax_body  = fig.add_subplot(gs[19:, :])

            name     = row.get(STUDENT_NAME_COL, "Student")
            latest   = row.get("LatestScore", np.nan)
            lvl      = int(row.get("LatestLevel", 0) or 0)
            pts_next = row.get("PtsToNextLevel", np.nan)
            pct_text = row.get("SchoolPercentileOrdinal", "")
            cg       = row.get(CURRENT_GRADE_COL, "")
            tested   = int(row.get("LatestGradeTested", cg if pd.notna(cg) else 8) or 0)

            # Title + name
            ax_title.text(0.0, 0.80, f"{title}", ha="left", va="center",
                          fontsize=18, weight="bold", transform=ax_title.transAxes)
            ax_title.text(0.0, 0.18, f"{name}", ha="left", va="center",
                          fontsize=16, weight="bold", transform=ax_title.transAxes)
            ax_title.axis("off")

            # Meta row with pill + two lines
            lvl_text = f"Level {lvl} ({['','Not Met','Nearly Met','Met','Exceeded'][lvl] if lvl in [1,2,3,4] else '—'})"
            pill_color = ISR_COLORS.get(lvl, "#DDDDDD")
            ax_meta.text(
                0.0, 0.94, lvl_text, transform=ax_meta.transAxes,
                ha="left", va="center", fontsize=10.5,
                bbox=dict(
                    boxstyle="round,pad=0.2,rounding_size=0.2",
                    facecolor=pill_color, edgecolor="black", linewidth=0.8, alpha=0.95
                ),
            )
            ax_meta.text(
                0.0, 0.62, f"Current Grade: {cg}  |  Latest Tested: Grade {tested}",
                transform=ax_meta.transAxes, ha="left", va="center", fontsize=10
            )
            pts_txt = (
                f"{int(pts_next)} pts to next level" if (pd.notna(pts_next) and lvl < 4)
                else "at top level" if lvl == 4 else "—"
            )
            latest_score = "" if pd.isna(latest) else int(round(latest))
            ax_meta.text(
                0.0, 0.32,
                f"Latest Score: {latest_score}  |  {pts_txt}  |  School Percentile: {pct_text}",
                transform=ax_meta.transAxes, ha="left", va="center", fontsize=10
            )
            ax_meta.axis('off')

            # Trend chart (side padding)
            build_trend_figure(row, subject, ax=ax_trend)
            pt = ax_trend.get_position()
            pad_x = 0.012
            ax_trend.set_position([pt.x0 + pad_x, pt.y0, pt.width - 2*pad_x, pt.height])
            ax_trend.tick_params(axis="y", pad=2)

            # ↓ Move the “Grade Tested” label DOWN (larger = farther below the axis)
            ax_trend.set_xlabel("Grade Tested", labelpad=2)   # try 10–16 to taste

            # Level key
            draw_level_key(ax_key, subject, tested if tested else (cg or 6))
            posk = ax_key.get_position()
            ax_key.set_position([posk.x0, posk.y0 - 0.020, posk.width, posk.height])

            # Growth chart (side padding + slight up)
            build_growth_figure(row, subject, ax=ax_growth)
            pos = ax_growth.get_position()
            ax_growth.set_position([pos.x0 + 0.012, pos.y0 + 0.012, pos.width - 2*0.012, pos.height])

            # Body text (safe, line-by-line) with bold section headers
            from textwrap import wrap
            
            blurb = what_this_means(level=lvl, subject=subject, percentile_text=pct_text or "—")
            yby   = year_by_year_lines(row, subject)
            
            # Tunables
            WRAP = 96        # characters per line before wrap
            LINE = 0.12     # vertical step per line (axes coords). Try 0.048–0.056 to taste
            GAP  = 0.018     # extra gap between the two blocks
            y     = 0.92     # starting y (axes coords; top=1)
            
            ax_body.axis('off')
            
            # Heading 1
            ax_body.text(0.0, y, "What this means",
                         transform=ax_body.transAxes, ha='left', va='top',
                         fontsize=11, fontweight='bold')
            y -= LINE
            
            # Blurb (wrapped)
            for t in wrap(blurb or "", width=WRAP):
                ax_body.text(0.0, y, t,
                             transform=ax_body.transAxes, ha='left', va='top',
                             fontsize=11)  # no linespacing
                y -= LINE
            
            # Gap between blocks
            y -= GAP
            
            # Heading 2
            ax_body.text(0.0, y, "Year-by-Year Performance",
                         transform=ax_body.transAxes, ha='left', va='top',
                         fontsize=11, fontweight='bold')
            y -= LINE
            
            # Year-by-year lines (each possibly wrapped)
            for line in (yby or []):
                for t in wrap(line, width=WRAP):
                    ax_body.text(0.0, y, t,
                                 transform=ax_body.transAxes, ha='left', va='top',
                                 fontsize=11)
                    y -= LINE

            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()

def what_this_means(level: int, subject: str, percentile_text: str) -> str:
    if subject == "Math":
        msgs = {
            1: "You’re currently Level 1 (Not Met). We’ll build number sense, operations with fractions/decimals, and strategies for multi-step problems.",
            2: "You’re currently Level 2 (Nearly Met). You’re close—focus on multi-step problem solving, precise computation, and explaining reasoning.",
            3: f"You’re currently Level 3 (Met). That places you around the {percentile_text} in your grade at our school.",
            4: f"You’re Level 4 (Exceeded). Excellent performance—keep pushing with enrichment tasks and deeper problem solving.",
        }
    else:
        msgs = {
            1: "You’re Level 1 (Not Met). We’ll work on reading complex texts, vocabulary, and citing evidence clearly.",
            2: "You’re Level 2 (Nearly Met). Keep practicing text analysis, structure, and revision for clarity and precision.",
            3: f"You’re currently Level 3 (Met). That places you around the {percentile_text} in your grade at our school.",
            4: f"You’re Level 4 (Exceeded). Push into advanced texts and extended writing to keep growing.",
        }
    return msgs.get(level, "Once you have a score, we’ll add next steps here.")

def year_by_year_lines(row: pd.Series, subject: str) -> List[str]:
    lines = []
    for g in range(3, 9):
        col = f"Score_{g*10}"
        if col in row.index and pd.notna(row[col]):
            s = int(round(float(row[col])))
            lvl = score_to_level(s, subject, g)
            pts = points_to_next_level(s, subject, g)
            nxt = "at top level" if lvl == 4 else f"{int(pts)} pts to next level"
            lines.append(f"G{g}: {s} — Level {lvl} ({level_name(lvl)}), {nxt}")
    return lines

def strip_html(s: str) -> str:
    import re
    return re.sub('<[^<]+?>', '', s)

if st.button("Build PDF of current selection"):
    pdf_bytes = build_pdf_bytes(view, subject, f"{school_name} — {subject} CAASPP Test Scores")
    st.download_button("Download PDF", data=pdf_bytes, file_name=f"{subject}_onepagers.pdf", mime="application/pdf")

# ---------- Render in-app (matching screenshot structure) ----------
for _, row in view.iterrows():
    name = row.get(STUDENT_NAME_COL, "Student")
    latest = row.get("LatestScore", np.nan)
    lvl = int(row.get("LatestLevel", 0) or 0)
    pts_next = row.get("PtsToNextLevel", np.nan)
    pct_text = row.get("SchoolPercentileOrdinal", "")
    cg = row.get(CURRENT_GRADE_COL, "")
    tested = int(row.get("LatestGradeTested", cg if pd.notna(cg) else 8) or 0)

    # Title + name
    st.markdown(f"# {school_name} — {subject} CAASPP Test Scores")
    st.markdown(f"## {name}")

    # Three-line header
    latest_score_txt = "" if pd.isna(latest) else int(round(latest))
    pts_txt = ("at top level" if lvl == 4 else f"{int(pts_next)} pts to next level") if pd.notna(pts_next) else "—"

    # Line 1: pill
    st.markdown(
        f'<div style="display:block; line-height:1; margin:2px 0 10px 0;">{level_pill_html(lvl)}</div>',
        unsafe_allow_html=True,
    )
    # Line 2
    st.markdown(f"**Current Grade:** {cg}  |  **Latest Tested:** Grade {tested}")
    # Line 3
    st.markdown(f"**Latest Score:** {latest_score_txt}  |  {pts_txt}  |  **School Percentile:** {pct_text}")

    fig_trend = build_trend_figure(row, subject)
    st.pyplot(fig_trend, dpi=160, clear_figure=True, use_container_width=True)
    st.markdown(level_key_inline(subject, tested), unsafe_allow_html=True)

    fig_growth = build_growth_figure(row, subject)
    st.pyplot(fig_growth, dpi=160, clear_figure=True, use_container_width=True)

    st.markdown("**What this means**")
    st.write(what_this_means(lvl, subject, pct_text or "—"))
    lines = year_by_year_lines(row, subject)
    if lines:
        st.markdown("**Year-by-Year Performance**")
        st.markdown("\n".join([f"- {t}" for t in lines]))
    st.divider()

st.caption("Done rendering selected students.")
