"""Student Success Predictor - Streamlit app.

The app never trains models. Developers train offline with `train.py`, which writes versioned
bundles to models/ and updates models/registry.json. The app loads, compares and serves those versions.
"""
import hmac
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import RAW_FEATURES, TARGET
from model_utils import LABEL_MAP, clean, predict

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")

BASE = Path(__file__).parent
MODELS = BASE / "models"
REGISTRY = MODELS / "registry.json"
METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_NAMES = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "precision": "Precision",
                "recall": "Recall", "f1": "F1 score", "roc_auc": "ROC-AUC"}
REQUIRED_BUNDLE_KEYS = {"model", "metrics", "scored", "importances", "params", "model_name"}

ACCENT, ACCENT_DARK, PASS_C = "#E53935", "#B71C1C", "#607D8B"
OUTCOME_COLORS = alt.Scale(domain=["Pass", "Fail"], range=[PASS_C, ACCENT])
# ---------------------------------------------------------------- light / dark theme (toggle in banner)
THEMES = {
    "light": {"base": "light", "primaryColor": "#E53935", "backgroundColor": "#FFFFFF",
              "secondaryBackgroundColor": "#FFF5F5", "textColor": "#1F2933"},
    "dark": {"base": "dark", "primaryColor": "#EF5350", "backgroundColor": "#121316",
             "secondaryBackgroundColor": "#1E1F24", "textColor": "#ECEFF1"},
}
if "theme" not in st.session_state:
    try:
        st.session_state.theme = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        st.session_state.theme = "light"
for _k, _v in THEMES[st.session_state.theme].items():
    st._config.set_option(f"theme.{_k}", _v)
if st.session_state.pop("theme_changed", False):
    st.rerun()   # theme is sent to the browser at the start of a run, so rerun once to apply it


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.session_state.theme_changed = True


IS_DARK = st.session_state.theme == "dark"

FEATURE_HELP = {
    "Submission.Actions": "Assignment submissions", "Quiz.Actions": "Quiz interactions",
    "Forum.Actions": "Forum posts/replies", "Create.Actions": "Content created",
    "Read.Actions": "Pages/resources read", "Update.Actions": "Content edits",
    "Delete.Actions": "Content deletions", "Days.Logged.In": "Distinct days logged in",
}

# ---------------------------------------------------------------- styling (works in light and dark)
st.markdown(f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1400px;}}
.st-key-hero {{background: linear-gradient(120deg, {ACCENT} 0%, {ACCENT_DARK} 100%); border-radius: 18px;
        padding: 24px 30px; margin-bottom: 14px; box-shadow: 0 10px 30px rgba(229,57,53,.25);}}
.st-key-hero h1 {{color: #fff !important; font-size: 2.05rem; margin: 0 0 4px 0; padding: 0; font-weight: 800;}}
.st-key-hero p {{color: rgba(255,255,255,.92) !important; margin: 0; font-size: 1rem;}}
.st-key-hero button {{background: rgba(255,255,255,.16) !important; color: #fff !important;
        border: 1px solid rgba(255,255,255,.45) !important; border-radius: 999px !important; font-weight: 700;
        white-space: nowrap; float: right;}}
.st-key-hero button:hover {{background: rgba(255,255,255,.28) !important;}}
.st-key-hero button p {{color: #fff !important;}}
div[data-testid="stMetric"] {{border: 1px solid rgba(128,128,128,.22); border-left: 4px solid {ACCENT};
        border-radius: 14px; padding: 14px 16px; background: rgba(229,57,53,.045);}}
div[data-testid="stMetricLabel"] p {{font-weight: 600; opacity: .8;}}
.stTabs [data-baseweb="tab-list"] {{gap: 6px;}}
.stTabs [data-baseweb="tab"] {{font-weight: 700; font-size: 1.02rem; padding: 10px 18px; border-radius: 10px 10px 0 0;}}
.stTabs [aria-selected="true"] {{background: rgba(229,57,53,.09);}}
.section {{font-size: 1.15rem; font-weight: 750; margin: 18px 0 6px 0; padding-left: 10px;
          border-left: 4px solid {ACCENT};}}
.sub {{opacity: .7; font-size: .88rem; margin: -2px 0 8px 14px;}}
.badge {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 700;
        background: {ACCENT}; color: #fff; margin-left: 6px; vertical-align: middle;}}
.badge.muted {{background: rgba(128,128,128,.25); color: inherit;}}
.result {{border-radius: 16px; padding: 22px 24px; border: 1px solid rgba(128,128,128,.25);}}
.result.pass {{background: rgba(96,125,139,.12); border-left: 6px solid {PASS_C};}}
.result.fail {{background: rgba(229,57,53,.10); border-left: 6px solid {ACCENT};}}
.result .big {{font-size: 1.6rem; font-weight: 800; margin-bottom: 2px;}}
.result .prob {{font-size: 2.6rem; font-weight: 800; line-height: 1.1;}}
.muted {{opacity: .7; font-size: .88rem;}}
</style>
""", unsafe_allow_html=True)


def section(title: str, sub: str | None = None):
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


# ---------------------------------------------------------------- model registry
def read_registry() -> dict:
    if not REGISTRY.exists():
        return {"active": None, "versions": []}
    return json.loads(REGISTRY.read_text())


@st.cache_resource(show_spinner="Loading model version…")
def load_bundle(version: str, mtime: float) -> dict:
    b = joblib.load(MODELS / f"{version}.joblib")
    b["scored_df"] = pd.DataFrame(b["scored"])
    b["importances_df"] = pd.DataFrame(b["importances"])
    b["roc_df"] = pd.DataFrame(b["metrics"]["roc"])
    return b


def get_bundle(version: str) -> dict:
    return load_bundle(version, (MODELS / f"{version}.joblib").stat().st_mtime)


def _admin_key() -> str:
    try:
        return str(st.secrets.get("ADMIN_KEY", "ADMIN"))
    except Exception:
        return "ADMIN"


ADMIN_KEY = _admin_key()


registry = read_registry()
if not registry["versions"]:
    st.error("No model versions found. Developers: run `python train.py` and commit the `models/` folder.")
    st.stop()
LIVE = registry["active"]
live = get_bundle(LIVE)
model = live["model"]
lm = live["metrics"]

# Student table from the live version (out-of-fold probabilities: each student scored by a model that never saw them)
students = live["scored_df"].copy()
students.insert(0, "Student ID", [f"S{i + 1:04d}" for i in range(len(students))])
students["Predicted"] = np.where(students["Pass probability"] >= 0.5, "Pass", "Fail")
students["Risk level"] = pd.cut(students["Pass probability"], [-0.01, 0.35, 0.65, 1.01],
                                labels=["High risk", "Moderate", "Low risk"]).astype(str)
students["Total.Actions"] = students[RAW_FEATURES[:-1]].sum(axis=1)
students["Activity level"] = pd.qcut(students["Total.Actions"].rank(method="first"), 4,
                                     labels=["Low", "Moderate", "High", "Very high"]).astype(str)

# ---------------------------------------------------------------- hero
with st.container(key="hero"):
    h_text, h_btn = st.columns([6, 1], vertical_alignment="center")
    h_text.markdown("""
<h1>🎓 Student Success Predictor</h1>
<p>Flags students likely to <b>pass</b> or at risk of <b>failing</b>, using learning-platform engagement
(submissions, quizzes, forum, content activity and days logged in).</p>
""", unsafe_allow_html=True)
    h_btn.button("☀️ Light mode" if IS_DARK else "🌙 Dark mode", on_click=toggle_theme, key="theme_btn")

tab_main, tab_model, tab_predict = st.tabs(["📊  Main", "🧠  Model", "🔮  Predict"])

# ================================================================= MAIN
with tab_main:
    with st.expander("🔎  Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        f_outcome = c1.multiselect("Actual outcome", ["Pass", "Fail"], default=["Pass", "Fail"])
        f_risk = c2.multiselect("Predicted risk level", ["High risk", "Moderate", "Low risk"],
                                default=["High risk", "Moderate", "Low risk"])
        dmin, dmax = int(students["Days.Logged.In"].min()), int(students["Days.Logged.In"].max())
        f_days = c3.slider("Days logged in", dmin, dmax, (dmin, dmax))
        f_search = c4.text_input("Student ID contains", placeholder="e.g. S01")
        c5, c6, c7, c8 = st.columns(4)
        f_level = c5.multiselect("Activity level", ["Low", "Moderate", "High", "Very high"],
                                 default=["Low", "Moderate", "High", "Very high"])
        f_quiz = c6.selectbox("Quiz usage", ["All", "Uses quizzes", "No quizzes"])
        f_forum = c7.selectbox("Forum usage", ["All", "Uses forum", "No forum"])
        f_correct = c8.selectbox("Prediction vs actual", ["All", "Correct", "Misclassified"])

    f = students[students["Outcome"].isin(f_outcome) & students["Risk level"].isin(f_risk)
                 & students["Days.Logged.In"].between(*f_days) & students["Activity level"].isin(f_level)]
    if f_search:
        f = f[f["Student ID"].str.contains(f_search.strip(), case=False)]
    if f_quiz != "All":
        f = f[(f["Quiz.Actions"] > 0) == (f_quiz == "Uses quizzes")]
    if f_forum != "All":
        f = f[(f["Forum.Actions"] > 0) == (f_forum == "Uses forum")]
    if f_correct != "All":
        f = f[(f["Predicted"] == f["Outcome"]) == (f_correct == "Correct")]

    if f.empty:
        st.warning("No students match the current filters.")
    else:
        k = st.columns(6)
        k[0].metric("Students", f"{len(f):,}", f"{len(f) / len(students):.0%} of all", delta_color="off")
        k[1].metric("Pass rate", f"{(f['Outcome'] == 'Pass').mean():.1%}")
        k[2].metric("High-risk students", f"{(f['Risk level'] == 'High risk').sum():,}",
                    f"{(f['Risk level'] == 'High risk').mean():.0%} of filtered", delta_color="off")
        k[3].metric("Avg days logged in", f"{f['Days.Logged.In'].mean():.1f}")
        k[4].metric("Median reads", f"{f['Read.Actions'].median():,.0f}")
        k[5].metric("Model accuracy here", f"{(f['Predicted'] == f['Outcome']).mean():.1%}")

        g1, g2 = st.columns(2)
        day_lbl = ["0–5", "6–10", "11–15", "16–20", "21–25", "26–30", "30+"]
        bands = pd.cut(f["Days.Logged.In"], [-1, 5, 10, 15, 20, 25, 30, 1000], labels=day_lbl).astype(str)
        with g1, st.container(border=True):
            section("Pass rate by days logged in", "Share of students who passed in each attendance band")
            agg = (f.assign(Band=bands).groupby("Band")
                   .agg(PassRate=("Outcome", lambda s: (s == "Pass").mean()), Students=("Outcome", "size"))
                   .reset_index())
            st.altair_chart(alt.Chart(agg).mark_bar(color=ACCENT, cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                            .encode(x=alt.X("Band:N", sort=day_lbl, title="Days logged in", axis=alt.Axis(labelAngle=0)),
                                    y=alt.Y("PassRate:Q", axis=alt.Axis(format="%"), title="Pass rate"),
                                    tooltip=["Band", alt.Tooltip("PassRate:Q", format=".1%"), "Students"])
                            .properties(height=300))
        with g2, st.container(border=True):
            section("Typical activity: pass vs fail", "Median count per activity type (log scale)")
            long = (f.groupby("Outcome")[RAW_FEATURES[:-1]].median().reset_index()
                    .melt("Outcome", var_name="Activity", value_name="Median"))
            long["Activity"] = long["Activity"].str.replace(".Actions", "", regex=False)
            st.altair_chart(alt.Chart(long).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                            .encode(y=alt.Y("Activity:N", title=None),
                                    x=alt.X("Median:Q", title="Median count", scale=alt.Scale(type="symlog")),
                                    color=alt.Color("Outcome:N", scale=OUTCOME_COLORS), yOffset="Outcome:N",
                                    tooltip=["Outcome", "Activity", "Median"])
                            .properties(height=300))

        g3, g4 = st.columns(2)
        with g3, st.container(border=True):
            section("Pass-rate heatmap", "Attendance × activity level · darker red = lower pass rate · cells with under 10 students hidden")
            hm = (f.assign(Band=bands).groupby(["Band", "Activity level"])
                  .agg(PassRate=("Outcome", lambda s: (s == "Pass").mean()), Students=("Outcome", "size"))
                  .reset_index())
            hm = hm[hm["Students"] >= 10]   # hide tiny, noisy cells
            levels = ["Low", "Moderate", "High", "Very high"]
            base = alt.Chart(hm).encode(
                x=alt.X("Activity level:N", sort=levels, title="Activity level (quartile)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Band:N", sort=day_lbl, title="Days logged in"))
            st.altair_chart(
                (base.mark_rect(cornerRadius=4, stroke="rgba(128,128,128,0.25)")
                 .encode(color=alt.Color("PassRate:Q", title="Pass rate", legend=alt.Legend(format="%"),
                                         scale=alt.Scale(domain=[0, 0.35, 0.6, 1], range=[ACCENT_DARK, ACCENT, "#FFCDD2", "#ECEFF1"])),
                         tooltip=["Band", "Activity level", alt.Tooltip("PassRate:Q", format=".1%"), "Students"])
                 + base.mark_text(fontSize=12, fontWeight="bold")
                 .encode(text=alt.Text("PassRate:Q", format=".0%"),
                         color=alt.condition("datum.PassRate < 0.55", alt.value("#FFFFFF"), alt.value("#263238"))))
                .properties(height=300))
        with g4, st.container(border=True):
            section("Predicted pass probability", "How confidently the model separates actual passes and fails")
            st.altair_chart(alt.Chart(f).mark_bar(opacity=0.8, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                            .encode(x=alt.X("Pass probability:Q", bin=alt.Bin(step=0.05), axis=alt.Axis(format="%"),
                                            title="Predicted pass probability"),
                                    y=alt.Y("count():Q", stack=None, title="Students"),
                                    color=alt.Color("Outcome:N", scale=OUTCOME_COLORS, title="Actual"))
                            .properties(height=300))

        with st.container(border=True):
            section("Students", "Click any column header to re-sort, or use the controls")
            s1, s2, s3 = st.columns([2, 1, 1])
            sort_opts = ["Pass probability", "Days.Logged.In", "Total.Actions", "Read.Actions",
                         "Submission.Actions", "Quiz.Actions", "Forum.Actions", "Student ID"]
            sort_by = s1.selectbox("Sort by", sort_opts, index=0)
            asc = s2.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"
            n_rows = s3.number_input("Rows to show", min_value=1, max_value=len(f), value=min(200, len(f)), step=50)
            cols = (["Student ID", "Outcome", "Predicted", "Pass probability", "Risk level", "Activity level"]
                    + RAW_FEATURES + ["Total.Actions"])
            view = f.sort_values(sort_by, ascending=asc)[cols].head(int(n_rows))
            st.dataframe(view, hide_index=True, height=430,
                         column_config={"Pass probability": st.column_config.ProgressColumn(
                             "Pass probability", format="%.2f", min_value=0.0, max_value=1.0)})
            st.download_button("⬇️  Download filtered students (CSV)", f[cols].to_csv(index=False).encode(),
                               "filtered_students.csv", "text/csv")

# ================================================================= MODEL
with tab_model:
    versions = [v["version"] for v in registry["versions"]]
    top1, top2 = st.columns([1, 3])
    view_v = top1.selectbox("Model version", versions[::-1],
                            format_func=lambda v: f"{v}  (live)" if v == LIVE else v)
    vb = get_bundle(view_v)
    vm = vb["metrics"]
    meta = next(v for v in registry["versions"] if v["version"] == view_v)
    badge = '<span class="badge">LIVE</span>' if view_v == LIVE else '<span class="badge muted">archived</span>'
    top2.markdown(
        f"<div style='padding-top:6px'><b style='font-size:1.15rem'>{vb['model_name']}</b> {badge}<br>"
        f"<span class='muted'>Created {meta['created']} · trained on {meta['records']:,} unique students · "
        f"data: {', '.join(vb.get('sources', ['students.csv']))}</span><br>"
        f"<span class='muted'>📝 {meta.get('notes') or 'No notes'}</span></div>", unsafe_allow_html=True)

    idx = versions.index(view_v)
    prev = registry["versions"][idx - 1] if idx > 0 else None
    m = st.columns(6)
    for col, key in zip(m, METRIC_KEYS):
        delta = f"{vm[key] - prev[key]:+.3f} vs {prev['version']}" if prev else None
        col.metric(METRIC_NAMES[key], f"{vm[key]:.3f}", delta)
    di = vb.get("data_info", {})
    st.caption(f"5-fold stratified cross-validation on unique students. The raw data had {di.get('original', '?'):,} rows; "
               f"{di.get('duplicates', '?')} exact duplicates (augmentation copies) and {di.get('conflicting', '?')} "
               "same-activity rows with conflicting labels were removed before training. Because classes are uneven "
               "(~63% pass), **balanced accuracy** is the fairest single number.")

    a, b, c = st.columns(3)
    with a, st.container(border=True):
        section("Confusion matrix")
        cm = np.asarray(vm["confusion"])
        cmd = pd.DataFrame([(act, prd, int(cm[i, j])) for i, act in enumerate(["Pass", "Fail"])
                            for j, prd in enumerate(["Pass", "Fail"])], columns=["Actual", "Predicted", "Count"])
        cmd["Correct"] = cmd["Actual"] == cmd["Predicted"]
        hi = cmd["Count"].max()
        base = alt.Chart(cmd).encode(x=alt.X("Predicted:N", sort=["Pass", "Fail"], axis=alt.Axis(labelAngle=0)),
                                     y=alt.Y("Actual:N", sort=["Pass", "Fail"]))
        st.altair_chart((base.mark_rect(cornerRadius=6).encode(
                            color=alt.Color("Count:Q", scale=alt.Scale(range=["#FFEBEE", ACCENT_DARK]), legend=None))
                         + base.mark_text(fontSize=22, fontWeight="bold").encode(
                            text="Count:Q",
                            color=alt.condition(f"datum.Count > {hi * 0.45}", alt.value("#FFFFFF"), alt.value("#263238"))))
                        .properties(height=280))
    with b, st.container(border=True):
        section(f"ROC curve · AUC {vm['roc_auc']:.3f}")
        diag = pd.DataFrame({"False positive rate": [0, 1], "True positive rate": [0, 1]})
        roc = vb["roc_df"]
        st.altair_chart((alt.Chart(roc).mark_area(color=ACCENT, opacity=0.12).encode(x="False positive rate:Q", y="True positive rate:Q")
                         + alt.Chart(roc).mark_line(color=ACCENT, strokeWidth=3).encode(x="False positive rate:Q", y="True positive rate:Q")
                         + alt.Chart(diag).mark_line(strokeDash=[5, 5], color="gray").encode(x="False positive rate:Q", y="True positive rate:Q"))
                        .properties(height=280))
    with c, st.container(border=True):
        section("Top feature importances")
        imp = vb["importances_df"].head(10)
        st.altair_chart(alt.Chart(imp).mark_bar(color=ACCENT, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                        .encode(y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("importance:Q", title="Importance"),
                                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")])
                        .properties(height=280))

    with st.container(border=True):
        section("Version history", "Every model version the developers have published")
        hist = pd.DataFrame(registry["versions"])
        hist.insert(1, "Status", np.where(hist["version"] == LIVE, "🔴 LIVE", "archived"))
        show = hist[["version", "Status", "created", "model_name", "records"] + METRIC_KEYS + ["notes"]].rename(
            columns={"version": "Version", "created": "Created", "model_name": "Model", "records": "Students",
                     "notes": "Notes", **METRIC_NAMES})
        h1, h2 = st.columns([3, 2])
        with h1:
            st.dataframe(show.iloc[::-1], hide_index=True,
                         column_config={METRIC_NAMES[k]: st.column_config.NumberColumn(format="%.3f") for k in METRIC_KEYS})
        with h2:
            trend = hist.melt("version", value_vars=["accuracy", "balanced_accuracy", "roc_auc"],
                              var_name="Metric", value_name="Score")
            trend["Metric"] = trend["Metric"].map(METRIC_NAMES)
            st.altair_chart(alt.Chart(trend).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2.5)
                            .encode(x=alt.X("version:N", title="Version", sort=versions, axis=alt.Axis(labelAngle=0)),
                                    y=alt.Y("Score:Q", scale=alt.Scale(zero=False)),
                                    color=alt.Color("Metric:N", scale=alt.Scale(range=[ACCENT, PASS_C, "#FFA000"])),
                                    tooltip=["version", "Metric", alt.Tooltip("Score:Q", format=".3f")])
                            .properties(height=230))
        st.download_button("⬇️  Download version history (CSV)", show.to_csv(index=False).encode(),
                           "model_versions.csv", "text/csv")

    with st.container(border=True):
        section("🔐 Update model", "Admins can upload a new model version (trained offline with train.py) or change the live version")
        if not st.session_state.get("is_admin"):
            pw = st.text_input("Admin key", type="password", placeholder="Enter admin key to unlock")
            if pw:
                if hmac.compare_digest(pw, ADMIN_KEY):
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Incorrect admin key.")
        else:
            c_up, c_live = st.columns([3, 2], gap="large")
            with c_up:
                mb = st.file_uploader("Upload model version (.joblib from train.py)", type="joblib", key="model_upload")
                if mb is not None:
                    try:
                        bundle = joblib.load(io.BytesIO(mb.getvalue()))
                        missing = REQUIRED_BUNDLE_KEYS - set(bundle)
                        if missing:
                            st.error(f"Not a valid model file. Missing: {', '.join(sorted(missing))}")
                        else:
                            bm = bundle["metrics"]
                            st.write(f"**{bundle['model_name']}**: accuracy {bm['accuracy']:.3f} · "
                                     f"balanced {bm['balanced_accuracy']:.3f} · AUC {bm['roc_auc']:.3f}")
                            notes = st.text_input("Release notes", value=bundle.get("notes", ""))
                            make_live = st.checkbox("Make this version live", value=True)
                            if st.button("Upload version", type="primary"):
                                new_v = f"v{max(int(v[1:]) for v in versions) + 1}"
                                created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                                bundle.update(version=new_v, notes=notes, created=created)
                                joblib.dump(bundle, MODELS / f"{new_v}.joblib", compress=3)
                                registry["versions"].append({
                                    "version": new_v, "created": created, "notes": notes,
                                    "model_name": bundle["model_name"], "records": len(bundle["scored"][TARGET]),
                                    **{k: round(float(bm[k]), 4) for k in METRIC_KEYS}})
                                if make_live:
                                    registry["active"] = new_v
                                REGISTRY.write_text(json.dumps(registry, indent=2))
                                st.rerun()
                    except Exception as e:
                        st.error(f"Could not load model file: {e}")
            with c_live:
                live_choice = st.selectbox("Live version", versions, index=versions.index(LIVE))
                if st.button("Set as live", disabled=live_choice == LIVE):
                    registry["active"] = live_choice
                    REGISTRY.write_text(json.dumps(registry, indent=2))
                    st.rerun()
                if st.button("Lock admin"):
                    st.session_state.is_admin = False
                    st.rerun()

# ================================================================= PREDICT
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")
    medians = students[RAW_FEATURES].median()
    with left, st.container(border=True):
        section("Student activity", f"Prefilled with median values · scored by live model {LIVE}")
        with st.form("predict_form"):
            pc = st.columns(2)
            inp = {feat: pc[i % 2].number_input(feat.replace(".", " "), min_value=0, value=int(medians[feat]),
                                                step=1, help=FEATURE_HELP[feat], key=f"p_{feat}")
                   for i, feat in enumerate(RAW_FEATURES)}
            st.form_submit_button("🔮  Predict", type="primary")
    with right:
        x = pd.DataFrame([inp])
        res = predict(model, x).iloc[0]
        p = float(res["Pass probability"])
        level = "High risk" if p < 0.35 else "Moderate risk" if p < 0.65 else "Low risk"
        ok = res["Prediction"] == "Pass"
        st.markdown(f"""
        <div class="result {'pass' if ok else 'fail'}">
          <div class="big">{'✅ Likely to PASS' if ok else '⚠️ At risk of FAILING'}</div>
          <div class="muted">Predicted pass probability</div>
          <div class="prob">{p:.0%}</div>
          <div class="muted">{level} · decision threshold 50%</div>
        </div>""", unsafe_allow_html=True)
        st.progress(p)
        with st.container(border=True):
            section("Compared with typical students", "Medians of passing and failing students (log scale)")
            grp = students.groupby("Outcome")[RAW_FEATURES].median()
            comp = pd.concat([grp, x.rename(index={0: "This student"})])
            comp = comp.reset_index(names="Group").melt("Group", var_name="Feature", value_name="Value")
            comp["Feature"] = comp["Feature"].str.replace(".Actions", "", regex=False)
            st.altair_chart(alt.Chart(comp).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                            .encode(y=alt.Y("Feature:N", title=None),
                                    x=alt.X("Value:Q", scale=alt.Scale(type="symlog"), title="Count"),
                                    color=alt.Color("Group:N", scale=alt.Scale(domain=["This student", "Pass", "Fail"],
                                                                               range=["#FFA000", PASS_C, ACCENT])),
                                    yOffset="Group:N", tooltip=["Group", "Feature", "Value"])
                            .properties(height=320))

    with st.container(border=True):
        section("Batch prediction", f"Upload a CSV with columns {', '.join(RAW_FEATURES)} (`rating` optional)")
        bup = st.file_uploader("CSV of students to score", type="csv", key="batch_upload")
        if bup is not None:
            try:
                bdf = pd.read_csv(bup)
                miss = [c for c in RAW_FEATURES if c not in bdf.columns]
                if miss:
                    st.error(f"Missing columns: {', '.join(miss)}")
                else:
                    feats = bdf[RAW_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
                    out = pd.concat([bdf, predict(model, feats)], axis=1)
                    bk = st.columns(3)
                    bk[0].metric("Students scored", f"{len(out):,}")
                    bk[1].metric("Predicted to pass", f"{(out['Prediction'] == 'Pass').mean():.1%}")
                    bk[2].metric("High-risk (<35%)", f"{(out['Pass probability'] < 0.35).sum():,}")
                    st.dataframe(out.sort_values("Pass probability"), hide_index=True,
                                 column_config={"Pass probability": st.column_config.ProgressColumn(
                                     "Pass probability", format="%.2f", min_value=0.0, max_value=1.0)})
                    st.download_button("⬇️  Download predictions", out.to_csv(index=False).encode(),
                                       "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Could not read file: {e}")
