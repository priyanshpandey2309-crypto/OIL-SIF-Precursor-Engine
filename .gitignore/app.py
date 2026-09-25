import pandas as pd
import plotly.express as px
import streamlit as st

from engine import SIFEngine, make_data

st.set_page_config("OIL SIF Precursor Engine", page_icon="⚠️", layout="wide")
st.markdown("""<style>
h1,h2,h3{font-family:'Segoe UI',Arial,sans-serif;letter-spacing:-.01em}
[data-testid=stMetricValue]{font-size:2rem}
</style>""", unsafe_allow_html=True)
AMBER, DARK = "#E08A00", "#2B2B2B"


@st.cache_resource(show_spinner="Training models on the first run...")
def load_engine():
    return SIFEngine()


@st.cache_data(show_spinner="Analysing reports...")
def run(_eng, raw):
    return _eng.analyze(raw)


eng = load_engine()
st.title("SIF precursor detection for HSSE reports")
st.caption("Finds the reports that carry fatal potential, maps them to IOGP Life-Saving Rules "
           "and shows where to intervene first.")

# ---------- data ----------
src = st.sidebar.radio("Data source", ["Demo data (synthetic)", "Upload CSV"])
if src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with columns: text, site, date (optional: report_id, report_type)", type="csv")
    if up is None:
        st.info("Upload a CSV with at least a `text` column, or switch to the demo data in the sidebar.")
        st.stop()
    raw = pd.read_csv(up)
    if "text" not in raw:
        st.error("The file needs a `text` column holding the free-text report.")
        st.stop()
    if "report_id" not in raw:
        raw["report_id"] = [f"R{i:05d}" for i in range(len(raw))]
    raw["site"] = raw["site"].fillna("Unknown") if "site" in raw else "Unknown"
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce") if "date" in raw else pd.Timestamp("2026-09-24")
    raw["date"] = raw["date"].fillna(pd.Timestamp("2026-09-24"))
else:
    raw = make_data(1200, seed=7)

d = run(eng, raw)
thr = st.sidebar.slider("SIF decision threshold", .2, .8, .5, .05,
                        help="Lower = catch more potential SIFs, at the cost of more false alarms.")
sites = st.sidebar.multiselect("Sites", sorted(d.site.unique()), default=sorted(d.site.unique()))
lo, hi = d.date.min().date(), d.date.max().date()
rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
min_n = st.sidebar.slider("Minimum reports for ranking", 1, 30, 5)
with st.sidebar.expander("Model quality (5-fold CV)"):
    m = eng.metrics
    st.write(f"SIF precision {m['precision']:.2f} · recall {m['recall']:.2f} · F1 {m['f1']:.2f}")
    st.write(f"LSR tagging accuracy {m['lsr_accuracy']:.2f}")
    st.caption("Measured on synthetic reports. Retrain on OIL's labelled reports before trusting these numbers.")

d = d[d.site.isin(sites)]
if len(rng) == 2:
    d = d[(d.date.dt.date >= rng[0]) & (d.date.dt.date <= rng[1])]
d = d.assign(sif_flag=d.sif_prob >= thr)
s = d[d.sif_flag]
if d.empty:
    st.warning("No reports match the filters.")
    st.stop()

# ---------- KPIs ----------
by_site = d.groupby("site").agg(reports=("text", "size"), sif=("sif_flag", "sum")).reset_index()
by_site["density"] = (by_site.sif / by_site.reports * 100).round(1)
top_site = by_site[by_site.reports >= min_n].sort_values("density", ascending=False)
c = st.columns(4)
c[0].metric("Reports analysed", f"{len(d):,}")
c[1].metric("Flagged SIF-potential", f"{len(s):,}", f"{len(s) / len(d):.0%} of reports", delta_color="off")
c[2].metric("Top Life-Saving Rule", s.lsr.mode()[0] if len(s) else "-")
c[3].metric("Highest-density site", top_site.iloc[0].site if len(top_site) else "-")

t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Where to intervene", "Life-Saving Rules", "Precursor patterns",
                                     "Flagged reports", "Try a report", "Reviewer feedback", "Data & validation"])

with t1:
    a, b = st.columns(2)
    a.subheader("Sites by SIF-precursor density")
    f = px.bar(top_site.sort_values("density"), x="density", y="site", orientation="h",
               text="sif", hover_data=["reports"], color_discrete_sequence=[AMBER])
    f.update_layout(xaxis_title="% of reports flagged SIF-potential", yaxis_title="", height=380)
    a.plotly_chart(f)
    a.caption("Bar labels show the number of flagged reports.")
    by_act = d.groupby("activity").agg(reports=("text", "size"), sif=("sif_flag", "sum")).reset_index()
    by_act["density"] = (by_act.sif / by_act.reports * 100).round(1)
    by_act = by_act[by_act.reports >= min_n]
    b.subheader("Activities by SIF-precursor density")
    f = px.bar(by_act.sort_values("density"), x="density", y="activity", orientation="h",
               text="sif", hover_data=["reports"], color_discrete_sequence=[DARK])
    f.update_layout(xaxis_title="% of reports flagged SIF-potential", yaxis_title="", height=380)
    b.plotly_chart(f)
    mo = d.assign(month=d.date.dt.to_period("M").astype(str)).groupby("month").sif_flag.mean().mul(100).reset_index()
    st.subheader("SIF share of reports over time")
    st.plotly_chart(px.line(mo, x="month", y="sif_flag", markers=True, color_discrete_sequence=[AMBER])
                    .update_layout(yaxis_title="% flagged", xaxis_title="", height=300))

with t2:
    a, b = st.columns([2, 3])
    lc = s.lsr.value_counts().rename_axis("Life-Saving Rule").reset_index(name="flagged reports")
    a.subheader("Flagged reports per rule")
    a.plotly_chart(px.bar(lc.sort_values("flagged reports"), x="flagged reports", y="Life-Saving Rule",
                          orientation="h", color_discrete_sequence=[AMBER]).update_layout(height=420, yaxis_title=""))
    b.subheader("Site × rule (flagged reports)")
    if len(s):
        pv = s.pivot_table(index="site", columns="lsr", values="text", aggfunc="count", fill_value=0)
        b.plotly_chart(px.imshow(pv, text_auto=True, aspect="auto", color_continuous_scale="Oranges")
                       .update_layout(height=420, xaxis_title="", yaxis_title=""))

with t3:
    if s.empty:
        st.info("No SIF-potential reports at this threshold.")
    else:
        a, b = st.columns(2)
        g = s.groupby(["activity", "location", "barrier_failure"]).size().reset_index(name="n")
        a.subheader("Activity → location → failed barrier")
        a.plotly_chart(px.treemap(g, path=["activity", "location", "barrier_failure"], values="n",
                                  color_discrete_sequence=px.colors.sequential.Oranges_r)
                       .update_layout(height=460, margin=dict(t=10, l=0, r=0, b=0)))
        bf = s.barrier_failure.value_counts().rename_axis("barrier").reset_index(name="n")
        b.subheader("Most common barrier failures")
        b.plotly_chart(px.bar(bf.sort_values("n"), x="n", y="barrier", orientation="h",
                              color_discrete_sequence=[DARK]).update_layout(height=460, yaxis_title=""))
        st.subheader("Recurring patterns (3+ flagged reports)")
        rec = g[g.n >= 3].sort_values("n", ascending=False).rename(
            columns={"n": "reports", "barrier_failure": "barrier failure"})
        st.dataframe(rec, hide_index=True)

with t4:
    cols = ["report_id", "date", "site", "report_type", "sif_prob", "confidence", "lsr", "activity",
            "location", "barrier_failure", "text"]
    cols = [x for x in cols if x in s]
    out = s.sort_values("sif_prob", ascending=False)[cols]
    st.dataframe(out, hide_index=True,
                 column_config={"sif_prob": st.column_config.ProgressColumn("SIF probability", min_value=0, max_value=1)})
    st.download_button("Download flagged reports (CSV)", out.to_csv(index=False), "sif_flagged_reports.csv")
    st.caption("Low-confidence rows are the ones most worth a reviewer's attention — see the Reviewer feedback tab.")

with t5:
    txt = st.text_area("Paste a report", height=120, value=(
        "Near miss: Crew started flange opening on the gas line at CPF without lockout tagout, "
        "isolation not verified, hissing noticed before bolts were removed."))
    if st.button("Analyse report") and txt.strip():
        r = eng.analyze(pd.DataFrame({"text": [txt]})).iloc[0]
        flag = r.sif_prob >= thr
        (st.error if flag else st.success)(
            f"{'SIF-potential' if flag else 'Not SIF-potential'} (probability {r.sif_prob:.0%}, {r.confidence.lower()} confidence)")
        x = st.columns(4)
        x[0].metric("Life-Saving Rule", r.lsr)
        x[1].metric("Activity", r.activity)
        x[2].metric("Location", r.location)
        x[3].metric("Barrier failure", r.barrier_failure)
        why = eng.explain(txt)
        if why:
            wf = pd.DataFrame(why, columns=["phrase", "contribution"]).sort_values("contribution")
            st.plotly_chart(px.bar(wf, x="contribution", y="phrase", orientation="h",
                                   color_discrete_sequence=[AMBER])
                            .update_layout(height=220, yaxis_title="", xaxis_title="Push toward SIF-potential"))

with t6:
    st.write("A human reviewer confirms or corrects the model's tag on flagged reports. "
             "Confirmed corrections are folded back into training, the way this system would "
             "keep improving once real HSE reviewers start using it on OIL's own reports.")
    if s.empty:
        st.info("No flagged reports to review at this threshold.")
    else:
        pick = st.selectbox("Report to review", s.sort_values("sif_prob", ascending=False).report_id,
                            format_func=lambda i: f"{i} — {s.set_index('report_id').loc[i, 'text'][:70]}...")
        row = s.set_index("report_id").loc[pick]
        st.write(row.text)
        st.write(f"Model says: **{'SIF-potential' if row.sif_prob >= thr else 'Not SIF-potential'}** "
                 f"({row.sif_prob:.0%}, {row.confidence.lower()} confidence) · Life-Saving Rule: **{row.lsr}**")
        c1, c2 = st.columns(2)
        model_flag = row.sif_prob >= thr
        opts = {"Confirm — model is right": model_flag, "Correct to SIF-potential": True,
                "Correct to not SIF-potential": False}
        default = "Confirm — model is right"
        choice = c1.radio("Reviewer decision", list(opts), index=list(opts).index(default))
        conf_lsr = c2.selectbox("Correct Life-Saving Rule (change if wrong)",
                                sorted(eng.lsr.classes_), index=sorted(eng.lsr.classes_).index(row.lsr))
        if st.button("Submit feedback"):
            eng.record_feedback(pick, row.text, model_flag, row.lsr, opts[choice], conf_lsr)
            st.success("Feedback recorded.")
        if not eng.feedback.empty:
            st.subheader(f"Feedback log ({len(eng.feedback)})")
            st.dataframe(eng.feedback[["report_id", "predicted_sif", "confirmed_sif", "predicted_lsr", "confirmed_lsr"]],
                        hide_index=True)
            corr = (eng.feedback.predicted_sif != eng.feedback.confirmed_sif).mean()
            st.write(f"Reviewer correction rate so far: {corr:.0%}")
            if st.button("Retrain on confirmed feedback"):
                new_m = eng.retrain()
                run.clear()
                st.success(f"Retrained. New CV F1: {new_m['f1']:.2f} (was {m['f1']:.2f}).")
                st.rerun()

with t7:
    st.subheader("Where the training data comes from today")
    st.write("The classifiers are trained on synthetic reports written to follow the SIF-precursor "
             "research the problem statement cites — DEKRA Martin & Black (2015), the EEI SIF "
             "Precursor model, and VelocityEHS's 2024 PSIF classifier — plus OIL's own Life-Saving "
             "Rules and typical site vocabulary (wellhead, GGS, CPF, rig floor). This lets the pipeline "
             "be demonstrated end-to-end before any real report is available.")
    st.subheader("Plan to move to OIL's real reports")
    st.markdown("""
1. **Label a seed set.** Have HSE tag 300–500 historical UA/UC, near-miss and incident reports as SIF-potential or not, and to a Life-Saving Rule. Two reviewers per report, disagreements resolved by a senior HSE officer.
2. **Swap the data.** Replace `make_data()` in `SIFEngine.__init__` with that labelled set; the rest of the pipeline (TF-IDF, classifiers, dashboard) needs no change.
3. **Re-measure.** Re-run the same 5-fold cross-validation this app already reports, on real data this time.
4. **Human-in-the-loop from day one.** The Reviewer feedback tab is how corrections on live reports get folded back in — start collecting them immediately, don't wait for a full labelled set.
5. **Language.** Extend the tokenizer/stopword list for Hindi/Assamese-English mixed phrasing common in field reports, once real examples show how often it occurs.
""")
    st.caption("This tab is the honest answer to \"how do you know this works on real data\": it doesn't yet, and this is exactly how we'd find out.")
