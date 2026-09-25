"""SIF-precursor NLP engine for OIL safety reports.

a) SIF-potential vs non-SIF classifier   (TF-IDF + logistic regression)
b) IOGP Life-Saving Rule tagger          (TF-IDF + logistic regression, keyword fallback)
c) Precursor extraction                  (activity, location, barrier failure)

OIL's real reports are not public, so make_data() builds realistic synthetic
reports for training and demo. Swap it for OIL's labelled data (see README).
"""
import random
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

OTHER = "Other / No LSR"
SITE_RATE = {"Duliajan": .30, "Naharkatia": .22, "Moran": .15, "Jorhat": .25,
             "Kharsang": .35, "Baghewala (Rajasthan)": .18, "Tinsukia": .12}
SITE_VOL = [4, 2, 2, 3, 1, 2, 2]
LOCS = ["wellhead", "rig floor", "GGS", "CPF", "tank farm", "pipeline ROW",
        "workshop", "drill site", "compressor house", "oil collecting station"]

# LSR -> (SIF-potential templates, non-SIF templates)
SCEN = {
    "Energy Isolation": (
        ["Technician started replacing pump seal at {loc} without lockout tagout; isolation not verified and residual pressure was released.",
         "Electrician opened MCC panel at {loc} for repair, energy not isolated, no tag-out applied, live terminals exposed.",
         "Valve found unlocked during flange opening at {loc}; hydrocarbon released because isolation not verified before breaking containment."],
        ["LOTO tag found with faded ink at {loc}; tag replaced the same day, isolation was properly verified.",
         "Isolation register at {loc} missing one signature, corrected, no work had started."]),
    "Hot Work": (
        ["Welding carried out near open drain at {loc} without hot work permit and gas test; flammable vapour present nearby.",
         "Grinding sparks fell close to condensate tank at {loc}, no fire watch, no fire blanket.",
         "Cutting torch used on pipeline at {loc} while gas test not done, LEL not checked."],
        ["Fire extinguisher at welding bay {loc} found past inspection date; replaced.",
         "Welder at {loc} not wearing welding gloves, spoken to and corrected."]),
    "Confined Space": (
        ["Worker entered tank at {loc} without confined space entry permit; oxygen level not tested and no standby person.",
         "Contractor descended into sump at {loc} with no gas test, rescue equipment not available, attendant absent."],
        ["Confined space permit at {loc} filled with wrong date, corrected before entry.",
         "Entry log at {loc} tank found untidy, updated by supervisor."]),
    "Line of Fire": (
        ["Crew stood under suspended pipe during rig floor handling at {loc}; exclusion zone not barricaded, load swung close to worker.",
         "Worker positioned within swing radius of tong at {loc}, near miss with drill pipe moving.",
         "High-pressure hose whipped at {loc} because whip-check not fitted, worker standing in line of discharge."],
        ["Small tool dropped from table at {loc}, landed on floor, no one nearby.",
         "Worker found standing close to moving forklift at {loc} but at safe distance, coached."]),
    "Working at Height": (
        ["Rigger worked at derrick platform at {loc} without full body harness anchored; no lifeline, fall from height narrowly avoided.",
         "Scaffold at {loc} without toe boards and handrails, worker climbed to 6 metres unsecured.",
         "Roof of pump house at {loc} accessed without fall protection, fragile sheet cracked."],
        ["Ladder at {loc} had a loose rung, tagged out and fixed, nobody was using it.",
         "Worker at {loc} climbed a 1 m step stool without helmet, advised."]),
    "Safe Mechanical Lifting": (
        ["Crane lifted heavy skid at {loc} with damaged sling, no banksman, load tilted and personnel beneath the load.",
         "Lifting plan not prepared for wellhead equipment at {loc}; crane outside load chart, outrigger not fully extended."],
        ["Lifting sling at {loc} had expired colour code tag, removed from service.",
         "Crane operator log book at {loc} not signed for one day, corrected."]),
    "Work Authorisation": (
        ["Contractor started excavation near live gas line at {loc} without work permit; no toolbox talk, underground drawing not checked.",
         "Work started on live wellhead at {loc} without permit to work, JSA not done."],
        ["Permit copy at {loc} not displayed at job site, displayed after reminder.",
         "Toolbox talk record at {loc} incomplete, corrected."]),
    "Bypassing Safety Controls": (
        ["High level shutdown interlock bypassed at {loc} to keep production running, no MOC and no approval; gas detector inhibited.",
         "Fire and gas alarm disabled at {loc} during maintenance and not reinstated, safety critical device override."],
        ["Gas detector at {loc} calibration due next week, scheduled.",
         "Alarm test log at {loc} filled late, corrected."]),
    "Driving": (
        ["Tanker driver near {loc} overspeeding on narrow road, seat belt not worn, vehicle overturned close to pedestrians.",
         "Contractor driver at {loc} drove at night without rest, vehicle hit barrier, driver fatigued, mobile phone in use."],
        ["Vehicle parked wrongly at {loc} gate, moved after advice.",
         "Reverse horn not working on van at {loc}, fixed the same day."]),
}
GENERAL = ["Water spill on floor near canteen at {loc}, mopped, employee slipped but no injury.",
           "Worker not wearing safety glasses while cleaning at {loc}, counselled.",
           "Minor cut on finger while opening a box at {loc}, first aid given.",
           "Stacked pipes untidy at {loc} store, rearranged.",
           "Waste bin overflowing at {loc}, cleared by housekeeping."]
PREFIX = {"UA": "Unsafe act observed:", "UC": "Unsafe condition:",
          "Near Miss": "Near miss:", "Incident": "Incident report:"}
FILL = ["Supervisor informed.", "Area secured.", "Reported during night shift.",
        "Reported during morning inspection.", "Crew of four involved.", "", "", ""]

LSR_KW = {
    "Energy Isolation": r"isolat|lockout|loto|tag-?out|energ|live terminal|residual pressure",
    "Hot Work": r"weld|grind|hot work|cutting|spark|fire watch",
    "Confined Space": r"confined|entered tank|sump|standby person|oxygen",
    "Line of Fire": r"line of fire|suspended|swing|exclusion|whip|struck by|falling object",
    "Working at Height": r"height|harness|scaffold|ladder|fall protection|lifeline|derrick",
    "Safe Mechanical Lifting": r"crane|sling|lifting|hoist|load chart|banksman",
    "Work Authorisation": r"permit|jsa|work authori|toolbox",
    "Bypassing Safety Controls": r"bypass|inhibit|override|interlock|disabled|shutdown",
    "Driving": r"driv|vehicle|tanker|overspeed|seat ?belt|overturn"}

ACTIVITY = {  # first match wins
    "Hot work (welding/cutting/grinding)": r"weld|grind|cutting torch|hot work|spark",
    "Excavation": r"excavat|digging|trench",
    "Lifting / crane operations": r"crane|lifting|sling|hoist|banksman",
    "Tank / vessel entry": r"entered tank|sump|confined space|tank entry",
    "Electrical work": r"electric|mcc|panel|terminals",
    "Pump / valve maintenance": r"pump|valve|flange|seal",
    "Drilling / rig floor handling": r"rig floor|drill pipe|tong|derrick|tripping",
    "Work at height / scaffolding": r"scaffold|ladder|harness|roof|height|platform",
    "Road transport / driving": r"driv|vehicle|tanker|overspeed",
    "High-pressure hose / line": r"hose|pressure|whip",
    "Wellhead operations": r"wellhead|workover|well servicing",
    "Safety system maintenance": r"interlock|alarm|detector|shutdown"}
BARRIER = {
    "Work permit missing": r"without (a )?(hot work |entry |work )?permit|permit to work|no work permit",
    "Isolation not verified": r"isolation (was )?not verified|not isolated|without (a )?lockout|no tag-?out",
    "Gas test not done": r"gas test not done|no gas test|without .{0,30}gas test|not tested|lel not checked",
    "Fall protection missing": r"harness|lifeline|handrail|fall protection|toe board",
    "Exclusion zone / barricade missing": r"exclusion zone|swing radius|under suspended|beneath the load|whip-check|line of discharge",
    "Safety device bypassed": r"bypass|inhibit|disabled|override",
    "Lifting equipment / plan deficiency": r"damaged sling|lifting plan|load chart|no banksman|outrigger",
    "Standby / rescue absent": r"standby|attendant absent|rescue equipment",
    "Fire watch / fire blanket missing": r"no fire watch|fire blanket",
    "Speed / seat belt / fatigue": r"overspeed|seat belt not|fatigue|mobile phone"}


def _first(patterns, text, default):
    for name, rx in patterns.items():
        if re.search(rx, text, re.I):
            return name
    return default


def extract(text):
    loc = next((l for l in LOCS if re.search(rf"\b{re.escape(l)}\b", text, re.I)), "Unspecified")
    return (_first(ACTIVITY, text, "Other / general"), loc,
            _first(BARRIER, text, "Not identified"))


def make_data(n=3000, seed=0, noise=0.0):
    """Synthetic UA/UC/near-miss/incident reports. noise = share of flipped SIF labels."""
    rng = random.Random(seed)
    bias = {s: [random.Random(sum(map(ord, s)) + i).random() ** 2 + .15 for i in range(len(SCEN))]
            for s in SITE_RATE}
    days = (pd.Timestamp("2026-09-24") - pd.Timestamp("2025-10-01")).days
    rows = []
    for i in range(n):
        site = rng.choices(list(SITE_RATE), weights=SITE_VOL)[0]
        sif = rng.random() < SITE_RATE[site]
        lsr_pick = rng.choices(list(SCEN), weights=bias[site])[0]
        if sif:
            lsr, t = lsr_pick, rng.choice(SCEN[lsr_pick][0])
        elif rng.random() < .4:
            lsr, t = OTHER, rng.choice(GENERAL)
        else:
            lsr, t = lsr_pick, rng.choice(SCEN[lsr_pick][1])
        rtype = rng.choice(list(PREFIX))
        text = " ".join(x for x in [PREFIX[rtype], t.format(loc=rng.choice(LOCS)), rng.choice(FILL)] if x)
        rows.append(dict(report_id=f"R{seed}-{i:05d}",
                         date=pd.Timestamp("2025-10-01") + pd.Timedelta(days=rng.randint(0, days)),
                         site=site, report_type=rtype, text=text, sif_true=sif,
                         sif_label=sif ^ (rng.random() < noise), lsr_true=lsr))
    return pd.DataFrame(rows)


FEEDBACK_COLS = ["report_id", "text", "predicted_sif", "predicted_lsr",
                  "confirmed_sif", "confirmed_lsr", "reviewer", "timestamp"]


class SIFEngine:
    """Wraps the two classifiers plus a reviewer-feedback loop.

    Real deployment: swap make_data() for OIL's labelled UA/UC, near-miss and
    incident reports (see README - Data & validation plan). The feedback loop
    below (record_feedback / retrain) is how the model would keep improving
    once HSE reviewers start confirming or correcting its tags on real data.
    """

    def __init__(self, seed=0, threshold=0.5):
        self._df = make_data(3000, seed, noise=0.04)
        self.threshold = threshold
        self.feedback = pd.DataFrame(columns=FEEDBACK_COLS)
        self._build()

    def _build(self):
        mk = lambda C: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                     LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
        self.sif, self.lsr = mk(3), mk(5)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pred = cross_val_predict(self.sif, self._df.text, self._df.sif_label, cv=cv)
        p, r, f, _ = precision_recall_fscore_support(self._df.sif_label, pred, average="binary")
        lp = cross_val_predict(self.lsr, self._df.text, self._df.lsr_true, cv=cv)
        self.metrics = dict(precision=p, recall=r, f1=f, lsr_accuracy=float((lp == self._df.lsr_true).mean()))
        self.sif.fit(self._df.text, self._df.sif_label)
        self.lsr.fit(self._df.text, self._df.lsr_true)

    @staticmethod
    def _kw(text):
        hits = {k: len(re.findall(rx, text, re.I)) for k, rx in LSR_KW.items()}
        best = max(hits, key=hits.get)
        return best if hits[best] else OTHER

    @staticmethod
    def confidence(prob):
        d = abs(prob - .5)
        return "High" if d >= .3 else "Medium" if d >= .12 else "Low"

    def analyze(self, df):
        d = df.copy().reset_index(drop=True)
        d["text"] = d["text"].fillna("").astype(str)
        d["sif_prob"] = self.sif.predict_proba(d.text)[:, 1]
        d["confidence"] = d.sif_prob.map(self.confidence)
        P, cls = self.lsr.predict_proba(d.text), self.lsr.classes_
        d["lsr"] = [cls[p.argmax()] if p.max() >= .4 else self._kw(t) for p, t in zip(P, d.text)]
        ex = pd.DataFrame(d.text.map(extract).tolist(), index=d.index,
                          columns=["activity", "location", "barrier_failure"])
        return pd.concat([d, ex], axis=1)

    def explain(self, text, k=6):
        vec, clf = self.sif.steps[0][1], self.sif.steps[1][1]
        contrib = vec.transform([text]).toarray()[0] * clf.coef_[0]
        names = vec.get_feature_names_out()
        return [(names[i], float(contrib[i])) for i in np.argsort(-contrib)[:k] if contrib[i] > 0]

    # ---- reviewer feedback loop ----
    def record_feedback(self, report_id, text, predicted_sif, predicted_lsr,
                        confirmed_sif, confirmed_lsr, reviewer=""):
        row = dict(report_id=report_id, text=text, predicted_sif=bool(predicted_sif),
                  predicted_lsr=predicted_lsr, confirmed_sif=bool(confirmed_sif),
                  confirmed_lsr=confirmed_lsr, reviewer=reviewer or "HSE reviewer",
                  timestamp=pd.Timestamp.now())
        self.feedback = pd.concat([self.feedback, pd.DataFrame([row])], ignore_index=True)
        return row

    def retrain(self):
        """Fold confirmed reviewer feedback back into training and refit.
        Corrections are duplicated so the model weights them like a small,
        trusted, human-verified batch on top of the synthetic base data."""
        if self.feedback.empty:
            return self.metrics
        fb = self.feedback.drop_duplicates("report_id", keep="last")
        extra = pd.DataFrame({"text": fb.text.astype(str), "sif_label": fb.confirmed_sif.astype(bool),
                              "lsr_true": fb.confirmed_lsr.astype(str)})
        extra = pd.concat([extra] * 3, ignore_index=True)  # upweight verified real-world corrections
        self._df = pd.concat([self._df, extra], ignore_index=True)
        self._build()
        return self.metrics
