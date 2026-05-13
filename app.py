import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fairlearn.metrics import demographic_parity_difference

warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_K = 10
ALPHA_GENDER = 0.05
ALPHA_RACE = 0.10

GENDER_OPTIONS = ["Female", "Male", "Non-binary"]
RACE_OPTIONS = ["Asian", "Black", "Hispanic", "Other", "White"]

SAMPLE_CSV_COLUMNS = [
    "resume_id", "resume_text", "skills_text",
    "education_text", "experience_text", "certifications_text",
    "gender", "race"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading scorer model...")
def load_scorer():
    return joblib.load(os.path.join(MODEL_DIR, "fair_scorer_model.joblib"))


def encode(texts, embedder):
    return embedder.encode(
        [clean_text(t) for t in texts],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def rowwise_cosine(a, b):
    return np.array([
        cosine_similarity(a[i].reshape(1, -1), b[i].reshape(1, -1))[0][0]
        for i in range(len(a))
    ])


def compute_features(df, job_text, embedder):
    """Embed all texts and compute cosine similarity features."""
    if "job_text" in df.columns:
        job_texts = df["job_text"].fillna(job_text).apply(clean_text).tolist()
    else:
        job_texts = [clean_text(job_text)] * len(df)
    job_emb = embedder.encode(job_texts, batch_size=32,
                              show_progress_bar=False, convert_to_numpy=True)

    resume_emb   = encode(df["resume_text"],         embedder)
    skills_emb   = encode(df["skills_text"],         embedder)
    edu_emb      = encode(df["education_text"],      embedder)
    exp_emb      = encode(df["experience_text"],     embedder)
    cert_emb     = encode(df["certifications_text"], embedder)

    df = df.copy()
    df["full_similarity"]           = rowwise_cosine(job_emb, resume_emb)
    df["skills_similarity"]         = rowwise_cosine(job_emb, skills_emb)
    df["education_similarity"]      = rowwise_cosine(job_emb, edu_emb)
    df["experience_similarity"]     = rowwise_cosine(job_emb, exp_emb)
    df["certifications_similarity"] = rowwise_cosine(job_emb, cert_emb)
    return df


FEATURE_COLS = [
    "full_similarity",
    "skills_similarity",
    "education_similarity",
    "experience_similarity",
    "certifications_similarity",
]


def apply_fair_scorer(df, scorer):
    df = df.copy()
    df["fair_score_raw"] = scorer.predict_proba(df[FEATURE_COLS])[:, 1]

    # demographic adjustments
    df["fair_score_adjusted"] = df["fair_score_raw"].copy()
    overall_mean = df["fair_score_raw"].mean()

    for g, grp_mean in df.groupby("gender")["fair_score_raw"].mean().items():
        adj = overall_mean - grp_mean
        df.loc[df["gender"] == g, "fair_score_adjusted"] += 0.10 * adj

    for r, grp_mean in df.groupby("race")["fair_score_raw"].mean().items():
        adj = overall_mean - grp_mean
        df.loc[df["race"] == r, "fair_score_adjusted"] += 0.05 * adj

    df["fair_score_adjusted"] = df["fair_score_adjusted"].clip(0, 1)

    # ranks — group by job_id if present (matches notebook behaviour)
    if "job_id" in df.columns:
        df["baseline_rank"] = df.groupby("job_id")["full_similarity"].rank(method="first", ascending=False).astype(int)
        df["fair_rank"]     = df.groupby("job_id")["fair_score_adjusted"].rank(method="first", ascending=False).astype(int)
    else:
        df["baseline_rank"] = df["full_similarity"].rank(method="first", ascending=False).astype(int)
        df["fair_rank"]     = df["fair_score_adjusted"].rank(method="first", ascending=False).astype(int)
    df["selected_baseline"] = (df["baseline_rank"] <= TOP_K).astype(int)
    df["selected_fair"]     = (df["fair_rank"] <= TOP_K).astype(int)
    return df


def _rerank_group(df_sorted, top_k, alpha_gender, alpha_race):
    gender_targets = df_sorted["gender"].value_counts(normalize=True).to_dict()
    race_targets   = df_sorted["race"].value_counts(normalize=True).to_dict()
    gender_counts  = {g: 0 for g in df_sorted["gender"].unique()}
    race_counts    = {r: 0 for r in df_sorted["race"].unique()}

    selected = []
    remaining = df_sorted.copy()

    for step in range(top_k):
        if len(remaining) == 0:
            break
        best_idx, best_score = None, -np.inf
        for idx, row in remaining.iterrows():
            g, r = row["gender"], row["race"]
            gp = gender_counts[g] / step if step > 0 else 0.0
            rp = race_counts[r]   / step if step > 0 else 0.0
            gpen = max(0.0, gp - gender_targets.get(g, 0.0))
            rpen = max(0.0, rp - race_targets.get(r, 0.0))
            score = row["fair_score_adjusted"] - alpha_gender * gpen - alpha_race * rpen
            if score > best_score:
                best_score, best_idx = score, idx
        chosen = remaining.loc[[best_idx]]
        selected.append(chosen)
        gender_counts[chosen.iloc[0]["gender"]] += 1
        race_counts[chosen.iloc[0]["race"]]     += 1
        remaining = remaining.drop(best_idx)

    final = pd.concat(selected + [remaining.sort_values("fair_score_adjusted", ascending=False)])
    final = final.reset_index(drop=True)
    final["reranked_position"] = np.arange(1, len(final) + 1)
    final["selected_reranked"] = (final["reranked_position"] <= top_k).astype(int)
    return final


def soft_rerank(df, top_k=TOP_K, alpha_gender=ALPHA_GENDER, alpha_race=ALPHA_RACE):
    df_sorted = df.sort_values("fair_score_adjusted", ascending=False).copy()
    if "job_id" in df_sorted.columns:
        parts = []
        for _, group in df_sorted.groupby("job_id"):
            parts.append(_rerank_group(group, top_k, alpha_gender, alpha_race))
        return pd.concat(parts).reset_index(drop=True)
    return _rerank_group(df_sorted, top_k, alpha_gender, alpha_race)


def dpd_or_nan(y_true, y_pred, sensitive):
    try:
        return demographic_parity_difference(y_true=y_true, y_pred=y_pred,
                                             sensitive_features=sensitive)
    except Exception:
        return float("nan")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fair Resume Screener", layout="wide")
st.title("Fair Resume Screening System")
st.caption("Bias-aware AI recruitment tool")

embedder = load_embedder()
scorer   = load_scorer()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top-K shortlist size", min_value=1, max_value=50, value=TOP_K)
    alpha_gender = st.slider("Gender fairness weight (a)", 0.0, 1.0, ALPHA_GENDER, 0.01)
    alpha_race   = st.slider("Race fairness weight (a)",   0.0, 1.0, ALPHA_RACE,   0.01)
    show_rerank  = st.checkbox("Apply soft re-ranking", value=True)
    st.divider()
    st.markdown("**Expected CSV columns**")
    st.code("\n".join(SAMPLE_CSV_COLUMNS))

# ── Resume upload ─────────────────────────────────────────────────────────────
st.subheader("1. Resume Data")

input_mode = st.radio("Input method", ["Use default dataset", "Upload CSV", "Manual entry"], horizontal=True)

job_text = ""

df_resumes = None

DEFAULT_CSV = os.path.join(BASE_DIR, "data", "resume_dataset.csv")

def apply_test_split(df):
    """Reproduce the exact train/val/test split used in the notebook (SEED=42, 70/15/15)."""
    _, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_relevant"])
    _, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label_relevant"])
    return test.reset_index(drop=True)

if input_mode == "Use default dataset":
    try:
        df_raw = pd.read_csv(DEFAULT_CSV)
        missing = [c for c in SAMPLE_CSV_COLUMNS if c not in df_raw.columns]
        if missing:
            st.error(f"Default CSV is missing columns: {missing}")
        else:
            if "label_relevant" in df_raw.columns:
                df_resumes = apply_test_split(df_raw)
                st.success(f"Loaded test split: {len(df_resumes)} resumes (from {len(df_raw)} total, matching notebook evaluation set).")
            else:
                df_resumes = df_raw
                st.success(f"Loaded {len(df_resumes)} resumes from resume_dataset.csv.")
            st.dataframe(df_resumes.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load resume_dataset.csv: {e}")

elif input_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            missing = [c for c in SAMPLE_CSV_COLUMNS if c not in df_raw.columns]
            if missing:
                st.error(f"CSV is missing columns: {missing}")
            else:
                if "label_relevant" in df_raw.columns:
                    df_resumes = apply_test_split(df_raw)
                    st.success(f"Loaded test split: {len(df_resumes)} resumes (from {len(df_raw)} total).")
                else:
                    df_resumes = df_raw
                    st.success(f"Loaded {len(df_resumes)} resumes.")
                st.dataframe(df_resumes.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

else:
    st.subheader("Job Description")
    job_text = st.text_area(
        "Paste the full job description here",
        height=180,
        placeholder="Job Title: Data Scientist\nCompany: TechCorp\nRequirements: Python, ML, SQL...",
    )
    st.info("Enter each resume manually. Click **Add another resume** to add more.")
    if "resume_rows" not in st.session_state:
        st.session_state.resume_rows = 1

    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("Add another resume"):
            st.session_state.resume_rows += 1
    with col_clear:
        if st.button("Clear all"):
            st.session_state.resume_rows = 1

    rows = []
    for i in range(st.session_state.resume_rows):
        with st.expander(f"Resume {i + 1}", expanded=(i == 0)):
            c1, c2 = st.columns(2)
            rid          = c1.text_input("Resume ID",   key=f"rid_{i}",  value=f"R{i+1:04d}")
            gender       = c2.selectbox("Gender",       GENDER_OPTIONS, key=f"gen_{i}")
            race         = c1.selectbox("Race",         RACE_OPTIONS,   key=f"race_{i}")
            resume_text  = st.text_area("Resume text",        key=f"rtxt_{i}", height=100)
            skills_text  = st.text_input("Skills",            key=f"sk_{i}")
            edu_text     = st.text_input("Education",         key=f"edu_{i}")
            exp_text     = st.text_area("Experience",         key=f"exp_{i}", height=80)
            cert_text    = st.text_input("Certifications",    key=f"cert_{i}")
            rows.append({
                "resume_id": rid, "gender": gender, "race": race,
                "resume_text": resume_text, "skills_text": skills_text,
                "education_text": edu_text, "experience_text": exp_text,
                "certifications_text": cert_text,
            })

    if any(r["resume_text"].strip() for r in rows):
        df_resumes = pd.DataFrame(rows)

# ── Run screening ─────────────────────────────────────────────────────────────
st.divider()
csv_has_job = df_resumes is not None and "job_text" in df_resumes.columns
can_run = df_resumes is not None and (bool(job_text) or csv_has_job)
run = st.button("Screen Resumes", type="primary", disabled=not can_run)

if run:
    if len(df_resumes) < 2:
        st.warning("Please provide at least 2 resumes for meaningful fairness analysis.")
    else:
        effective_job_text = job_text if job_text else df_resumes["job_text"].iloc[0]
        with st.spinner("Computing embeddings and scores..."):
            df_feat = compute_features(df_resumes, effective_job_text, embedder)
            df_scored = apply_fair_scorer(df_feat, scorer)
            if show_rerank:
                df_final = soft_rerank(df_scored, top_k=top_k,
                                       alpha_gender=alpha_gender, alpha_race=alpha_race)
            else:
                df_final = df_scored.copy()
                df_final["selected_reranked"] = df_final["selected_fair"]

        st.success("Screening complete!")

        # ── Results tabs ──────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["Rankings", "Fairness Metrics", "Charts"])

        with tab1:
            st.subheader(f"Top-{top_k} Shortlist")

            cols = ["resume_id", "gender", "race",
                    "full_similarity", "skills_similarity",
                    "fair_score_raw", "fair_score_adjusted",
                    "baseline_rank", "fair_rank"]

            # sort: shortlisted first (by score desc), then rejected (by score desc)
            display_df = df_final.sort_values(
                ["selected_reranked", "fair_score_adjusted"],
                ascending=[False, False]
            )[cols].copy()
            display_df.insert(0, "overall_rank", range(1, len(display_df) + 1))

            shortlist_ids = set(df_final.loc[df_final["selected_reranked"] == 1, "resume_id"].tolist())
            display_df["Shortlisted"] = display_df["resume_id"].apply(lambda x: "Yes" if x in shortlist_ids else "No")
            display_df.columns = display_df.columns.str.replace("_", " ").str.title()

            def highlight_shortlisted(row):
                colour = "background-color: #1a7a3c; color: white;" if row["Shortlisted"] == "Yes" else ""
                return [colour] * len(row)

            st.dataframe(
                display_df.style.apply(highlight_shortlisted, axis=1),
                use_container_width=True,
                height=420,
            )
            st.caption("Dark green rows = shortlisted candidates.")

            csv_out = df_final.to_csv(index=False).encode()
            st.download_button("Download full results CSV", csv_out,
                               file_name="screening_results.csv", mime="text/csv")

        with tab2:
            sel_col = "selected_reranked"
            y_true = df_final["label_relevant"] if "label_relevant" in df_final.columns else df_final[sel_col]

            base_dpd_g  = dpd_or_nan(y_true, df_final["selected_baseline"], df_final["gender"])
            base_dpd_r  = dpd_or_nan(y_true, df_final["selected_baseline"], df_final["race"])
            fair_dpd_g  = dpd_or_nan(y_true, df_final["selected_fair"],     df_final["gender"])
            fair_dpd_r  = dpd_or_nan(y_true, df_final["selected_fair"],     df_final["race"])
            rerank_dpd_g = dpd_or_nan(y_true, df_final[sel_col],            df_final["gender"])
            rerank_dpd_r = dpd_or_nan(y_true, df_final[sel_col],            df_final["race"])

            st.subheader("Demographic Parity Difference (DPD)")
            st.caption("Lower is fairer. A DPD of 0 means equal selection rates across groups.")

            st.markdown("**Gender DPD**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline",        f"{base_dpd_g:.3f}")
            c2.metric("Fair Scorer",     f"{fair_dpd_g:.3f}",
                      delta=f"{fair_dpd_g - base_dpd_g:+.3f}", delta_color="inverse")
            c3.metric("After Re-ranking" if show_rerank else "Fair Scorer (final)",
                      f"{rerank_dpd_g:.3f}",
                      delta=f"{rerank_dpd_g - base_dpd_g:+.3f}", delta_color="inverse")

            st.markdown("**Race DPD**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline",        f"{base_dpd_r:.3f}")
            c2.metric("Fair Scorer",     f"{fair_dpd_r:.3f}",
                      delta=f"{fair_dpd_r - base_dpd_r:+.3f}", delta_color="inverse")
            c3.metric("After Re-ranking" if show_rerank else "Fair Scorer (final)",
                      f"{rerank_dpd_r:.3f}",
                      delta=f"{rerank_dpd_r - base_dpd_r:+.3f}", delta_color="inverse")

            metrics_df = pd.DataFrame({
                "System":     ["Baseline (cosine)", "Fair Scorer", "After Re-ranking"],
                "Gender DPD": [base_dpd_g, fair_dpd_g, rerank_dpd_g],
                "Race DPD":   [base_dpd_r, fair_dpd_r, rerank_dpd_r],
            })
            st.dataframe(metrics_df.style.format({"Gender DPD": "{:.3f}", "Race DPD": "{:.3f}"}),
                         use_container_width=True, hide_index=True)

            st.subheader("Selection Rate by Gender")
            g_base = df_final.groupby("gender")["selected_baseline"].mean().rename("Baseline")
            g_fair = df_final.groupby("gender")[sel_col].mean().rename("Fair")
            st.dataframe(pd.concat([g_base, g_fair], axis=1).style.format("{:.1%}"),
                         use_container_width=True)

            st.subheader("Selection Rate by Race")
            r_base = df_final.groupby("race")["selected_baseline"].mean().rename("Baseline")
            r_fair = df_final.groupby("race")[sel_col].mean().rename("Fair")
            st.dataframe(pd.concat([r_base, r_fair], axis=1).style.format("{:.1%}"),
                         use_container_width=True)

        with tab3:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # gender selection rate
            g_compare = pd.DataFrame({
                "Baseline": df_final.groupby("gender")["selected_baseline"].mean(),
                "Fair":     df_final.groupby("gender")[sel_col].mean(),
            })
            g_compare.plot(kind="bar", ax=axes[0], color=["#aec6cf", "#1f77b4"])
            axes[0].set_title("Selection Rate by Gender")
            axes[0].set_ylabel("Rate")
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis="x", rotation=0)

            # race selection rate
            r_compare = pd.DataFrame({
                "Baseline": df_final.groupby("race")["selected_baseline"].mean(),
                "Fair":     df_final.groupby("race")[sel_col].mean(),
            })
            r_compare.plot(kind="bar", ax=axes[1], color=["#aec6cf", "#1f77b4"])
            axes[1].set_title("Selection Rate by Race")
            axes[1].set_ylabel("Rate")
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis="x", rotation=30)

            plt.tight_layout()
            st.pyplot(fig)

            # score distribution
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            for g in df_final["gender"].unique():
                subset = df_final[df_final["gender"] == g]["fair_score_adjusted"]
                ax2.hist(subset, bins=15, alpha=0.6, label=g)
            ax2.set_title("Fair Score Distribution by Gender")
            ax2.set_xlabel("Fair Score (adjusted)")
            ax2.set_ylabel("Count")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
