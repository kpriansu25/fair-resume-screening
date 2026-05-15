import os
import re
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fairlearn.metrics import demographic_parity_difference

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Constants (match notebook) ────────────────────────────────────────────────
TOP_K        = 10
ALPHA_GENDER = 0.05
ALPHA_RACE   = 0.1

FEATURE_COLS = [
    "full_similarity",
    "skills_similarity",
    "education_similarity",
    "experience_similarity",
    "certifications_similarity",
]

GENDER_OPTIONS = ["Female", "Male", "Non-binary"]
RACE_OPTIONS   = ["Asian", "Black", "Hispanic", "Other", "White"]

REQUIRED_COLS = [
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


def encode_texts(texts, embedder):
    cleaned = [clean_text(t) for t in texts]
    return embedder.encode(cleaned, batch_size=32, show_progress_bar=False, convert_to_numpy=True)


def rowwise_cosine(a, b):
    return np.array([
        cosine_similarity(a[i].reshape(1, -1), b[i].reshape(1, -1))[0][0]
        for i in range(len(a))
    ])


def apply_test_split(df):
    """80/10/10 stratified split matching notebook SEED=42."""
    _, temp = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label_relevant"])
    _, test = train_test_split(temp, test_size=0.5, random_state=SEED, stratify=temp["label_relevant"])
    return test.reset_index(drop=True)


def compute_features(df, job_text_fallback, embedder):
    df = df.copy()
    # clean all text columns upfront (notebook cell 15)
    for col in ["job_text", "resume_text", "skills_text", "education_text",
                "experience_text", "certifications_text"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # per-row job embeddings when job_text column present
    if "job_text" in df.columns:
        job_texts = df["job_text"].fillna(clean_text(job_text_fallback)).tolist()
    else:
        job_texts = [clean_text(job_text_fallback)] * len(df)

    job_emb  = embedder.encode(job_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    res_emb  = encode_texts(df["resume_text"],         embedder)
    sk_emb   = encode_texts(df["skills_text"],         embedder)
    edu_emb  = encode_texts(df["education_text"],      embedder)
    exp_emb  = encode_texts(df["experience_text"],     embedder)
    cert_emb = encode_texts(df["certifications_text"], embedder)

    df["full_similarity"]           = rowwise_cosine(job_emb, res_emb)
    df["skills_similarity"]         = rowwise_cosine(job_emb, sk_emb)
    df["education_similarity"]      = rowwise_cosine(job_emb, edu_emb)
    df["experience_similarity"]     = rowwise_cosine(job_emb, exp_emb)
    df["certifications_similarity"] = rowwise_cosine(job_emb, cert_emb)
    return df


def compute_baseline(df):
    """Notebook cells 31-36: baseline rank + selection per job."""
    df = df.copy()
    if "job_id" in df.columns:
        df["baseline_rank"] = df.groupby("job_id")["full_similarity"] \
            .rank(method="first", ascending=False)
    else:
        df["baseline_rank"] = df["full_similarity"] \
            .rank(method="first", ascending=False)
    df["selected_top_k"] = (df["baseline_rank"] <= TOP_K).astype(int)
    return df


def compute_fair_scores(df, scorer):
    """Notebook cells 46-49: fair score + adjustment + fair rank + selection."""
    df = df.copy()
    for col in ["fair_score_raw", "fair_score_adjusted", "fair_rank", "selected_top_k_fair"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df["fair_score_raw"] = scorer.predict_proba(df[FEATURE_COLS])[:, 1]

    df["fair_score_adjusted"] = df["fair_score_raw"].copy()
    overall_mean = df["fair_score_raw"].mean()

    # gender adjustment (notebook cell 47)
    gender_means = df.groupby("gender")["fair_score_raw"].mean().to_dict()
    for g, gm in gender_means.items():
        df.loc[df["gender"] == g, "fair_score_adjusted"] += 0.1 * (overall_mean - gm)

    # race adjustment (notebook cell 47)
    race_means = df.groupby("race")["fair_score_raw"].mean().to_dict()
    for r, rm in race_means.items():
        df.loc[df["race"] == r, "fair_score_adjusted"] += 0.05 * (overall_mean - rm)

    df["fair_score_adjusted"] = df["fair_score_adjusted"].clip(0, 1)

    # fair rank per job (notebook cell 48)
    if "job_id" in df.columns:
        df["fair_rank"] = df.groupby("job_id")["fair_score_adjusted"] \
            .rank(method="first", ascending=False)
    else:
        df["fair_rank"] = df["fair_score_adjusted"] \
            .rank(method="first", ascending=False)

    df["selected_top_k_fair"] = (df["fair_rank"] <= TOP_K).astype(int)
    return df


def soft_fair_rerank(df, top_k=TOP_K, alpha_gender=ALPHA_GENDER, alpha_race=ALPHA_RACE):
    """Notebook cells 51-52: greedy soft re-ranking per job."""
    def rerank_group(group_df):
        group_df = group_df.sort_values("fair_score_adjusted", ascending=False).copy()
        gender_targets = group_df["gender"].value_counts(normalize=True).to_dict()
        race_targets   = group_df["race"].value_counts(normalize=True).to_dict()
        gender_counts  = {g: 0 for g in group_df["gender"].unique()}
        race_counts    = {r: 0 for r in group_df["race"].unique()}

        selected  = []
        remaining = group_df.copy()

        for step in range(top_k):
            if len(remaining) == 0:
                break
            best_idx, best_score = None, -np.inf
            for idx, row in remaining.iterrows():
                g, r = row["gender"], row["race"]
                gp = gender_counts[g] / step if step > 0 else 0.0
                rp = race_counts[r]   / step if step > 0 else 0.0
                gender_penalty = max(0.0, gp - gender_targets.get(g, 0.0))
                race_penalty   = max(0.0, rp - race_targets.get(r, 0.0))
                score = row["fair_score_adjusted"] - alpha_gender * gender_penalty - alpha_race * race_penalty
                if score > best_score:
                    best_score, best_idx = score, idx
            chosen = remaining.loc[[best_idx]]
            selected.append(chosen)
            gender_counts[chosen.iloc[0]["gender"]] += 1
            race_counts[chosen.iloc[0]["race"]]     += 1
            remaining = remaining.drop(best_idx)

        result = pd.concat(selected + [remaining.sort_values("fair_score_adjusted", ascending=False)])
        result = result.reset_index(drop=True)
        result["reranked_position"]   = np.arange(1, len(result) + 1)
        result["selected_top_k_reranked"] = (result["reranked_position"] <= top_k).astype(int)
        return result

    if "job_id" in df.columns:
        parts = []
        for job_id, group in df.sort_values("fair_score_adjusted", ascending=False).groupby("job_id"):
            g = rerank_group(group)
            g["job_id"] = job_id
            parts.append(g)
        return pd.concat(parts, ignore_index=True)
    return rerank_group(df.sort_values("fair_score_adjusted", ascending=False).copy())


def dpd_safe(y_true, y_pred, sensitive):
    try:
        return demographic_parity_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    except Exception:
        return float("nan")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fair Resume Screener", layout="wide")
st.title("Fair Resume Screening System")
st.caption("Bias-aware AI recruitment tool — dissertation project")

embedder = load_embedder()
scorer   = load_scorer()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k        = st.number_input("Top-K shortlist size", min_value=1, max_value=50, value=TOP_K)
    alpha_gender = st.slider("Gender fairness weight", 0.0, 1.0, ALPHA_GENDER, 0.01)
    alpha_race   = st.slider("Race fairness weight",   0.0, 1.0, ALPHA_RACE,   0.01)
    show_rerank  = st.checkbox("Apply soft re-ranking", value=True)
    st.divider()
    st.markdown("**Required CSV columns**")
    st.code("\n".join(REQUIRED_COLS))

# ── Data input ────────────────────────────────────────────────────────────────
st.subheader("1. Resume Data")
input_mode = st.radio("Input method", ["Use default dataset", "Upload CSV", "Manual entry"], horizontal=True)

job_text  = ""
df_input  = None

DEFAULT_CSV = os.path.join(DATA_DIR, "resume_dataset.csv")

if input_mode == "Use default dataset":
    try:
        df_raw = pd.read_csv(DEFAULT_CSV)
        missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
        if missing:
            st.error(f"CSV is missing columns: {missing}")
        else:
            if "label_relevant" in df_raw.columns:
                df_input = apply_test_split(df_raw)
                st.success(f"Loaded test split: {len(df_input)} resumes from {len(df_raw)} total (80/10/10 split, seed=42).")
            else:
                df_input = df_raw.reset_index(drop=True)
                st.success(f"Loaded {len(df_input)} resumes.")
            st.dataframe(df_input.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load resume_dataset.csv: {e}")

elif input_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
            if missing:
                st.error(f"CSV is missing columns: {missing}")
            else:
                if "label_relevant" in df_raw.columns:
                    df_input = apply_test_split(df_raw)
                    st.success(f"Loaded test split: {len(df_input)} resumes from {len(df_raw)} total.")
                else:
                    df_input = df_raw.reset_index(drop=True)
                    st.success(f"Loaded {len(df_input)} resumes.")
                st.dataframe(df_input.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

else:
    st.subheader("Job Description")
    job_text = st.text_area(
        "Paste the job description here",
        height=150,
        placeholder="Job Title: Data Scientist\nRequirements: Python, ML, SQL...",
    )
    st.info("Enter resumes manually below.")
    if "resume_rows" not in st.session_state:
        st.session_state.resume_rows = 1
    c1, c2 = st.columns(2)
    if c1.button("Add another resume"):
        st.session_state.resume_rows += 1
    if c2.button("Clear all"):
        st.session_state.resume_rows = 1

    rows = []
    for i in range(st.session_state.resume_rows):
        with st.expander(f"Resume {i + 1}", expanded=(i == 0)):
            ca, cb = st.columns(2)
            rid         = ca.text_input("Resume ID",       key=f"rid_{i}",  value=f"R{i+1:04d}")
            gender      = cb.selectbox("Gender",           GENDER_OPTIONS,  key=f"gen_{i}")
            race        = ca.selectbox("Race",             RACE_OPTIONS,    key=f"race_{i}")
            resume_text = st.text_area("Resume text",      key=f"rtxt_{i}", height=100)
            skills      = st.text_input("Skills",          key=f"sk_{i}")
            edu         = st.text_input("Education",       key=f"edu_{i}")
            exp         = st.text_area("Experience",       key=f"exp_{i}",  height=80)
            cert        = st.text_input("Certifications",  key=f"cert_{i}")
            rows.append({
                "resume_id": rid, "gender": gender, "race": race,
                "resume_text": resume_text, "skills_text": skills,
                "education_text": edu, "experience_text": exp,
                "certifications_text": cert,
            })
    if any(r["resume_text"].strip() for r in rows):
        df_input = pd.DataFrame(rows)

# ── Run ───────────────────────────────────────────────────────────────────────
st.divider()
csv_has_job = df_input is not None and "job_text" in df_input.columns
can_run     = df_input is not None and (bool(job_text) or csv_has_job)
run = st.button("Screen Resumes", type="primary", disabled=not can_run)

if run:
    if len(df_input) < 2:
        st.warning("Please provide at least 2 resumes.")
    else:
        job_text_fallback = job_text if job_text else df_input["job_text"].iloc[0]

        with st.spinner("Computing embeddings and scores..."):
            df_feat     = compute_features(df_input, job_text_fallback, embedder)
            test_ranked = compute_baseline(df_feat)
            test_df     = compute_fair_scores(test_ranked, scorer)
            if show_rerank:
                reranked_df = soft_fair_rerank(test_df, top_k=top_k,
                                               alpha_gender=alpha_gender, alpha_race=alpha_race)
            else:
                reranked_df = test_df.copy()
                reranked_df["reranked_position"]      = reranked_df["fair_rank"]
                reranked_df["selected_top_k_reranked"] = reranked_df["selected_top_k_fair"]

        st.success("Screening complete!")

        has_label = "label_relevant" in reranked_df.columns
        y_true    = reranked_df["label_relevant"] if has_label else reranked_df["selected_top_k_fair"]

        tab1, tab2, tab3 = st.tabs(["Rankings", "Fairness Metrics", "Charts"])

        # ── Tab 1: Rankings ───────────────────────────────────────────────────
        with tab1:
            st.subheader(f"Top-{top_k} Shortlist (per job)")

            has_job = "job_id" in reranked_df.columns
            id_cols = (["job_id"] if has_job else []) + ["resume_id", "gender", "race"]
            score_cols = ["full_similarity", "fair_score_raw", "fair_score_adjusted"]
            rank_cols  = ["baseline_rank", "fair_rank", "reranked_position"]

            disp = reranked_df[id_cols + score_cols + rank_cols + ["selected_top_k_reranked"]].copy()
            disp["Shortlisted"] = disp["selected_top_k_reranked"].map({1: "Yes", 0: "No"})

            # sort: shortlisted first within each job, then by fair_rank
            sort_keys = (["job_id", "selected_top_k_reranked", "fair_rank"]
                         if has_job else ["selected_top_k_reranked", "fair_rank"])
            disp = disp.sort_values(
                sort_keys,
                ascending=([True, False, True] if has_job else [False, True])
            ).reset_index(drop=True)
            disp.drop(columns=["selected_top_k_reranked"], inplace=True)

            disp.columns = disp.columns.str.replace("_", " ").str.title()

            def highlight_shortlisted(row):
                style = "background-color: #1a7a3c; color: white;" if row["Shortlisted"] == "Yes" else ""
                return [style] * len(row)

            st.dataframe(
                disp.style.apply(highlight_shortlisted, axis=1),
                use_container_width=True,
                height=440,
            )
            st.caption("Dark green = shortlisted (top-k per job).")

            csv_out = reranked_df.to_csv(index=False).encode()
            st.download_button("Download full results CSV", csv_out,
                               file_name="screening_results.csv", mime="text/csv")

        # ── Tab 2: Fairness Metrics ───────────────────────────────────────────
        with tab2:
            st.subheader("Demographic Parity Difference (DPD)")
            st.caption("Lower is fairer. 0 = equal selection rates across groups.")

            base_dpd_g   = dpd_safe(y_true, reranked_df["selected_top_k"],          reranked_df["gender"])
            base_dpd_r   = dpd_safe(y_true, reranked_df["selected_top_k"],          reranked_df["race"])
            fair_dpd_g   = dpd_safe(y_true, reranked_df["selected_top_k_fair"],     reranked_df["gender"])
            fair_dpd_r   = dpd_safe(y_true, reranked_df["selected_top_k_fair"],     reranked_df["race"])
            rerank_dpd_g = dpd_safe(y_true, reranked_df["selected_top_k_reranked"], reranked_df["gender"])
            rerank_dpd_r = dpd_safe(y_true, reranked_df["selected_top_k_reranked"], reranked_df["race"])

            st.markdown("**Gender DPD**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline", f"{base_dpd_g:.3f}")
            c2.metric("Fair Scorer", f"{fair_dpd_g:.3f}",
                      delta=f"{fair_dpd_g - base_dpd_g:+.3f}", delta_color="inverse")
            c3.metric("After Re-ranking", f"{rerank_dpd_g:.3f}",
                      delta=f"{rerank_dpd_g - base_dpd_g:+.3f}", delta_color="inverse")

            st.markdown("**Race DPD**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline", f"{base_dpd_r:.3f}")
            c2.metric("Fair Scorer", f"{fair_dpd_r:.3f}",
                      delta=f"{fair_dpd_r - base_dpd_r:+.3f}", delta_color="inverse")
            c3.metric("After Re-ranking", f"{rerank_dpd_r:.3f}",
                      delta=f"{rerank_dpd_r - base_dpd_r:+.3f}", delta_color="inverse")

            comparison_df = pd.DataFrame({
                "System":     ["Baseline", "Fair Scorer", "After Re-ranking"],
                "Gender DPD": [base_dpd_g, fair_dpd_g, rerank_dpd_g],
                "Race DPD":   [base_dpd_r, fair_dpd_r, rerank_dpd_r],
            })
            st.dataframe(
                comparison_df.style.format({"Gender DPD": "{:.3f}", "Race DPD": "{:.3f}"}),
                use_container_width=True, hide_index=True
            )

            st.subheader("Selection Rate by Gender")
            g_sel = pd.concat([
                reranked_df.groupby("gender")["selected_top_k"].mean().rename("Baseline"),
                reranked_df.groupby("gender")["selected_top_k_fair"].mean().rename("Fair Scorer"),
                reranked_df.groupby("gender")["selected_top_k_reranked"].mean().rename("Re-ranked"),
            ], axis=1)
            st.dataframe(g_sel.style.format("{:.1%}"), use_container_width=True)

            st.subheader("Selection Rate by Race")
            r_sel = pd.concat([
                reranked_df.groupby("race")["selected_top_k"].mean().rename("Baseline"),
                reranked_df.groupby("race")["selected_top_k_fair"].mean().rename("Fair Scorer"),
                reranked_df.groupby("race")["selected_top_k_reranked"].mean().rename("Re-ranked"),
            ], axis=1)
            st.dataframe(r_sel.style.format("{:.1%}"), use_container_width=True)

        # ── Tab 3: Charts ─────────────────────────────────────────────────────
        with tab3:
            sns.set_theme(style="whitegrid", context="talk")

            # DPD comparison (notebook cell 57)
            plot_df = comparison_df.melt(
                id_vars="System", value_vars=["Gender DPD", "Race DPD"],
                var_name="Metric", value_name="DPD"
            )
            fig0, ax0 = plt.subplots(figsize=(10, 5))
            custom_palette = {"Gender DPD": "#147d8b", "Race DPD": "#606060"}
            sns.barplot(data=plot_df, x="System", y="DPD", hue="Metric",
                        palette=custom_palette, ax=ax0)
            for container in ax0.containers:
                ax0.bar_label(container, fmt="%.3f", padding=3, fontsize=11)
            gender_patch = mpatches.Patch(color="#147d8b", label="Gender DPD")
            race_patch   = mpatches.Patch(color="#606060", label="Race DPD")
            ax0.legend(handles=[gender_patch, race_patch], title="Metric")
            ax0.set_title("Demographic Parity Difference Across Systems")
            ax0.set_ylabel("DPD")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig0)

            # Gender selection rate (notebook cell 58)
            gender_baseline = reranked_df.groupby("gender")["selected_top_k"].mean() \
                .reset_index(name="selection_rate")
            gender_baseline["system"] = "Baseline"
            gender_fair = reranked_df.groupby("gender")["selected_top_k_fair"].mean() \
                .reset_index(name="selection_rate")
            gender_fair["system"] = "Fair Scorer"
            gender_reranked = reranked_df.groupby("gender")["selected_top_k_reranked"].mean() \
                .reset_index(name="selection_rate")
            gender_reranked["system"] = "Re-ranked"
            gender_compare = pd.concat([gender_baseline, gender_fair, gender_reranked], ignore_index=True)

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=gender_compare, x="gender", y="selection_rate",
                        hue="system", palette="Set1", ax=ax1)
            for container in ax1.containers:
                ax1.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
            ax1.set_title("Selection Rate by Gender")
            ax1.set_xlabel("Gender")
            ax1.set_ylabel("Selection Rate in Top-10")
            ax1.set_ylim(0, gender_compare["selection_rate"].max() + 0.1)
            plt.tight_layout()
            st.pyplot(fig1)

            # Race selection rate (notebook cell 59)
            race_baseline = reranked_df.groupby("race")["selected_top_k"].mean() \
                .reset_index(name="selection_rate")
            race_baseline["system"] = "Baseline"
            race_fair = reranked_df.groupby("race")["selected_top_k_fair"].mean() \
                .reset_index(name="selection_rate")
            race_fair["system"] = "Fair Scorer"
            race_reranked = reranked_df.groupby("race")["selected_top_k_reranked"].mean() \
                .reset_index(name="selection_rate")
            race_reranked["system"] = "Re-ranked"
            race_compare = pd.concat([race_baseline, race_fair, race_reranked], ignore_index=True)

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            sns.barplot(data=race_compare, x="race", y="selection_rate",
                        hue="system", palette="Set2", ax=ax2)
            for container in ax2.containers:
                ax2.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
            ax2.set_title("Selection Rate by Race")
            ax2.set_xlabel("Race")
            ax2.set_ylabel("Selection Rate in Top-10")
            ax2.set_ylim(0, race_compare["selection_rate"].max() + 0.1)
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig2)

            # Score distribution by gender
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            for g in reranked_df["gender"].unique():
                subset = reranked_df[reranked_df["gender"] == g]["fair_score_adjusted"]
                ax3.hist(subset, bins=15, alpha=0.6, label=g)
            ax3.set_title("Fair Score Distribution by Gender")
            ax3.set_xlabel("Fair Score (adjusted)")
            ax3.set_ylabel("Count")
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3)
