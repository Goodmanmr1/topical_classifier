# To run:
#   pip install streamlit scikit-learn pandas numpy openai google-generativeai
#   streamlit run topical_keyword_analysis_app.py

import io
import re
import json
from collections import Counter
from typing import List, Dict, Set, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# Stopword & intent logic
# -----------------------------

GRAMMAR_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "and", "or", "to", "from", "by",
    "on", "in", "at", "for", "with", "about", "as",
    "is", "are", "was", "were", "be", "been", "being",
}

INTENT_TOKENS: Set[str] = {
    "near", "me", "nearby",
    "for", "with", "without",
    "vs", "versus",
    "best", "top", "cheap", "cheapest", "free",
    "how", "what", "where", "when", "why", "can", "should",
    "things", "do",
}


# Configuration Constants
class LLMConfig:
    """Configuration for LLM operations."""
    MAX_PHRASES_FOR_LABEL = 8
    MAX_EXAMPLES_FOR_LABEL = 5
    MAX_TOKENS_FOR_LABEL = 32
    MAX_TOKENS_FOR_QA = 256
    DEFAULT_TEMPERATURE = 0.2
    TEMPERATURE_CREATIVE = 0.3


class ClusteringConfig:
    """Configuration for clustering."""
    DEFAULT_N_CLUSTERS = 12
    MAX_CLUSTERS_FOR_QA = 5
    MAX_PHRASES_PER_CLUSTER = 15
    MAX_KEYWORDS_PER_CLUSTER = 20


# -----------------------------
# Basic text utilities
# -----------------------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_grammar_stopwords: bool = True) -> List[str]:
    norm = normalize_text(text)
    tokens = norm.split()
    if remove_grammar_stopwords:
        tokens = [t for t in tokens if t not in GRAMMAR_STOPWORDS]
    return tokens


# -----------------------------
# API key helper
# -----------------------------

def get_api_key_from_secrets_or_input(
    provider: str,
    default_key_name: str,
    sidebar_label: str,
) -> Optional[str]:
    default_key = st.secrets.get(default_key_name, "")
    key = st.sidebar.text_input(
        sidebar_label,
        value=default_key,
        type="password",
        help=f"Leave empty to disable {provider} features.",
    )
    key = key.strip()
    return key or None


# -----------------------------
# Embeddings + LLM helpers
# -----------------------------

def parse_llm_json_response(raw: str, required_keys: Optional[List[str]] = None) -> Dict:
    """
    Parse JSON response from LLM with validation and error handling.
    
    Args:
        raw: Raw response string from LLM
        required_keys: List of required keys in the JSON response
    
    Returns:
        Parsed dictionary
    
    Raises:
        ValueError: If JSON is malformed or missing required keys
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("LLM response is empty or not a string")
    
    raw = raw.strip()
    
    try:
        # Try to extract JSON if wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw
        
        parsed = json.loads(json_str)
        
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        
        # Validate required keys
        if required_keys:
            missing = set(required_keys) - set(parsed.keys())
            if missing:
                raise ValueError(f"Missing required keys: {missing}")
        
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)[:100]}")


def get_embeddings(
    texts: List[str],
    provider: str,
    model: str,
    api_key: str,
) -> np.ndarray:
    if provider == "OpenAI":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install OpenAI client: pip install openai"
            ) from e

        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=model, input=texts)
        embs = np.array([d.embedding for d in resp.data])
        return embs

    elif provider == "Gemini":
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "Install Gemini client: pip install google-generativeai"
            ) from e

        genai.configure(api_key=api_key)
        embs: List[List[float]] = []
        for t in texts:
            resp = genai.embed_content(model=model, content=t)
            if isinstance(resp, dict):
                embedding = resp.get("embedding")
            else:
                embedding = getattr(resp, "embedding", None)
            if embedding is None:
                raise ValueError("Unexpected Gemini embed_content response.")
            embs.append(embedding)
        return np.array(embs)

    else:
        raise ValueError(f"Unknown provider for embeddings: {provider}")


def llm_chat(
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    if provider == "OpenAI":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install OpenAI client: pip install openai"
            ) from e
        client = OpenAI(api_key=api_key)
        # Build request kwargs - GPT-5 models don't support temperature parameter
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": max_tokens,
        }
        
        # Only include temperature for models that support it (not GPT-5 series)
        if not any(gpt5_model in model for gpt5_model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]):
            kwargs["temperature"] = temperature
        
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    elif provider == "Gemini":
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "Install Gemini client: pip install google-generativeai"
            ) from e
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        
        # Gemini expects system_prompt as a separate system instruction
        # We'll prepend it to the user message for proper handling
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            resp = model_obj.generate_content(combined_prompt)
            if not resp or not hasattr(resp, 'text') or not resp.text:
                raise ValueError("Empty or invalid response from Gemini API")
            return resp.text.strip()
        except AttributeError as ae:
            raise ValueError(f"Unexpected Gemini response format: {ae}") from ae

    else:
        raise ValueError(f"Unknown provider for LLM: {provider}")


def llm_label_cluster(
    provider: str,
    model: str,
    api_key: str,
    cluster_phrases: List[str],
    example_keywords: List[str],
    max_len_phrases: int = None,
    max_len_examples: int = None,
) -> str:
    if max_len_phrases is None:
        max_len_phrases = LLMConfig.MAX_PHRASES_FOR_LABEL
    if max_len_examples is None:
        max_len_examples = LLMConfig.MAX_EXAMPLES_FOR_LABEL
        
    phrases_snippet = cluster_phrases[:max_len_phrases]
    examples_snippet = example_keywords[:max_len_examples]

    system_prompt = "You are an expert SEO topical classifier."
    user_prompt = (
        "I will give you phrases that belong to the same topical cluster, "
        "and some example search keywords that contain them.\n\n"
        "Phrases:\n"
        "- " + "\n- ".join(phrases_snippet) + "\n\n"
        "Example keywords:\n"
        "- " + "\n- ".join(examples_snippet) + "\n\n"
        "Task: Propose a very short, human-readable category label that generalizes these.\n"
        "Rules:\n"
        "- 3–5 words max.\n"
        "- No quotes, no punctuation at the ends.\n"
        "- Use a conceptual label, not a copy of one phrase.\n"
        "Answer with ONLY the label."
    )

    return llm_chat(
        provider=provider,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=LLMConfig.MAX_TOKENS_FOR_LABEL,
        temperature=LLMConfig.TEMPERATURE_CREATIVE,
    )


# -----------------------------
# Phrase extraction
# -----------------------------

def build_phrase_to_keywords_map(
    df_keywords: pd.DataFrame,
    keyword_col: str,
    phrases: List[str],
) -> Dict[str, List[str]]:
    """
    Build mapping from phrases to keywords efficiently.
    
    Uses optimized substring matching to avoid O(N*M) complexity.
    
    Args:
        df_keywords: DataFrame with keywords
        keyword_col: Name of keyword column
        phrases: List of phrases to match
    
    Returns:
        Dictionary mapping phrase -> list of matching keywords
    """
    if not phrases or df_keywords.empty:
        return {}
    
    phrase_to_keywords: Dict[str, List[str]] = {p: [] for p in phrases}
    
    # Escape special regex characters and create pattern
    import re as regex_module
    escaped_phrases = [regex_module.escape(p) for p in phrases]
    pattern = regex_module.compile("|".join(escaped_phrases))
    
    for _, row in df_keywords.iterrows():
        kw = str(row[keyword_col])
        norm_kw = normalize_text(kw)
        
        # Find all matching phrases in one pass
        matches = set(pattern.findall(norm_kw))
        for match in matches:
            phrase_to_keywords[match].append(kw)
    
    # Remove empty entries
    return {k: v for k, v in phrase_to_keywords.items() if v}


def extract_phrases_from_keywords(
    keywords: List[str],
    include_unigrams: bool = True,
    include_bigrams: bool = True,
    include_trigrams: bool = True,
    min_freq: int = 3,
) -> pd.DataFrame:
    n_values: List[int] = []
    if include_unigrams:
        n_values.append(1)
    if include_bigrams:
        n_values.append(2)
    if include_trigrams:
        n_values.append(3)

    counter: Counter = Counter()

    for kw in keywords:
        tokens = tokenize(kw, remove_grammar_stopwords=True)
        if not tokens:
            continue
        for n in n_values:
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i + n])
                counter[(phrase, n)] += 1

    rows = [
        {"phrase": phrase, "n": n, "count": count}
        for (phrase, n), count in counter.items()
        if count >= min_freq
    ]

    if not rows:
        return pd.DataFrame(columns=["phrase", "n", "count"])

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


# -----------------------------
# Topic clustering
# -----------------------------

def build_topic_clusters_tfidf(
    phrases_df: pd.DataFrame,
    n_clusters: int = 10,
    max_phrases: int = 300,
) -> pd.DataFrame:
    if phrases_df.empty:
        return phrases_df

    # Only copy when we'll modify the dataframe
    phrases_subset = phrases_df.sort_values("count", ascending=False).head(max_phrases)
    phrases = phrases_subset["phrase"].tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    X = vectorizer.fit_transform(phrases)

    n_clusters = max(1, min(n_clusters, len(phrases)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Now copy for modification
    result = phrases_subset.copy()
    result.reset_index(drop=True, inplace=True)
    result["cluster_id"] = labels

    cluster_labels: Dict[int, str] = {}
    for cid in sorted(set(labels)):
        cluster_phrases = result[result["cluster_id"] == cid] \
            .sort_values("count", ascending=False)
        top_examples = cluster_phrases["phrase"].head(3).tolist()
        label = ", ".join(top_examples)
        cluster_labels[cid] = label

    result["cluster_label_preview"] = result["cluster_id"].map(cluster_labels)
    return result


def build_topic_clusters_semantic(
    phrases_df: pd.DataFrame,
    n_clusters: int,
    max_phrases: int,
    emb_provider: str,
    emb_model: str,
    emb_api_key: str,
) -> pd.DataFrame:
    if phrases_df.empty:
        return phrases_df

    # Only copy when we'll modify the dataframe
    phrases_subset = phrases_df.sort_values("count", ascending=False).head(max_phrases)
    phrases = phrases_subset["phrase"].tolist()

    embs = get_embeddings(
        texts=phrases,
        provider=emb_provider,
        model=emb_model,
        api_key=emb_api_key,
    )

    n_clusters = max(1, min(n_clusters, len(phrases)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embs)

    # Now copy for modification
    result = phrases_subset.copy()
    result.reset_index(drop=True, inplace=True)
    result["cluster_id"] = labels

    cluster_labels: Dict[int, str] = {}
    for cid in sorted(set(labels)):
        cluster_phrases = result[result["cluster_id"] == cid] \
            .sort_values("count", ascending=False)
        top_examples = cluster_phrases["phrase"].head(3).tolist()
        label = ", ".join(top_examples)
        cluster_labels[cid] = label

    result["cluster_label_preview"] = result["cluster_id"].map(cluster_labels)
    return result


def generate_llm_cluster_labels(
    phrases_df: pd.DataFrame,
    df_keywords: pd.DataFrame,
    keyword_col: str,
    provider: str,
    chat_model: str,
    api_key: str,
    max_keywords_per_cluster: int = 20,
) -> pd.DataFrame:
    if phrases_df.empty:
        phrases_df["cluster_label_llm"] = None
        return phrases_df

    phrases_df = phrases_df.copy()
    cluster_labels_llm: Dict[int, str] = {}

    # Use optimized phrase-to-keywords mapping
    phrases_list = phrases_df["phrase"].unique().tolist()
    phrase_to_keywords = build_phrase_to_keywords_map(df_keywords, keyword_col, phrases_list)

    for cid, group in phrases_df.groupby("cluster_id"):
        cluster_phrases = group.sort_values("count", ascending=False)["phrase"].tolist()

        example_kw_set: List[str] = []
        for phrase in cluster_phrases:
            kws = phrase_to_keywords.get(phrase, [])
            for k in kws:
                if k not in example_kw_set:
                    example_kw_set.append(k)
                if len(example_kw_set) >= max_keywords_per_cluster:
                    break
            if len(example_kw_set) >= max_keywords_per_cluster:
                break

        if not example_kw_set:
            example_kw_set = ["(no direct keyword examples, phrases only)"]

        try:
            label = llm_label_cluster(
                provider=provider,
                model=chat_model,
                api_key=api_key,
                cluster_phrases=cluster_phrases,
                example_keywords=example_kw_set,
            )
        except Exception as e:
            label = group["cluster_label_preview"].iloc[0]
            st.warning(f"LLM labeling failed for cluster {cid}: {e}")

        cluster_labels_llm[cid] = label

    phrases_df["cluster_label_llm"] = phrases_df["cluster_id"].map(cluster_labels_llm)
    return phrases_df


def tag_keywords_with_topics(
    df_keywords: pd.DataFrame,
    keyword_col: str,
    topic_phrases_df: pd.DataFrame,
    use_llm_labels: bool = True,
) -> pd.DataFrame:
    df_keywords = df_keywords.copy()

    if topic_phrases_df.empty:
        df_keywords["Topic_Phrases"] = None
        df_keywords["Topic_Clusters"] = None
        return df_keywords

    if use_llm_labels and "cluster_label_qa" in topic_phrases_df.columns and topic_phrases_df["cluster_label_qa"].notna().any():
        label_col = "cluster_label_qa"
    elif use_llm_labels and "cluster_label_llm" in topic_phrases_df.columns and topic_phrases_df["cluster_label_llm"].notna().any():
        label_col = "cluster_label_llm"
    else:
        label_col = "cluster_label_preview"

    phrase_to_cluster = dict(
        zip(topic_phrases_df["phrase"], topic_phrases_df[label_col])
    )
    phrases = list(phrase_to_cluster.keys())

    df_keywords["__normalized_keyword"] = df_keywords[keyword_col].astype(str).apply(normalize_text)

    matched_phrases_col: List[Optional[str]] = []
    matched_clusters_col: List[Optional[str]] = []

    for _, row in df_keywords.iterrows():
        norm_kw = row["__normalized_keyword"]
        matched_phrases = []
        matched_clusters = set()

        for phrase in phrases:
            if phrase in norm_kw:
                matched_phrases.append(phrase)
                matched_clusters.add(phrase_to_cluster[phrase])

        if matched_phrases:
            matched_phrases_col.append(" | ".join(sorted(set(matched_phrases))))
            matched_clusters_col.append(" | ".join(sorted(matched_clusters)))
        else:
            matched_phrases_col.append(None)
            matched_clusters_col.append(None)

    df_keywords["Topic_Phrases"] = matched_phrases_col
    df_keywords["Topic_Clusters"] = matched_clusters_col
    df_keywords.drop(columns=["__normalized_keyword"], inplace=True, errors="ignore")
    return df_keywords


# -----------------------------
# Intent / modifier mining (Pass 2)
# -----------------------------

def extract_intent_phrases(
    keywords: List[str],
    min_freq: int = 3,
) -> pd.DataFrame:
    counter: Counter = Counter()

    for kw in keywords:
        tokens = tokenize(kw, remove_grammar_stopwords=False)
        if not tokens:
            continue
        for n in (2, 3):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i + n]
                if any(t in INTENT_TOKENS for t in phrase_tokens):
                    phrase = " ".join(phrase_tokens)
                    counter[(phrase, n)] += 1

    rows = [
        {"intent_phrase": phrase, "n": n, "count": count}
        for (phrase, n), count in counter.items()
        if count >= min_freq
    ]

    if not rows:
        return pd.DataFrame(columns=["intent_phrase", "n", "count"])

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


def tag_keywords_with_intent(
    df_keywords: pd.DataFrame,
    keyword_col: str,
    intent_df: pd.DataFrame,
    top_n_phrases: int = 50,
) -> pd.DataFrame:
    df_keywords = df_keywords.copy()

    if intent_df.empty:
        df_keywords["Intent_Phrases"] = None
        return df_keywords

    df_keywords["__normalized_keyword"] = df_keywords[keyword_col].astype(str).apply(normalize_text)

    intent_df = intent_df.sort_values("count", ascending=False).head(top_n_phrases)
    intent_phrases = intent_df["intent_phrase"].tolist()

    tagged_intent: List[Optional[str]] = []
    for _, row in df_keywords.iterrows():
        norm_kw = row["__normalized_keyword"]
        found = []
        for phrase in intent_phrases:
            if phrase in norm_kw:
                found.append(phrase)
        tagged_intent.append(" | ".join(sorted(set(found))) if found else None)

    df_keywords["Intent_Phrases"] = tagged_intent
    df_keywords.drop(columns=["__normalized_keyword"], inplace=True, errors="ignore")
    return df_keywords


# -----------------------------
# LLM QA: cluster-level
# -----------------------------

def run_cluster_qa(
    phrases_df: pd.DataFrame,
    df_keywords: pd.DataFrame,
    keyword_col: str,
    provider: str,
    chat_model: str,
    api_key: str,
    max_clusters_for_qa: int = None,
    max_phrases_per_cluster: int = None,
    max_keywords_per_cluster: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For up to max_clusters_for_qa clusters, ask LLM to:
      - Suggest improved label
      - Flag misfit phrases
      - Provide notes / subthemes

    Returns:
      - updated phrases_df with cluster_label_qa & qa flags
      - cluster_qa_summary_df for display
    """
    if max_clusters_for_qa is None:
        max_clusters_for_qa = ClusteringConfig.MAX_CLUSTERS_FOR_QA
    if max_phrases_per_cluster is None:
        max_phrases_per_cluster = ClusteringConfig.MAX_PHRASES_PER_CLUSTER
    if max_keywords_per_cluster is None:
        max_keywords_per_cluster = ClusteringConfig.MAX_KEYWORDS_PER_CLUSTER
    if phrases_df.empty:
        phrases_df["cluster_label_qa"] = None
        phrases_df["qa_misfit_flag"] = False
        return phrases_df, pd.DataFrame()

    phrases_df = phrases_df.copy()
    phrases_df["cluster_label_qa"] = None
    phrases_df["qa_misfit_flag"] = False

    summary_rows = []

    # Choose clusters with largest total occurrences
    cluster_stats = []
    for cid, group in phrases_df.groupby("cluster_id"):
        total_count = group["count"].sum()
        cluster_stats.append((cid, total_count))
    cluster_stats.sort(key=lambda x: x[1], reverse=True)
    clusters_to_review = [cid for cid, _ in cluster_stats[:max_clusters_for_qa]]

    # Build phrase -> keywords map for context using optimized function
    phrase_to_keywords = build_phrase_to_keywords_map(
        df_keywords, 
        keyword_col, 
        phrases_df["phrase"].unique().tolist()
    )

    for cid in clusters_to_review:
        group = phrases_df[phrases_df["cluster_id"] == cid].sort_values("count", ascending=False)
        phrases = group["phrase"].tolist()
        phrases_snippet = phrases[:max_phrases_per_cluster]

        example_kw_set: List[str] = []
        for phrase in phrases_snippet:
            kws = phrase_to_keywords.get(phrase, [])
            for k in kws:
                if k not in example_kw_set:
                    example_kw_set.append(k)
                if len(example_kw_set) >= max_keywords_per_cluster:
                    break
            if len(example_kw_set) >= max_keywords_per_cluster:
                break
        if not example_kw_set:
            example_kw_set = ["(no direct keyword examples, phrases only)"]

        system_prompt = (
            "You are an expert SEO topical classifier auditing topic clusters.\n"
            "You will be given phrases that belong to a single cluster and example search keywords.\n"
        )
        user_prompt = (
            "Cluster phrases:\n"
            "- " + "\n- ".join(phrases_snippet) + "\n\n"
            "Example keywords:\n"
            "- " + "\n- ".join(example_kw_set) + "\n\n"
            "Your tasks:\n"
            "1) Propose a short conceptual label for this cluster (3–5 words).\n"
            "2) List any phrases from the list that look like they do NOT belong in this cluster.\n"
            "3) Provide a short note on any obvious subthemes.\n\n"
            "Respond ONLY in JSON with keys: label, misfit_phrases, notes, subthemes.\n"
            "Example format:\n"
            "{\n"
            '  \"label\": \"Park Type\",\n'
            '  \"misfit_phrases\": [\"phrase1\", \"phrase2\"],\n'
            '  \"notes\": \"Short comment\",\n'
            '  \"subthemes\": [\"Water Parks\", \"Theme Parks\"]\n'
            "}"
        )

        try:
            raw = llm_chat(
                provider=provider,
                model=chat_model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=LLMConfig.MAX_TOKENS_FOR_QA,
                temperature=LLMConfig.DEFAULT_TEMPERATURE,
            )
            # Use safe JSON parsing with required keys validation
            parsed = parse_llm_json_response(
                raw,
                required_keys=["label", "misfit_phrases", "notes", "subthemes"]
            )
        except (ValueError, json.JSONDecodeError) as e:
            st.warning(f"Cluster QA failed for cluster {cid}: {str(e)[:150]}")
            parsed = {
                "label": group["cluster_label_preview"].iloc[0],
                "misfit_phrases": [],
                "notes": "LLM QA failed, using preview label.",
                "subthemes": [],
            }

        label_qa = parsed.get("label") or group["cluster_label_preview"].iloc[0]
        misfit_phrases = parsed.get("misfit_phrases") or []
        notes = parsed.get("notes") or ""
        subthemes = parsed.get("subthemes") or []

        # Store QA label
        phrases_df.loc[phrases_df["cluster_id"] == cid, "cluster_label_qa"] = label_qa

        # Flag misfit phrases
        misfit_phrases_set = set(misfit_phrases)
        phrases_df.loc[
            (phrases_df["cluster_id"] == cid) &
            (phrases_df["phrase"].isin(misfit_phrases_set)),
            "qa_misfit_flag"
        ] = True

        summary_rows.append({
            "Cluster ID": cid,
            "QA Label": label_qa,
            "Preview Label": group["cluster_label_preview"].iloc[0],
            "Distinct phrases": len(group),
            "Total occurrences": int(group["count"].sum()),
            "Misfit phrases count": len(misfit_phrases_set),
            "Notes": notes,
            "Subthemes": " | ".join(subthemes) if subthemes else None,
        })

    cluster_qa_summary_df = pd.DataFrame(summary_rows).sort_values(
        "Total occurrences", ascending=False
    ) if summary_rows else pd.DataFrame()

    return phrases_df, cluster_qa_summary_df


# -----------------------------
# LLM QA: keyword-level
# -----------------------------

def build_taxonomy_snapshot(
    phrases_df: pd.DataFrame,
) -> str:
    """
    Build a compact taxonomy text summary for the LLM:
      - For each cluster: label + top phrases.
    """
    if phrases_df.empty:
        return "No taxonomy available."

    if "cluster_label_qa" in phrases_df.columns and phrases_df["cluster_label_qa"].notna().any():
        label_col = "cluster_label_qa"
    elif "cluster_label_llm" in phrases_df.columns and phrases_df["cluster_label_llm"].notna().any():
        label_col = "cluster_label_llm"
    else:
        label_col = "cluster_label_preview"

    lines = []
    for cid, group in phrases_df.groupby("cluster_id"):
        label = group[label_col].iloc[0]
        top_phrases = group.sort_values("count", ascending=False)["phrase"].head(6).tolist()
        lines.append(
            f"Cluster {cid}: {label}\n"
            f"  Phrases: {', '.join(top_phrases)}"
        )
    return "\n".join(lines)


def run_keyword_qa(
    df_keywords: pd.DataFrame,
    keyword_col: str,
    topic_phrases_df: pd.DataFrame,
    provider: str,
    chat_model: str,
    api_key: str,
    max_keywords_for_qa: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For up to max_keywords_for_qa keywords (focusing on those without Topic_Clusters),
    ask the LLM to suggest topics from the discovered taxonomy.

    Returns:
      - updated df_keywords with LLM_Suggested_Topics and Final_Topic_Clusters
      - keyword_qa_df (only rows that were sent to LLM)
    """
    df_keywords = df_keywords.copy()

    # Build taxonomy snapshot text
    taxonomy_text = build_taxonomy_snapshot(topic_phrases_df)

    # Select candidate keywords for QA: those with no Topic_Clusters
    candidates = df_keywords[df_keywords["Topic_Clusters"].isna()].copy()
    candidates = candidates.head(max_keywords_for_qa)

    if candidates.empty:
        df_keywords["LLM_Suggested_Topics"] = None
        df_keywords["Final_Topic_Clusters"] = df_keywords["Topic_Clusters"]
        return df_keywords, pd.DataFrame()

    qa_rows = []

    for idx, row in candidates.iterrows():
        kw = str(row[keyword_col])
        current_topics = row.get("Topic_Clusters") or ""

        system_prompt = (
            "You are an expert SEO topical classifier.\n"
            "You will be given a taxonomy of topic clusters and one search keyword.\n"
            "Your job is to decide which existing topics apply to that keyword.\n"
        )
        user_prompt = (
            "Here is the current taxonomy (clusters and phrases):\n"
            f"{taxonomy_text}\n\n"
            f"Keyword: {kw}\n"
            f"Currently assigned topics: {current_topics or '(none)'}\n\n"
            "From the existing cluster labels, choose all that clearly apply to this keyword.\n"
            "Return ONLY JSON with key 'topics', where 'topics' is a list of cluster labels.\n"
            "Example:\n"
            "{ \"topics\": [\"Park Type\", \"Holiday / Season\"] }"
        )

        try:
            raw = llm_chat(
                provider=provider,
                model=chat_model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=128,
                temperature=LLMConfig.DEFAULT_TEMPERATURE,
            )
            parsed = parse_llm_json_response(raw, required_keys=["topics"])
            topics = parsed.get("topics", [])
            if not isinstance(topics, list):
                topics = []
        except (ValueError, json.JSONDecodeError) as e:
            st.warning(f"Keyword QA failed for keyword '{kw}': {str(e)[:100]}")
            topics = []

        topics_str = " | ".join(sorted(set(topics))) if topics else None

        df_keywords.loc[idx, "LLM_Suggested_Topics"] = topics_str

        qa_rows.append({
            "Keyword": kw,
            "Current_Topics": current_topics or None,
            "LLM_Suggested_Topics": topics_str,
        })

    # Build Final_Topic_Clusters: original, but fill empty with LLM suggestions
    df_keywords["Final_Topic_Clusters"] = df_keywords["Topic_Clusters"]
    mask_fill = df_keywords["Final_Topic_Clusters"].isna() & df_keywords["LLM_Suggested_Topics"].notna()
    df_keywords.loc[mask_fill, "Final_Topic_Clusters"] = df_keywords.loc[mask_fill, "LLM_Suggested_Topics"]

    keyword_qa_df = pd.DataFrame(qa_rows)

    return df_keywords, keyword_qa_df


# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(
        page_title="Topical Keyword Analysis (Semantic + LLM QA)",
        layout="wide",
    )

    st.title("Topical Keyword Analysis with Semantic Clustering & LLM QA")

    st.markdown(
        """
        **Pipeline overview**

        - **Pass 1 – Topical discovery**
          - Extract frequent 1–3 word phrases (grammar stopwords removed).
          - Cluster them with TF-IDF or **semantic embeddings** (OpenAI or Gemini).
          - Optionally use an LLM to **name clusters**.

        - **Pass 2 – Intent / modifiers**
          - Use full text (stopwords kept) to mine 2–3 word intent phrases
            (*near me, for kids, things to do*).
          - Tag each keyword with those phrases.

        - **Pass 3 – LLM QA & refinement (optional)**
          - **Cluster QA:** LLM audits selected clusters, suggests better labels & flags misfit phrases.
          - **Keyword QA:** LLM reviews untagged keywords and suggests topics from the existing taxonomy.
        """
    )

    # 1. Keyword data
    st.sidebar.header("1. Keyword data")
    file = st.sidebar.file_uploader(
        "Upload keyword CSV",
        type=["csv"],
    )

    keyword_col_name = st.sidebar.text_input(
        "Keyword column name",
        value="Keyword",
        max_chars=255,
    )

    # File size warning
    st.sidebar.info(
        "⚠️ **Processing time note:** Files with >50,000 rows or >10MB may take "
        "significantly longer to process, especially with semantic clustering and LLM QA enabled."
    )

    # 2. Topical phrase extraction
    st.sidebar.header("2. Topical phrase extraction (Pass 1)")
    include_unigrams = st.sidebar.checkbox("Include unigrams", value=True)
    include_bigrams = st.sidebar.checkbox("Include bigrams", value=True)
    include_trigrams = st.sidebar.checkbox("Include trigrams", value=True)

    min_phrase_freq = st.sidebar.number_input(
        "Min frequency for topical phrases",
        min_value=1,
        max_value=1000,
        value=3,
        step=1,
    )

    n_clusters = st.sidebar.slider(
        "Number of topic clusters",
        min_value=2,
        max_value=40,
        value=12,
    )

    max_phrases_for_clustering = st.sidebar.number_input(
        "Max phrases to cluster (top by frequency)",
        min_value=50,
        max_value=3000,
        value=400,
        step=50,
    )

    # 3. Semantic & LLM options
    st.sidebar.header("3. Semantic & LLM options")
    use_semantic_clustering = st.sidebar.checkbox(
        "Use semantic embeddings for clustering",
        value=True,
    )

    llm_provider = st.sidebar.selectbox(
        "LLM / Embedding provider",
        options=["None", "OpenAI", "Gemini"],
        index=1,
    )

    emb_model_default = {
        "OpenAI": "text-embedding-3-small",
        "Gemini": "text-embedding-004",
    }.get(llm_provider, "")

    chat_model_default = {
        "OpenAI": "gpt-4.1-mini",
        "Gemini": "gemini-1.5-flash",
    }.get(llm_provider, "")

    emb_model = st.sidebar.text_input(
        "Embedding model",
        value=emb_model_default,
    )

    chat_model = st.sidebar.text_input(
        "Chat model",
        value=chat_model_default,
    )

    enable_llm_cluster_labels = st.sidebar.checkbox(
        "Use LLM to name clusters (Pass 1)",
        value=True,
    )

    api_key: Optional[str] = None
    if llm_provider == "OpenAI":
        api_key = get_api_key_from_secrets_or_input(
            provider="OpenAI",
            default_key_name="openai_api_key",
            sidebar_label="OpenAI API key",
        )
    elif llm_provider == "Gemini":
        api_key = get_api_key_from_secrets_or_input(
            provider="Gemini",
            default_key_name="gemini_api_key",
            sidebar_label="Gemini API key",
        )

    if llm_provider == "None":
        use_semantic_clustering = False
        enable_llm_cluster_labels = False

    # 4. Intent analysis
    st.sidebar.header("4. Intent / modifier analysis (Pass 2)")
    run_intent_pass = st.sidebar.checkbox(
        "Run intent/modifier phrase mining",
        value=True,
    )

    min_intent_freq = st.sidebar.number_input(
        "Min frequency for intent phrases",
        min_value=1,
        max_value=1000,
        value=3,
        step=1,
    )

    top_intent_phrases_for_tagging = st.sidebar.number_input(
        "Top intent phrases to use for tagging",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

    # 5. LLM QA & refinement
    st.sidebar.header("5. LLM QA & refinement (Pass 3)")
    run_cluster_qa_flag = st.sidebar.checkbox(
        "Run LLM QA on top clusters",
        value=False,
    )
    max_clusters_for_qa = st.sidebar.number_input(
        "Max clusters for QA",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    run_keyword_qa_flag = st.sidebar.checkbox(
        "Run LLM QA on untagged keywords",
        value=False,
    )
    max_keywords_for_qa = st.sidebar.number_input(
        "Max untagged keywords for QA",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
    )

    # -----------------------------
    # Main flow
    if file is None:
        st.info("Upload a keyword CSV in the sidebar to begin.")
        return

    # Validate keyword column name
    if not keyword_col_name or not isinstance(keyword_col_name, str):
        st.error("Keyword column name is invalid")
        return
    
    if len(keyword_col_name) > 255:
        st.error("Keyword column name is too long (max 255 characters)")
        return
    
    if keyword_col_name.startswith("__"):
        st.error("Keyword column name cannot start with '__'")
        return

    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read CSV: {type(e).__name__}: {str(e)[:100]}")
        return

    if df.empty:
        st.error("CSV file is empty")
        return

    if keyword_col_name not in df.columns:
        st.error(
            f"Keyword column '{keyword_col_name}' not found in CSV.\n\n"
            f"Available columns: {', '.join(df.columns.tolist())}"
        )
        return

    # Warn about large files
    file_rows = len(df)
    if file_rows > 50000:
        st.warning(
            f"⚠️ Large file detected ({file_rows:,} rows). Processing may take several minutes. "
            "Consider disabling semantic clustering or LLM QA for faster results."
        )

    df = df.copy()
    df[keyword_col_name] = df[keyword_col_name].astype(str)

    st.subheader("Keyword data preview")
    st.dataframe(df.head(25))

    st.markdown("---")
    st.subheader("Run multi-pass analysis")

    if st.button("Run topical + intent + QA"):
        with st.spinner("Running topical discovery and intent mining..."):
            keywords_list = df[keyword_col_name].fillna("").astype(str).tolist()

            # Pass 1: topical phrases
            topical_phrases_df = extract_phrases_from_keywords(
                keywords=keywords_list,
                include_unigrams=include_unigrams,
                include_bigrams=include_bigrams,
                include_trigrams=include_trigrams,
                min_freq=min_phrase_freq,
            )

            if topical_phrases_df.empty:
                st.warning("No topical phrases found with the current settings.")
            else:
                try:
                    if use_semantic_clustering and llm_provider in ("OpenAI", "Gemini") and api_key and emb_model:
                        topical_phrases_df = build_topic_clusters_semantic(
                            phrases_df=topical_phrases_df,
                            n_clusters=n_clusters,
                            max_phrases=max_phrases_for_clustering,
                            emb_provider=llm_provider,
                            emb_model=emb_model,
                            emb_api_key=api_key,
                        )
                    else:
                        topical_phrases_df = build_topic_clusters_tfidf(
                            phrases_df=topical_phrases_df,
                            n_clusters=n_clusters,
                            max_phrases=max_phrases_for_clustering,
                        )
                except Exception as e:
                    st.error(f"Error during topic clustering: {e}")
                    topical_phrases_df = build_topic_clusters_tfidf(
                        phrases_df=topical_phrases_df,
                        n_clusters=n_clusters,
                        max_phrases=max_phrases_for_clustering,
                    )

            # Optional LLM cluster naming
            if (
                enable_llm_cluster_labels
                and not topical_phrases_df.empty
                and llm_provider in ("OpenAI", "Gemini")
                and api_key
                and chat_model
            ):
                try:
                    with st.spinner("Calling LLM to generate cluster labels..."):
                        topical_phrases_df = generate_llm_cluster_labels(
                            phrases_df=topical_phrases_df,
                            df_keywords=df,
                            keyword_col=keyword_col_name,
                            provider=llm_provider,
                            chat_model=chat_model,
                            api_key=api_key,
                        )
                except Exception as e:
                    st.warning(f"LLM cluster labeling failed. Using preview labels only. Error: {e}")
                    topical_phrases_df["cluster_label_llm"] = None

            # Tag keywords with topics (pre-QA)
            df_enriched = tag_keywords_with_topics(
                df_keywords=df,
                keyword_col=keyword_col_name,
                topic_phrases_df=topical_phrases_df,
                use_llm_labels=True,
            )

            # Pass 2: intent phrases
            if run_intent_pass:
                intent_df = extract_intent_phrases(
                    keywords=keywords_list,
                    min_freq=min_intent_freq,
                )
                df_enriched = tag_keywords_with_intent(
                    df_keywords=df_enriched,
                    keyword_col=keyword_col_name,
                    intent_df=intent_df,
                    top_n_phrases=top_intent_phrases_for_tagging,
                )
            else:
                intent_df = pd.DataFrame(columns=["intent_phrase", "n", "count"])
                df_enriched["Intent_Phrases"] = None

            # Pass 3: LLM QA
            cluster_qa_summary_df = pd.DataFrame()
            keyword_qa_df = pd.DataFrame()

            if (
                run_cluster_qa_flag
                and not topical_phrases_df.empty
                and llm_provider in ("OpenAI", "Gemini")
                and api_key
                and chat_model
            ):
                with st.spinner("Running LLM QA on clusters..."):
                    topical_phrases_df, cluster_qa_summary_df = run_cluster_qa(
                        phrases_df=topical_phrases_df,
                        df_keywords=df_enriched,
                        keyword_col=keyword_col_name,
                        provider=llm_provider,
                        chat_model=chat_model,
                        api_key=api_key,
                        max_clusters_for_qa=max_clusters_for_qa,
                    )
                    # Re-tag keywords using QA-updated labels
                    df_enriched = tag_keywords_with_topics(
                        df_keywords=df_enriched,
                        keyword_col=keyword_col_name,
                        topic_phrases_df=topical_phrases_df,
                        use_llm_labels=True,
                    )

            if (
                run_keyword_qa_flag
                and not topical_phrases_df.empty
                and llm_provider in ("OpenAI", "Gemini")
                and api_key
                and chat_model
            ):
                with st.spinner("Running LLM QA on untagged keywords..."):
                    df_enriched, keyword_qa_df = run_keyword_qa(
                        df_keywords=df_enriched,
                        keyword_col=keyword_col_name,
                        topic_phrases_df=topical_phrases_df,
                        provider=llm_provider,
                        chat_model=chat_model,
                        api_key=api_key,
                        max_keywords_for_qa=max_keywords_for_qa,
                    )
            else:
                df_enriched["LLM_Suggested_Topics"] = df_enriched.get("LLM_Suggested_Topics", None)
                df_enriched["Final_Topic_Clusters"] = df_enriched.get("Topic_Clusters")

        st.success("Analysis complete.")

        # Tabs for output
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Topical phrases & clusters",
                "Intent phrases",
                "Enriched keyword table",
                "Cluster QA (LLM)",
                "Keyword QA (LLM)",
            ]
        )

        with tab1:
            st.markdown("### Topical phrases and cluster assignments")
            if topical_phrases_df.empty:
                st.write("No topical phrases found.")
            else:
                st.dataframe(topical_phrases_df.head(500))

                st.markdown("#### Topic cluster summary")
                label_col = "cluster_label_qa" if "cluster_label_qa" in topical_phrases_df.columns and topical_phrases_df["cluster_label_qa"].notna().any() \
                    else ("cluster_label_llm" if "cluster_label_llm" in topical_phrases_df.columns and topical_phrases_df["cluster_label_llm"].notna().any()
                          else "cluster_label_preview")

                summary_rows = []
                for cid, group in topical_phrases_df.groupby("cluster_id"):
                    total_phrases = len(group)
                    total_count = group["count"].sum()
                    label_preview = group[label_col].iloc[0]
                    misfit_count = int(group["qa_misfit_flag"].sum()) if "qa_misfit_flag" in group.columns else 0
                    summary_rows.append(
                        {
                            "Cluster ID": cid,
                            "Label": label_preview,
                            "Distinct phrases": total_phrases,
                            "Total occurrences": int(total_count),
                            "Misfit phrases (QA)": misfit_count,
                        }
                    )
                summary_df = pd.DataFrame(summary_rows).sort_values(
                    "Total occurrences", ascending=False
                )
                st.dataframe(summary_df)

        with tab2:
            st.markdown("### Intent / modifier phrases (Pass 2)")
            if not run_intent_pass:
                st.write("Intent analysis disabled.")
            else:
                if intent_df.empty:
                    st.write("No intent phrases found with the current frequency threshold.")
                else:
                    st.dataframe(intent_df.head(500))

        with tab3:
            st.markdown("### Enriched keyword table")
            st.dataframe(df_enriched.head(500))

            csv_buf = io.StringIO()
            df_enriched.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download enriched keyword CSV",
                data=csv_buf.getvalue(),
                file_name="keywords_topical_enriched.csv",
                mime="text/csv",
            )

        with tab4:
            st.markdown("### Cluster QA (LLM)")
            if cluster_qa_summary_df.empty:
                st.write("Cluster QA not run or no results.")
            else:
                st.dataframe(cluster_qa_summary_df)

        with tab5:
            st.markdown("### Keyword QA (LLM)")
            if keyword_qa_df.empty:
                st.write("Keyword QA not run or no results (e.g., no untagged keywords).")
            else:
                st.dataframe(keyword_qa_df.head(500))


if __name__ == "__main__":
    main()
