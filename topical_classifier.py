import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai
import google.generativeai as genai

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_text(text, remove_stopwords=True):
    """Clean and tokenize text."""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return ' '.join(tokens)
    
    return text

def extract_ngrams(text, n=1):
    """Extract n-grams from text."""
    tokens = text.split()
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def safe_json_parse(text):
    """
    Safely parse JSON from LLM response, handling markdown code blocks and extra text.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in the text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# ============================================================================
# TOPICAL PHRASE EXTRACTION
# ============================================================================

def extract_topical_phrases(df, keyword_col='Keyword', min_freq=3, max_ngram=3):
    """
    Extract frequent phrases from keywords (stopwords removed).
    Returns DataFrame with phrases and their frequencies.
    """
    st.info("🔍 Extracting topical phrases...")
    
    # Clean keywords (remove stopwords for topic discovery)
    cleaned_keywords = df[keyword_col].apply(lambda x: clean_text(x, remove_stopwords=True))
    
    # Extract all n-grams
    phrase_counter = Counter()
    for text in cleaned_keywords:
        if not text:
            continue
        for n in range(1, max_ngram + 1):
            ngrams = extract_ngrams(text, n)
            phrase_counter.update(ngrams)
    
    # Filter by frequency
    phrases = [(phrase, count) for phrase, count in phrase_counter.items() 
               if count >= min_freq and len(phrase.split()) <= max_ngram]
    
    phrases_df = pd.DataFrame(phrases, columns=['Phrase', 'Frequency'])
    phrases_df = phrases_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    st.success(f"✅ Found {len(phrases_df)} frequent phrases")
    return phrases_df

# ============================================================================
# CLUSTERING
# ============================================================================

def cluster_phrases_tfidf(phrases_df, n_clusters=20):
    """Cluster phrases using TF-IDF + KMeans."""
    st.info(f"🎯 Clustering {len(phrases_df)} phrases into {n_clusters} topics (TF-IDF)...")
    
    if len(phrases_df) < n_clusters:
        n_clusters = max(2, len(phrases_df) // 2)
        st.warning(f"⚠️ Too few phrases. Reducing clusters to {n_clusters}")
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(phrases_df['Phrase'])
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    phrases_df['Cluster'] = kmeans.fit_predict(X)
    
    # Generate preview labels (top 3 phrases per cluster)
    cluster_labels = {}
    for cluster_id in range(n_clusters):
        cluster_phrases = phrases_df[phrases_df['Cluster'] == cluster_id].nlargest(3, 'Frequency')
        label = ', '.join(cluster_phrases['Phrase'].tolist())
        cluster_labels[cluster_id] = label
    
    phrases_df['Cluster_Preview'] = phrases_df['Cluster'].map(cluster_labels)
    
    st.success("✅ Clustering complete")
    return phrases_df

def get_embedding_openai(text, client):
    """Get OpenAI embedding for text."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"OpenAI embedding error: {e}")
        return None

def get_embedding_gemini(text, model_name="models/embedding-001"):
    """Get Gemini embedding for text."""
    try:
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="clustering"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Gemini embedding error: {e}")
        return None

def cluster_phrases_semantic(phrases_df, n_clusters=20, provider='openai', api_key=None):
    """Cluster phrases using semantic embeddings."""
    st.info(f"🎯 Clustering {len(phrases_df)} phrases into {n_clusters} topics ({provider} embeddings)...")
    
    if len(phrases_df) < n_clusters:
        n_clusters = max(2, len(phrases_df) // 2)
        st.warning(f"⚠️ Too few phrases. Reducing clusters to {n_clusters}")
    
    # Get embeddings for PHRASES (not keywords!)
    embeddings = []
    phrases_to_embed = phrases_df['Phrase'].tolist()
    
    progress_bar = st.progress(0)
    
    if provider == 'openai':
        client = openai.OpenAI(api_key=api_key)
        # Batch process for efficiency
        batch_size = 100
        for i in range(0, len(phrases_to_embed), batch_size):
            batch = phrases_to_embed[i:i+batch_size]
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                st.error(f"Error getting embeddings for batch {i}: {e}")
                return None
            
            progress_bar.progress(min((i + batch_size) / len(phrases_to_embed), 1.0))
    
    elif provider == 'gemini':
        genai.configure(api_key=api_key)
        for i, phrase in enumerate(phrases_to_embed):
            emb = get_embedding_gemini(phrase)
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append([0] * 768)  # Fallback
            
            if (i + 1) % 10 == 0:
                progress_bar.progress((i + 1) / len(phrases_to_embed))
    
    progress_bar.empty()
    
    if not embeddings or len(embeddings) != len(phrases_df):
        st.error("❌ Failed to get embeddings for all phrases")
        return None
    
    # Cluster the phrase embeddings
    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    phrases_df['Cluster'] = kmeans.fit_predict(X)
    
    # Generate preview labels
    cluster_labels = {}
    for cluster_id in range(n_clusters):
        cluster_phrases = phrases_df[phrases_df['Cluster'] == cluster_id].nlargest(3, 'Frequency')
        label = ', '.join(cluster_phrases['Phrase'].tolist())
        cluster_labels[cluster_id] = label
    
    phrases_df['Cluster_Preview'] = phrases_df['Cluster'].map(cluster_labels)
    
    st.success("✅ Semantic clustering complete")
    return phrases_df

# ============================================================================
# LLM CLUSTER LABELING
# ============================================================================

def label_clusters_with_llm(phrases_df, provider='openai', api_key=None, model=None):
    """Generate human-readable labels for clusters using LLM."""
    st.info(f"🏷️ Generating cluster labels with {provider}...")
    
    cluster_ids = phrases_df['Cluster'].unique()
    llm_labels = {}
    
    progress_bar = st.progress(0)
    
    for i, cluster_id in enumerate(cluster_ids):
        cluster_phrases = phrases_df[phrases_df['Cluster'] == cluster_id].nlargest(10, 'Frequency')
        phrases_list = cluster_phrases['Phrase'].tolist()
        
        prompt = f"""Given these related keyword phrases from a search keyword dataset:

{', '.join(phrases_list)}

Provide a short (1-3 word) conceptual label that captures the main theme. 

Respond with ONLY a JSON object in this exact format (no other text):
{{"label": "Your Label Here"}}"""
        
        try:
            if provider == 'openai':
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                response_text = response.choices[0].message.content
            
            elif provider == 'gemini':
                genai.configure(api_key=api_key)
                model_obj = genai.GenerativeModel(model or 'gemini-1.5-flash')
                response = model_obj.generate_content(prompt)
                response_text = response.text
            
            # Parse JSON safely
            result = safe_json_parse(response_text)
            if result and 'label' in result:
                llm_labels[cluster_id] = result['label']
            else:
                # Fallback: use preview label
                llm_labels[cluster_id] = phrases_df[phrases_df['Cluster'] == cluster_id]['Cluster_Preview'].iloc[0][:30]
                st.warning(f"⚠️ Could not parse label for cluster {cluster_id}, using preview")
        
        except Exception as e:
            st.error(f"❌ Error labeling cluster {cluster_id}: {e}")
            llm_labels[cluster_id] = phrases_df[phrases_df['Cluster'] == cluster_id]['Cluster_Preview'].iloc[0][:30]
        
        progress_bar.progress((i + 1) / len(cluster_ids))
    
    progress_bar.empty()
    
    phrases_df['LLM_Label'] = phrases_df['Cluster'].map(llm_labels)
    st.success(f"✅ Labeled {len(llm_labels)} clusters")
    
    return phrases_df

# ============================================================================
# INTENT PHRASE EXTRACTION
# ============================================================================

def extract_intent_phrases(df, keyword_col='Keyword', min_freq=3):
    """
    Extract intent/modifier phrases (WITH stopwords) that contain intent signals.
    """
    st.info("🎯 Extracting intent phrases...")
    
    # Intent-bearing words
    intent_words = {
        'near', 'me', 'for', 'with', 'how', 'what', 'where', 'when', 'why',
        'best', 'cheap', 'free', 'guide', 'tips', 'reviews', 'map',
        'hours', 'tickets', 'deals', 'coupons', 'kids', 'adults',
        'things', 'to', 'do', 'visit', 'open', 'closed', 'directions'
    }
    
    # Keep stopwords for intent extraction
    cleaned_keywords = df[keyword_col].apply(lambda x: clean_text(x, remove_stopwords=False))
    
    # Extract 2-3 word phrases containing intent words
    intent_counter = Counter()
    
    for text in cleaned_keywords:
        if not text:
            continue
        
        # Extract bigrams and trigrams
        for n in [2, 3]:
            ngrams = extract_ngrams(text, n)
            for ngram in ngrams:
                # Check if any intent word is in the ngram
                ngram_words = set(ngram.split())
                if ngram_words & intent_words:  # Intersection
                    intent_counter[ngram] += 1
    
    # Filter by frequency
    intent_phrases = [(phrase, count) for phrase, count in intent_counter.items() 
                      if count >= min_freq]
    
    intent_df = pd.DataFrame(intent_phrases, columns=['Intent_Phrase', 'Frequency'])
    intent_df = intent_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    st.success(f"✅ Found {len(intent_df)} intent phrases")
    return intent_df

# ============================================================================
# KEYWORD TAGGING
# ============================================================================

def tag_keywords_with_topics(df, phrases_df, keyword_col='Keyword'):
    """Tag each keyword with matching topical phrases and clusters."""
    st.info("🏷️ Tagging keywords with topics...")
    
    # Create phrase lookup
    phrase_to_cluster = dict(zip(phrases_df['Phrase'], phrases_df['Cluster']))
    phrase_to_label = dict(zip(phrases_df['Phrase'], phrases_df.get('LLM_Label', phrases_df['Cluster_Preview'])))
    
    topic_phrases_list = []
    topic_clusters_list = []
    
    for keyword in df[keyword_col]:
        cleaned = clean_text(keyword, remove_stopwords=True)
        
        found_phrases = []
        found_clusters = set()
        
        # Check which phrases appear in this keyword
        for phrase in phrase_to_cluster.keys():
            if phrase in cleaned:
                found_phrases.append(phrase)
                found_clusters.add(phrase_to_label.get(phrase, f"Cluster_{phrase_to_cluster[phrase]}"))
        
        topic_phrases_list.append(', '.join(found_phrases) if found_phrases else '')
        topic_clusters_list.append(', '.join(sorted(found_clusters)) if found_clusters else '')
    
    df['Topic_Phrases'] = topic_phrases_list
    df['Topic_Clusters'] = topic_clusters_list
    
    st.success("✅ Keywords tagged with topics")
    return df

def tag_keywords_with_intent(df, intent_df, keyword_col='Keyword'):
    """Tag each keyword with matching intent phrases."""
    st.info("🏷️ Tagging keywords with intent...")
    
    intent_phrases_set = set(intent_df['Intent_Phrase'])
    intent_list = []
    
    for keyword in df[keyword_col]:
        cleaned = clean_text(keyword, remove_stopwords=False)
        
        found_intents = []
        for intent_phrase in intent_phrases_set:
            if intent_phrase in cleaned:
                found_intents.append(intent_phrase)
        
        intent_list.append(', '.join(found_intents) if found_intents else '')
    
    df['Intent_Phrases'] = intent_list
    
    st.success("✅ Keywords tagged with intent")
    return df

# ============================================================================
# CLUSTER QA
# ============================================================================

def perform_cluster_qa(phrases_df, top_n=20, provider='openai', api_key=None, model=None):
    """
    LLM reviews top clusters and provides refined labels and identifies misfits.
    """
    st.info(f"🔍 Performing QA on top {top_n} clusters...")
    
    # Get top clusters by total phrase frequency
    cluster_volumes = phrases_df.groupby('Cluster')['Frequency'].sum().sort_values(ascending=False)
    top_clusters = cluster_volumes.head(top_n).index.tolist()
    
    qa_results = []
    progress_bar = st.progress(0)
    
    for i, cluster_id in enumerate(top_clusters):
        cluster_data = phrases_df[phrases_df['Cluster'] == cluster_id].sort_values('Frequency', ascending=False)
        
        top_phrases = cluster_data.head(15)['Phrase'].tolist()
        current_label = cluster_data.iloc[0].get('LLM_Label', cluster_data.iloc[0]['Cluster_Preview'])
        
        prompt = f"""Review this keyword cluster labeled "{current_label}":

Top phrases: {', '.join(top_phrases)}

Tasks:
1. Provide a refined label (1-3 words) that best captures the theme
2. Identify any phrases that don't fit (list them or say "none")
3. Note any subthemes you observe

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "cluster_label_qa": "Refined Label",
  "misfit_phrases": ["phrase1", "phrase2"],
  "notes": "Brief observations about subthemes"
}}"""
        
        try:
            if provider == 'openai':
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                response_text = response.choices[0].message.content
            
            elif provider == 'gemini':
                genai.configure(api_key=api_key)
                model_obj = genai.GenerativeModel(model or 'gemini-1.5-flash')
                response = model_obj.generate_content(prompt)
                response_text = response.text
            
            # Parse response safely
            qa_result = safe_json_parse(response_text)
            
            if qa_result:
                qa_results.append({
                    'Cluster': cluster_id,
                    'Original_Label': current_label,
                    'QA_Label': qa_result.get('cluster_label_qa', current_label),
                    'Misfit_Phrases': ', '.join(qa_result.get('misfit_phrases', [])),
                    'Misfit_Count': len(qa_result.get('misfit_phrases', [])),
                    'Notes': qa_result.get('notes', '')
                })
            else:
                st.warning(f"⚠️ Could not parse QA response for cluster {cluster_id}")
                qa_results.append({
                    'Cluster': cluster_id,
                    'Original_Label': current_label,
                    'QA_Label': current_label,
                    'Misfit_Phrases': '',
                    'Misfit_Count': 0,
                    'Notes': 'QA parsing failed'
                })
        
        except Exception as e:
            st.error(f"❌ Cluster QA failed for cluster {cluster_id}: {e}")
            qa_results.append({
                'Cluster': cluster_id,
                'Original_Label': current_label,
                'QA_Label': current_label,
                'Misfit_Phrases': '',
                'Misfit_Count': 0,
                'Notes': f'Error: {str(e)}'
            })
        
        progress_bar.progress((i + 1) / len(top_clusters))
    
    progress_bar.empty()
    
    qa_df = pd.DataFrame(qa_results)
    
    # Update phrases_df with QA labels
    qa_label_map = dict(zip(qa_df['Cluster'], qa_df['QA_Label']))
    phrases_df['QA_Label'] = phrases_df['Cluster'].map(qa_label_map)
    
    # Flag misfit phrases
    misfit_set = set()
    for _, row in qa_df.iterrows():
        if row['Misfit_Phrases']:
            misfits = [p.strip() for p in row['Misfit_Phrases'].split(',')]
            misfit_set.update(misfits)
    
    phrases_df['Is_Misfit'] = phrases_df['Phrase'].isin(misfit_set)
    
    st.success(f"✅ QA complete for {len(qa_df)} clusters")
    return phrases_df, qa_df

# ============================================================================
# KEYWORD QA (for untagged keywords)
# ============================================================================

def perform_keyword_qa(df, phrases_df, keyword_col='Keyword', sample_size=50, 
                       provider='openai', api_key=None, model=None):
    """
    For keywords with no topics, ask LLM to suggest which topics apply.
    """
    st.info(f"🔍 Performing QA on untagged keywords...")
    
    # Find keywords with no topics
    untagged = df[df['Topic_Clusters'] == ''].copy()
    
    if len(untagged) == 0:
        st.info("✅ All keywords have topics assigned")
        return df, pd.DataFrame()
    
    st.info(f"Found {len(untagged)} untagged keywords")
    
    # Sample
    sample_kws = untagged.sample(min(sample_size, len(untagged)), random_state=42)
    
    # Build taxonomy snapshot
    cluster_summary = []
    for cluster_id in phrases_df['Cluster'].unique():
        cluster_data = phrases_df[phrases_df['Cluster'] == cluster_id]
        label = cluster_data.iloc[0].get('QA_Label') or cluster_data.iloc[0].get('LLM_Label') or cluster_data.iloc[0]['Cluster_Preview']
        top_phrases = cluster_data.nlargest(5, 'Frequency')['Phrase'].tolist()
        cluster_summary.append(f"- {label}: {', '.join(top_phrases)}")
    
    taxonomy = '\n'.join(cluster_summary)
    
    qa_results = []
    progress_bar = st.progress(0)
    
    for i, (idx, row) in enumerate(sample_kws.iterrows()):
        keyword = row[keyword_col]
        
        prompt = f"""Given this taxonomy of keyword topics:

{taxonomy}

Which topic label(s) apply to this keyword: "{keyword}"?

Respond with ONLY a JSON object:
{{
  "suggested_topics": ["Topic1", "Topic2"]
}}

If no topics fit, return empty array."""
        
        try:
            if provider == 'openai':
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                response_text = response.choices[0].message.content
            
            elif provider == 'gemini':
                genai.configure(api_key=api_key)
                model_obj = genai.GenerativeModel(model or 'gemini-1.5-flash')
                response = model_obj.generate_content(prompt)
                response_text = response.text
            
            result = safe_json_parse(response_text)
            
            if result and 'suggested_topics' in result:
                suggested = ', '.join(result['suggested_topics'])
            else:
                suggested = ''
            
            qa_results.append({
                'Keyword': keyword,
                'Original_Topics': row['Topic_Clusters'],
                'LLM_Suggested_Topics': suggested
            })
        
        except Exception as e:
            st.error(f"❌ Keyword QA failed for '{keyword}': {e}")
            qa_results.append({
                'Keyword': keyword,
                'Original_Topics': row['Topic_Clusters'],
                'LLM_Suggested_Topics': ''
            })
        
        progress_bar.progress((i + 1) / len(sample_kws))
    
    progress_bar.empty()
    
    kw_qa_df = pd.DataFrame(qa_results)
    
    # Merge suggestions back into main dataframe
    suggestion_map = dict(zip(kw_qa_df['Keyword'], kw_qa_df['LLM_Suggested_Topics']))
    
    df['LLM_Suggested_Topics'] = df[keyword_col].map(suggestion_map).fillna('')
    
    # Create final topics column (use original if available, else LLM suggestion)
    df['Final_Topic_Clusters'] = df.apply(
        lambda x: x['Topic_Clusters'] if x['Topic_Clusters'] else x['LLM_Suggested_Topics'],
        axis=1
    )
    
    st.success(f"✅ QA complete for {len(kw_qa_df)} keywords")
    return df, kw_qa_df

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="Keyword Topical Classifier", layout="wide")
    
    st.title("🎯 Keyword Topical Classifier")
    st.markdown("""
    **3-Pass Classification System:**
    1. **Topical Discovery**: Find themes and cluster related phrases
    2. **Intent/Modifier Discovery**: Extract intent patterns (near me, for kids, etc.)
    3. **LLM QA & Refinement**: Improve labels and fill gaps
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV with keywords", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Column selection
            keyword_col = st.selectbox("Keyword Column", df.columns.tolist())
            
            st.markdown("---")
            st.subheader("Pass 1: Topics")
            min_phrase_freq = st.slider("Min phrase frequency", 2, 20, 3)
            max_ngram = st.slider("Max n-gram size", 1, 5, 3)
            n_clusters = st.slider("Number of clusters", 5, 100, 30)
            
            clustering_method = st.radio("Clustering method", 
                                        ["TF-IDF", "Semantic (OpenAI)", "Semantic (Gemini)"])
            
            use_llm_labels = st.checkbox("Generate LLM labels for clusters", value=True)
            
            st.markdown("---")
            st.subheader("Pass 2: Intent")
            min_intent_freq = st.slider("Min intent phrase frequency", 2, 20, 3)
            
            st.markdown("---")
            st.subheader("Pass 3: QA")
            enable_cluster_qa = st.checkbox("Enable Cluster QA", value=True)
            cluster_qa_top_n = st.slider("QA top N clusters", 5, 50, 20)
            
            enable_keyword_qa = st.checkbox("Enable Keyword QA", value=True)
            keyword_qa_sample = st.slider("QA sample size", 10, 200, 50)
            
            st.markdown("---")
            st.subheader("🔑 API Keys")
            
            provider = st.radio("LLM Provider", ["openai", "gemini"])
            
            if provider == 'openai':
                api_key = st.text_input("OpenAI API Key", type="password")
                model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
            else:
                api_key = st.text_input("Gemini API Key", type="password")
                model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
            
            st.markdown("---")
            run_analysis = st.button("🚀 Run Analysis", type="primary")
        else:
            run_analysis = False
    
    # Main content area
    if uploaded_file and run_analysis:
        if not api_key and (use_llm_labels or enable_cluster_qa or enable_keyword_qa):
            st.error("❌ Please provide an API key for LLM features")
            return
        
        # Store results in session state
        with st.spinner("Running analysis..."):
            
            # PASS 1: Topical Discovery
            st.header("📊 Pass 1: Topical Discovery")
            
            phrases_df = extract_topical_phrases(df, keyword_col, min_phrase_freq, max_ngram)
            
            if clustering_method == "TF-IDF":
                phrases_df = cluster_phrases_tfidf(phrases_df, n_clusters)
            elif clustering_method == "Semantic (OpenAI)":
                phrases_df = cluster_phrases_semantic(phrases_df, n_clusters, 'openai', api_key)
            else:
                phrases_df = cluster_phrases_semantic(phrases_df, n_clusters, 'gemini', api_key)
            
            if phrases_df is None:
                st.error("❌ Clustering failed")
                return
            
            if use_llm_labels and api_key:
                phrases_df = label_clusters_with_llm(phrases_df, provider, api_key, model)
            
            # Tag keywords with topics
            df = tag_keywords_with_topics(df, phrases_df, keyword_col)
            
            # PASS 2: Intent Discovery
            st.header("🎯 Pass 2: Intent Discovery")
            intent_df = extract_intent_phrases(df, keyword_col, min_intent_freq)
            df = tag_keywords_with_intent(df, intent_df, keyword_col)
            
            # PASS 3: LLM QA
            st.header("🔍 Pass 3: LLM QA & Refinement")
            
            qa_df = pd.DataFrame()
            kw_qa_df = pd.DataFrame()
            
            if enable_cluster_qa and api_key:
                phrases_df, qa_df = perform_cluster_qa(phrases_df, cluster_qa_top_n, 
                                                      provider, api_key, model)
                # Re-tag keywords with QA labels
                df = tag_keywords_with_topics(df, phrases_df, keyword_col)
            
            if enable_keyword_qa and api_key:
                df, kw_qa_df = perform_keyword_qa(df, phrases_df, keyword_col, 
                                                  keyword_qa_sample, provider, api_key, model)
        
        st.success("✅ Analysis complete!")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Topics & Clusters", 
            "🎯 Intent Phrases",
            "📋 Enriched Keywords",
            "🔍 Cluster QA",
            "🔍 Keyword QA"
        ])
        
        with tab1:
            st.subheader("Topical Phrases & Clusters")
            st.dataframe(phrases_df, use_container_width=True)
            
            # Download button
            csv = phrases_df.to_csv(index=False)
            st.download_button("⬇️ Download Topics CSV", csv, "topics.csv", "text/csv")
        
        with tab2:
            st.subheader("Intent Phrases")
            st.dataframe(intent_df, use_container_width=True)
            
            csv = intent_df.to_csv(index=False)
            st.download_button("⬇️ Download Intent CSV", csv, "intent.csv", "text/csv")
        
        with tab3:
            st.subheader("Enriched Keywords")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Keywords CSV", csv, "keywords_enriched.csv", "text/csv")
        
        with tab4:
            if not qa_df.empty:
                st.subheader("Cluster QA Results")
                st.dataframe(qa_df, use_container_width=True)
                
                csv = qa_df.to_csv(index=False)
                st.download_button("⬇️ Download Cluster QA CSV", csv, "cluster_qa.csv", "text/csv")
            else:
                st.info("Cluster QA was not enabled or produced no results")
        
        with tab5:
            if not kw_qa_df.empty:
                st.subheader("Keyword QA Results")
                st.dataframe(kw_qa_df, use_container_width=True)
                
                csv = kw_qa_df.to_csv(index=False)
                st.download_button("⬇️ Download Keyword QA CSV", csv, "keyword_qa.csv", "text/csv")
            else:
                st.info("Keyword QA was not enabled or produced no results")

if __name__ == "__main__":
    main()
