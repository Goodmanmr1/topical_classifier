# Topical Classifier

A Streamlit-based application for multi-pass analysis of keywords, extracting topical phrases, intent signals, and generating intelligent cluster labels using semantic embeddings and LLM capabilities.

## Features

- **Topical Phrase Extraction**: Extracts meaningful phrases (unigrams, bigrams, trigrams) from keywords with configurable frequency thresholds
- **Semantic Clustering**: Groups phrases using either TF-IDF or semantic embeddings (OpenAI/Gemini)
- **Intent Mining**: Identifies intent-based modifiers and qualifiers in keywords
- **LLM-Powered Labeling**: Generates human-readable cluster labels using OpenAI or Google Gemini APIs
- **QA Validation**: Validates cluster assignments through LLM-based question-answering
- **Multi-pass Analysis**: Comprehensive three-pass analysis (topical → intent → QA)
- **CSV Export**: Download enriched keywords with topic assignments and intent tags

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/topical_classifier.git
cd topical_classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Run

```bash
streamlit run topical_classifier.py
```

The app will open in your browser at `http://localhost:8501`

### Configuration

#### Data Input
- Upload a CSV file with keywords
- Select the column containing keywords to analyze

#### Analysis Parameters

**Phrase Extraction:**
- Include unigrams, bigrams, trigrams
- Set minimum phrase frequency threshold

**Clustering:**
- Choose between TF-IDF and semantic clustering (requires API key)
- Configure number of clusters (3-20)
- Adjust maximum phrases per cluster

**Intent Analysis:**
- Enable/disable intent pass
- Set minimum frequency threshold
- Configure top phrases for tagging

**LLM Features:**
- Enable cluster labeling with LLM
- Run cluster-level QA validation
- Run keyword-level QA validation
- Choose between OpenAI or Google Gemini
- Select appropriate embedding and chat models

#### API Keys

Provide API keys through the sidebar. Keys can be sourced from:
- Environment: `OPENAI_API_KEY` and `GOOGLE_API_KEY` in `.streamlit/secrets.toml`
- UI: Enter directly in the sidebar (takes precedence over environment)

**Streamlit Secrets Setup:**

Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "..."
```

## Model Recommendations

### OpenAI
- **Embeddings**: `text-embedding-3-small` (fast, efficient)
- **Chat**: `gpt-4-turbo` or `gpt-3.5-turbo`

### Google Gemini
- **Embeddings**: `models/embedding-001`
- **Chat**: `gemini-pro`

## Output

### Tabs Overview

1. **Topical phrases & clusters**: View extracted phrases and their cluster assignments with summary statistics
2. **Intent phrases**: See identified intent modifiers and their frequencies
3. **Enriched keyword table**: Original keywords with assigned topics, intent phrases, and QA results (downloadable CSV)
4. **Cluster QA (LLM)**: LLM validation results for cluster assignments
5. **Keyword QA (LLM)**: LLM suggestions for untagged keywords

## Architecture

### Core Components

- **Text Utilities**: Normalization and tokenization with stopword filtering
- **Phrase Extraction**: N-gram generation with frequency-based filtering
- **Clustering**: TF-IDF and semantic embedding-based grouping
- **Intent Detection**: Pattern matching against intent tokens
- **LLM Integration**: JSON-based prompting for labels and QA
- **Tagging Engine**: Assigns clusters and intents to keywords

### Configuration Classes

- `LLMConfig`: Token limits and temperature settings
- `ClusteringConfig`: Clustering parameters and constraints

## Requirements

See `requirements.txt` for full dependency list:

- **streamlit**: Web framework
- **scikit-learn**: TF-IDF vectorization and K-means clustering
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **openai**: OpenAI API client
- **google-generativeai**: Google Gemini API client

## Troubleshooting

### API Key Issues
- Ensure API keys are set in either `.streamlit/secrets.toml` or entered in the sidebar
- Verify you have appropriate permissions for the selected models

### Clustering Failures
- If semantic clustering fails, the app automatically falls back to TF-IDF
- Check API availability and quota limits

### Empty Results
- Increase `min_freq` parameters in phrase extraction settings
- Ensure input keywords contain sufficient variety
- Check that the selected keyword column is correct

## Performance Notes

- Semantic clustering is slower than TF-IDF but provides better topical coherence
- LLM QA is limited to top 5 clusters by default to manage API costs
- Large keyword datasets (10,000+) may require longer processing times

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

---

**Last Updated**: November 2024
