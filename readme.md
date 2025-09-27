# Multi-Agent Medical Data Assistant

This project is a multi-agent system for querying and analyzing medical datasets (heart disease, cancer, diabetes) and performing medical web searches using LLMs (Google Gemini, GPT-4o-mini). It provides natural language interfaces for data exploration and web-based medical research.

## Features

- **Natural Language to SQL**: Ask questions about heart disease, cancer, or diabetes datasets in plain English.
- **Medical Web Search**: Summarizes evidence-based information from the web.
- **Multi-Agent Routing**: Automatically routes questions to the appropriate tool (database or web search).
- **LLM Summarization**: Uses Google Gemini or GPT-4o-mini for summarizing results and routing.
- **Extensible**: Easily add new datasets or tools.

## Project Structure

```
.
├── agent_main.py                # Main entrypoint for the multi-agent system
├── convert_csvs_to_sqlite.py    # Converts CSVs to SQLite databases
├── query_heart_count.py         # Example: query heart disease database
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys, etc.)
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── The_Cancer_data_1500_V2.csv
├── dbs/
│   ├── cancer.db
│   ├── diabetes.db
│   └── heart_disease.db
├── output/
│   ├── cancer.db
│   ├── diabetes.db
│   └── heart_disease.db
└── tools/
    ├── cancer_tool.py
    ├── db_tool_base.py
    ├── diabetes_tool.py
    ├── gemini_llm.py
    ├── heart_disease_tool.py
    ├── medical_web_search.py
```

## Setup

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd multi_agent
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory with your API keys:

```
GOOGLE_API_KEY=your_google_gemini_api_key
SERPAPI_API_KEY=your_serpapi_key   # Optional, for web search
BING_SEARCH_API_KEY=your_bing_key  # Optional, for web search fallback
```

### 4. Prepare Databases

Convert the CSV files to SQLite databases:

```sh
python convert_csvs_to_sqlite.py --out_dir=output
```

This will create `cancer.db`, `diabetes.db`, and `heart_disease.db` in the `output/` directory.

## Usage

### Main Agent

Run the main agent:

```sh
python agent_main.py
```

You will be prompted to enter a question. Example questions:

- "How many patients in the cancer dataset have a BMI over 30?"
- "What is the average age of heart disease patients?"
- "What are the symptoms of diabetes?"
- "Show the distribution of cancer diagnoses by gender."

The agent will automatically route your question to the appropriate tool (database or web search).

### Query Heart Disease Example

You can run a direct query example:

```sh
python query_heart_count.py
```

This script demonstrates querying the heart disease database for specific counts.

## Adding New Tools or Datasets

- To add a new dataset, place the CSV in `data/`, update `convert_csvs_to_sqlite.py`, and create a new tool in `tools/`.
- To add a new web search provider, extend `tools/medical_web_search.py`.

## Code Overview

- [`agent_main.py`](agent_main.py): Main orchestration, routing, and LLM-based classification.
- [`tools/db_tool_base.py`](tools/db_tool_base.py): Base class for database-backed tools, handles NL-to-SQL, validation, execution, and summarization.
- [`tools/heart_disease_tool.py`](tools/heart_disease_tool.py), [`tools/cancer_tool.py`](tools/cancer_tool.py), [`tools/diabetes_tool.py`](tools/diabetes_tool.py): Disease-specific database tools.
- [`tools/medical_web_search.py`](tools/medical_web_search.py): Web search and summarization tool.
- [`tools/gemini_llm.py`](tools/gemini_llm.py): Google Gemini LLM wrapper.
- [`convert_csvs_to_sqlite.py`](convert_csvs_to_sqlite.py): Converts CSVs to SQLite databases.

## Requirements

- Python 3.9+
- Google Gemini API key (for LLM features)
- (Optional) SerpAPI or Bing API key for web search

## Security & Privacy

- This tool is for research/educational use only.
- Do not use with sensitive or personally identifiable medical data.
- Always consult a healthcare professional for medical advice.

## License

[MIT License](LICENSE) (add your license file if needed)

---

**Contact:**  
For questions or contributions, open an issue or pull request on GitHub.

**Author**

**Tariful Islam Tarif**
