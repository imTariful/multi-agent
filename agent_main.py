# agent_main.py

import os
import argparse

# Load CLI arg / .env before importing langchain so OPENAI_API_KEY is present
parser = argparse.ArgumentParser(description="Run MultiToolAgent")
parser.add_argument("--api-key", help="OpenAI API key (optional). If provided it will be set for this run.")
parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name to use (default: gpt-4o-mini)")
args, _ = parser.parse_known_args()

# If CLI key provided, set it in env for this process
if args.api_key:
    os.environ["OPENAI_API_KEY"] = args.api_key

# Try to load .env if present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not required; skip if not installed
    pass

# Quick check: ensure an API key exists before importing/initializing LLMs
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("ERROR: OPENAI_API_KEY is not set. Provide it via --api-key, set the OPENAI_API_KEY env var, or add to a .env file.")
    print("Example (PowerShell): $env:OPENAI_API_KEY='sk-...'; & .\\.venv\\Scripts\\python.exe .\\agent_main.py")
    raise SystemExit(1)

# Now safe to import LangChain/OpenAI
from tools.heart_disease_tool import HeartDiseaseDBTool
from tools.cancer_tool import CancerDBTool
from tools.diabetes_tool import DiabetesDBTool
from tools.medical_web_search import MedicalWebSearchTool

# Use LangChain stable imports
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def _extract_text_from_resp(resp):
    """
    Robust extraction for several langchain return shapes.
    """
    # Case: simple string
    if isinstance(resp, str):
        return resp
    # Case: AIMessage or similar with .content
    if hasattr(resp, "content"):
        return getattr(resp, "content")
    # Case: list/sequence of messages
    try:
        if isinstance(resp, (list, tuple)) and len(resp) > 0:
            first = resp[0]
            if hasattr(first, "content"):
                return first.content
            return str(first)
    except Exception:
        pass
    # Case: LLMResult with .generations
    if hasattr(resp, "generations"):
        try:
            gens = resp.generations
            if gens and len(gens) > 0 and len(gens[0]) > 0:
                # common layout: generations[0][0].text
                candidate = gens[0][0]
                if hasattr(candidate, "text"):
                    return candidate.text
                if hasattr(candidate, "content"):
                    return candidate.content
        except Exception:
            pass
    # Fallback
    return str(resp)

class MultiToolAgent:
    def __init__(self, llm_model_name: str = "gpt-4o-mini"):
        # instantiate tools (they may internally create their own ChatOpenAI instances)
        self.heart_tool = HeartDiseaseDBTool(llm_model_name=llm_model_name)
        self.cancer_tool = CancerDBTool(llm_model_name=llm_model_name)
        self.diabetes_tool = DiabetesDBTool(llm_model_name=llm_model_name)
        self.web_tool = MedicalWebSearchTool(llm_model_name=llm_model_name)

        # primary LLM used for routing/classification & short summaries
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0)

    def _heuristic_route(self, question: str):
        q = question.lower()
        web_keywords = [
            "symptom", "symptoms", "definition", "what is", "treatment", "cure",
            "causes", "diagnosis", "how to treat", "how is", "side effect"
        ]
        data_keywords = [
            "count", "mean", "average", "median", "percent", "percentage", "correlation",
            "distribution", "how many", "rate", "incidence", "prevalence", "age",
            "cholesterol", "glucose", "bp", "blood", "sex", "gender"
        ]
        if any(k in q for k in web_keywords):
            return "web"
        if any(k in q for k in data_keywords):
            return "db"
        return "ask_llm"

    def _llm_classify(self, question: str):
        system = SystemMessage(content=(
            "You are an assistant that selects which tool should handle a user's medical question. "
            "Options: [heart_db, cancer_db, diabetes_db, web]. Answer with a single token exactly: "
            "heart_db or cancer_db or diabetes_db or web."
        ))
        human = HumanMessage(content=f"Question: {question}\nWhich tool should we use? Provide only one of: heart_db, cancer_db, diabetes_db, web.")
        resp = self.llm([system, human])
        text = _extract_text_from_resp(resp)
        return text.strip().lower()

    def handle(self, question: str):
        route = self._heuristic_route(question)
        if route == "web":
            return self.web_tool.run(question)
        if route == "db":
            q = question.lower()
            if any(w in q for w in ["heart", "cardio", "cholesterol", "ecg", "angina", "bp", "blood pressure"]):
                return self.heart_tool.run(question)
            if any(w in q for w in ["cancer", "tumor", "tumour", "malign", "benign", "oncology"]):
                return self.cancer_tool.run(question)
            if any(w in q for w in ["diabetes", "glucose", "a1c", "hba1c", "insulin"]):
                return self.diabetes_tool.run(question)
            cls = self._llm_classify(question)
            if cls == "heart_db":
                return self.heart_tool.run(question)
            if cls == "cancer_db":
                return self.cancer_tool.run(question)
            if cls == "diabetes_db":
                return self.diabetes_tool.run(question)
            return "I'm not sure which DB to use; try specifying the disease in the question."

        # ask_llm fallback
        cls = self._llm_classify(question)
        if cls == "web":
            return self.web_tool.run(question)
        if cls == "heart_db":
            return self.heart_tool.run(question)
        if cls == "cancer_db":
            return self.cancer_tool.run(question)
        if cls == "diabetes_db":
            return self.diabetes_tool.run(question)
        return "Couldn't decide; please rephrase or mention disease explicitly."


if __name__ == "__main__":
    agent = MultiToolAgent(llm_model_name=args.model)
    print("Agent ready. Type a question or 'quit' to exit.")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        out = agent.handle(q)
        print("\n--- Agent response ---")
        print(out)
        print("\n----------------------\n")
