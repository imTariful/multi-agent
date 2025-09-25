# tools/medical_web_search.py
import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class MedicalWebSearchTool:
    def __init__(self, serpapi_key=None, bing_key=None, llm_model_name='gpt-4o-mini'):
        self.serpapi_key = serpapi_key or os.getenv('SERPAPI_API_KEY')
        self.bing_key = bing_key or os.getenv('BING_SEARCH_API_KEY')
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0)

    def _search_serpapi(self, q, num=5):
        if not self.serpapi_key:
            return []
        params = {
            "engine": "google",
            "q": q,
            "api_key": self.serpapi_key,
            "num": num
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
        data = r.json()
        snippets = []
        for item in data.get("organic_results", [])[:num]:
            title = item.get("title")
            snippet = item.get("snippet") or item.get("snippet_highlighted_words") or ""
            link = item.get("link")
            snippets.append({"title": title, "snippet": snippet, "link": link})
        return snippets

    def run(self, query: str, num_results: int = 4):
        # 1) search
        results = self._search_serpapi(query, num=num_results)
        if not results and self.bing_key:
            # Implement simple Bing fallback (omitted for brevity)
            pass
        if not results:
            return "No web results found (no API key configured)."

        # 2) Summarize using LLM
        top_text = "\n\n".join([f"{r['title']}\n{r['snippet']}\n{r['link']}" for r in results])
        system = SystemMessage(content="You are a careful medical researcher. Summarize the most important, evidence-backed points from the search results. Do not give misleading medical advice; recommend seeing a clinician when appropriate.")
        human = HumanMessage(content=f"User asked: {query}\n\nSearch snippets:\n{top_text}\n\nProduce a short (3-6 sentence) clear summary, and list 2 reliable next steps the user can take (e.g., 'see a doctor', 'read WHO page', etc.).")
        resp = self.llm.invoke([system, human])
        return resp.content.strip()
