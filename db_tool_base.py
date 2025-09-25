# tools/db_tool_base.py
import sqlite3
import re
from typing import List
import textwrap
import os

# Import an LLM client from LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

SAFE_SQL_WHITELIST = re.compile(r'^\s*select\b', re.IGNORECASE)
FORBIDDEN_PATTERNS = re.compile(r'\b(insert|update|delete|drop|alter|create|attach|detach|replace|pragma)\b', re.IGNORECASE)

class DBToolBase:
    def __init__(self, db_path: str, table_name: str, llm_model_name: str = "gpt-4o-mini"):
        self.db_path = db_path
        self.table_name = table_name
        # Use LangChain ChatOpenAI — set OPENAI_API_KEY in env
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0)
        self.schema = self._read_schema()

    def _read_schema(self) -> str:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({self.table_name})")
        cols = cur.fetchall()  # [(cid, name, type, notnull, dflt_value, pk), ...]
        conn.close()
        if not cols:
            return f"Table {self.table_name} not found in DB."
        schema_lines = [f"- {c[1]} ({c[2] or 'TEXT'})" for c in cols]
        return "\n".join(schema_lines)

    def _nl_to_sql(self, question: str, max_rows: int = 200) -> str:
        """
        Ask the LLM to generate a safe SQL SELECT that uses the available columns.
        We instruct the LLM to only produce a single SELECT statement and nothing else.
        """
        system = SystemMessage(content=textwrap.dedent(f"""
            You are a helpful assistant that translates natural language questions into SQLITE SELECT statements.
            The database has the following table and columns:
            Table: {self.table_name}
            Columns:
            {self.schema}

            Constraints:
            - Return ONLY a single valid SQLite SELECT statement; do NOT include any explanation, backticks, or SQL comments.
            - The statement must be read-only (SELECT). Do NOT use INSERT/UPDATE/DELETE/DROP/ALTER/PRAGMA.
            - Limit results to at most {max_rows} rows using LIMIT.
            - Use correct column names and table name exactly as provided.
        """))
        human = HumanMessage(content=f"Question: {question}\n\nReturn only the SQL SELECT statement.")
        resp = self.llm.invoke([system, human])
        sql = resp.content.strip()
        # Clean: sometimes models include trailing semicolons or backticks
        sql = sql.rstrip(';')
        return sql

    def _validate_sql(self, sql: str) -> bool:
        if FORBIDDEN_PATTERNS.search(sql):
            return False
        if not SAFE_SQL_WHITELIST.match(sql):
            return False
        return True

    def _execute_sql(self, sql: str, row_limit: int = 200):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(row_limit)
            conn.close()
            return cols, rows
        except Exception as e:
            conn.close()
            raise

    def run(self, question: str, max_rows: int = 50) -> str:
        # Step 1: NL -> SQL
        sql = self._nl_to_sql(question, max_rows=max_rows)
        # Step 2: Validate
        if not self._validate_sql(sql):
            return "I refused to run that SQL because it looks unsafe. Please ask a read-only question."
        # Step 3: Execute
        try:
            cols, rows = self._execute_sql(sql, row_limit=max_rows)
        except Exception as e:
            return f"SQL execution error: {e}\nSQL: {sql}"
        # Step 4: Summarize results using the LLM
        return self._summarize_rows(question, sql, cols, rows)

    def _summarize_rows(self, question: str, sql: str, cols, rows):
        # Create a small table preview and ask the LLM to summarize
        preview = []
        # Show up to 8 rows in preview
        preview_rows = rows[:8]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        preview.append(header)
        preview.append(sep)
        for r in preview_rows:
            preview.append("| " + " | ".join(str(x) for x in r) + " |")
        preview_text = "\n".join(preview) if cols else "No columns returned."

        system = SystemMessage(content="You are a concise assistant that summarizes SQL query results for users.")
        human = HumanMessage(content=f"""
        Question: {question}
        SQL: {sql}
        Results preview (up to 8 rows):
        {preview_text}

        Provide:
        1) A one-sentence summary of the main numeric or categorical result relevant to the question (if applicable).
        2) A short note about how many rows were returned in total (or 'no rows').
        3) If the user asked an aggregative/statistical question, include the numeric answer or the top few numbers.
        Keep it short and plain.
        """)
        resp = self.llm.invoke([system, human])
        summary = resp.content.strip()
        # Also include the raw row count
        total_count = len(rows)
        return f"{summary}\n\n(Raw SQL executed: {sql})"
