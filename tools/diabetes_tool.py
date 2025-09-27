# tools/diabetes_tool.py
from tools.db_tool_base import DBToolBase
class DiabetesDBTool(DBToolBase):
    def __init__(self, db_path='dbs/diabetes.db', table_name='diabetes', llm_model_name='gpt-4o-mini', unsafe_mode=False):
        super().__init__(db_path=db_path, table_name=table_name, llm_model_name=llm_model_name, unsafe_mode=unsafe_mode)
