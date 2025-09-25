# tools/heart_disease_tool.py
from tools.db_tool_base import DBToolBase

class HeartDiseaseDBTool(DBToolBase):
    def __init__(self, db_path='dbs/heart_disease.db', table_name='heart_disease', llm_model_name='gpt-4o-mini'):
        super().__init__(db_path=db_path, table_name=table_name, llm_model_name=llm_model_name)
