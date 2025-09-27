# tools/cancer_tool.py
from tools.db_tool_base import DBToolBase
class CancerDBTool(DBToolBase):
    def __init__(self, db_path='dbs/cancer.db', table_name='cancer', llm_model_name='gpt-4o-mini', unsafe_mode=False):
        super().__init__(db_path=db_path, table_name=table_name, llm_model_name=llm_model_name, unsafe_mode=unsafe_mode)
