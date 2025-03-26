import chromadb
import dspy
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
db_collection = chroma_client.get_collection(name="sql_schema")

# Create SQLite persistent database connection
def create_db_connection():
    engine = create_engine("sqlite:///electrical_parts.db", echo=True)
    return engine.connect()

def validate_sql_query(query: str) -> bool:
    """Validate that the query is a SELECT statement and is not empty."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError(
            "Only SELECT queries are allowed for safety. Please rephrase your question to get information instead of modifying data."
        )
    return True

def execute_sql(query: str):
    """Executes the SQL query in SQLite and fetches results."""
    try:
        validate_sql_query(query)
        with create_db_connection() as conn:
            result = conn.execute(text(query)).fetchall()
        return result
    except Exception as e:
        return {"error": str(e), "valid": False}

# DSPy Retrieve Schema Module
class RetrieveSchema(dspy.Module):
    def forward(self, user_query: str):
        """Retrieves relevant schema details from ChromaDB."""
        table_results = db_collection.query(
            query_texts=[user_query], n_results=3, where={"type": "table"}
        )
        tables = [doc["table_name"] for doc in table_results.get("metadatas", [])[0]]

        column_results = db_collection.query(
            query_texts=[user_query],
            n_results=3,
            where={"$and": [{"type": {"$eq": "column"}}, {"table": {"$in": tables}}]},
        )
        columns = [
            (doc["table"], doc["columns"])
            for doc in column_results.get("metadatas", [])[0]
        ]

        relationship_results = db_collection.query(
            query_texts=[user_query], n_results=3, where={"type": "relationship"}
        )
        relationships = [
            (doc["table1"], doc["table2"], doc["relationship_type"])
            for doc in relationship_results.get("metadatas", [])[0]
        ]

        # pprint({"tables": tables, "columns": columns, "relationships": relationships})
        return {"tables": tables, "columns": columns, "relationships": relationships}

# DSPy Structured Output for SQL Generation
class GenerateSQL(dspy.Signature):
    """Generate a SQL query to answer the user's question.
       Take in consideration the history of the conversation too for better results"""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    history: list[str] = dspy.InputField()
    sql_query: str = dspy.OutputField()
    execution_result_observations: list = dspy.OutputField()
    answer: str = dspy.OutputField()

