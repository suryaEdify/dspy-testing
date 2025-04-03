import chromadb
import dspy
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from train_set import train_data
import os

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
        # validate_sql_query(query)
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
    """Generate appropriate response to the user's question about the database.
        Dont just give tabular data, instead give in a meaningful polite sentence format with proper result and not just [result]
    For SELECT queries: Generate SQL and answer.
    For other operations: Return standard warning message."""
    
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    history: list[str] = dspy.InputField(default=[])
    sql_query: str = dspy.OutputField(desc="Empty if operation not allowed")
    answer: str = dspy.OutputField()


lm = dspy.LM('groq/qwen-2.5-32b', api_key=os.getenv('GROQ_API_KEY'))
dspy.configure(lm=lm)


# Define your ReAct module
sql_query_generator = dspy.ReAct(GenerateSQL, tools=[execute_sql])

# Define a wrapper module for optimization
class SQLAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = RetrieveSchema()
        self.react = sql_query_generator
        
    def forward(self, question, context=None, history=None):
        history = history or []
        if context is None:
            context = self.retrieve(question)
            
        response = self.react(question=question, context=context, history=history)
        
        # ReAct returns the full trace, we need to extract the final prediction
        if hasattr(response, 'answer'):
            return dspy.Prediction(
                answer=response.answer,
                sql_query=getattr(response, 'sql_query', '')
            )
        return dspy.Prediction(answer="Error: No valid response generated", sql_query="")
    

def validate_prediction(example, pred, trace=None):
    try:
        # For non-SELECT operations
        if not example.sql_query:
            return pred.answer.startswith("Sorry, but you are not allowed")
        
        # For SELECT queries
        if not hasattr(pred, 'sql_query') or not pred.sql_query:
            return False
            
        return (example.sql_query.lower().strip() == pred.sql_query.lower().strip() and 
                example.answer.lower() in pred.answer.lower())
    except Exception as e:
        print(f"Validation error: {e}")
        return False


optimizer = dspy.BootstrapFewShot(
    metric=validate_prediction,
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
    teacher_settings=dict(lm=dspy.LM('groq/qwen-2.5-32b', api_key=os.getenv('GROQ_API_KEY')))
)

agent = SQLAgent()

# Optimize
optimized_agent = optimizer.compile(
    agent, 
    trainset=train_data[:5]
)

# Save/Load
optimized_agent.save('optimized_agent.json')

# Usage remains the same
response = optimized_agent("How many transformers are in stock?")
print(response.answer)  # "Sorry, but you are not allowed..."