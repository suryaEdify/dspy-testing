import os
import dspy
from dotenv import load_dotenv
from sql_agent import GenerateSQL, execute_sql, RetrieveSchema
import sqlite3
import datetime
load_dotenv()


## This is simple placing order flow is just to show that why DSpy isnt capable 
## of handling conversational flows like how Langchain does it!

## The default user is take as [Alice Cooper]
### To change the user, during conversatrion, there's another huge
### logic to be implemented, which is not implemented here!

lm = dspy.LM('groq/llama-3.3-70b-versatile', api_key=os.getenv('GROQ_API_KEY'))
dspy.configure(lm=lm)

sql_query_generator = dspy.ReAct(GenerateSQL, tools=[execute_sql])


retrieve_schema = RetrieveSchema()


class SQLChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.history = []

    def forward(self, user_input: str):
        """Handles user queries and maintains conversation history for better results"""

        query_context = retrieve_schema(user_input)
        response = sql_query_generator(question=user_input, context=query_context, history=self.history)
        # Store conversation history
        self.history.append(f"User: {user_input}")
        self.history.append(f"Bot: {response.answer}")
        return response.answer
        

assistant = SQLChatbot()

print("Assistant: Ask me database queries! Type 'exit' to quit.")
while True:
    #logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = assistant(user_input)
    print(f"Assistant: {response}")