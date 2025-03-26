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


class GetterSignature(dspy.Signature):
    """Try to get product name and quantity from the user's message
        If Product name or quantity is not provided, ask user to provide it"""
    user_input: str = dspy.InputField(desc="user's message")
    product_name: str = dspy.OutputField(desc="Get Product Name from the user's message")
    quantity: int = dspy.OutputField(desc="Get Quantity from the user's message")
    product_name_present: bool = dspy.OutputField(desc="Say whether the product name is provided or not")
    quantity_present: bool = dspy.OutputField(desc="Say whether the quantity is provided or not")

class GetterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(GetterSignature)

    def forward(self, user_input: str):
        extracted = self.predict(user_input=user_input)
        return extracted


details_predictor = GetterModule()


class BuySentiment(dspy.Signature):
    """Tell if user intents to buy anything or not
        If user asks about price, quantity, etc, return False
        Only return True if user asks about buying something"""
    user_input: str = dspy.InputField(desc="user's message")
    buy_intent: bool = dspy.OutputField(desc="True if user intents to buy anything, False otherwise")

class BuyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(BuySentiment)

    def forward(self, user_input: str):
        extracted = self.predict(user_input=user_input)
        return extracted.buy_intent

BUY_INTENT_PREDICTOR = BuyModule()


def is_product_available(product_name, quantity, db_path="electrical_parts.db"):
    """Check if the product is available in the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT product_id, price FROM products WHERE name = ? AND stock_quantity > ?", (product_name, quantity))
    result = cursor.fetchone()

    conn.close()
    if result:
        return result
    else:
        return False

def calculate_total_price(product_name, quantity, db_path="electrical_parts.db"):
    """Calculate the total price of the product"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT price FROM products WHERE name = ?", (product_name))
    result = cursor.fetchone()
    return result and result[0] * quantity


def placing_order(quantity, total_price, product_id, unit_price, db_path="electrical_parts.db"):
    """Place order by updating the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO orders (customer_id, order_date, total_amount) VALUES (?, ?, ?)", (1, datetime.date.today(), total_price))
    order_id = cursor.lastrowid

    cursor.execute("INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)", (order_id, product_id, quantity, unit_price))
    cursor.execute("UPDATE products SET stock_quantity = stock_quantity - ? WHERE product_id = ?", (quantity, product_id))

    conn.commit()
    conn.close()



# Define a DSPy Signature for formatting tabular output
class FormatAsTable(dspy.Signature):
    """First appolagize for the inconvenience that these are the only products available for tim and 
       then Converts raw product details into a readable table format."""
    raw_data = dspy.InputField(desc="List of product details from the database")
    formatted_table = dspy.OutputField(desc="Apology and Formatted table representation")

# DSPy Module for Formatting
table_formatter = dspy.Predict(FormatAsTable)

def fetch_product_details(db_path="electrical_parts.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.name AS product_name, p.price, p.stock_quantity, 
               c.name AS category_name, s.name AS supplier_name
        FROM products p
        JOIN categories c ON p.category_id = c.category_id
        JOIN suppliers s ON p.supplier_id = s.supplier_id;
    """)

    result = cursor.fetchall()
    conn.close()

    # Convert raw data into structured input for DSPy
    formatted_result = table_formatter(raw_data=result)
    
    return formatted_result.formatted_table  # Extract the formatted output


retrieve_schema = RetrieveSchema()
_buy = False


class SQLChatbot(dspy.Module):

    def __init__(self):
        super().__init__()
        self.history = []

    def forward(self, user_input: str):
        """Handles user queries and maintains conversation history for better results
           We will first check if user wants to buy something or just querying about the database"""

        buy_intent = BUY_INTENT_PREDICTOR(user_input)
        _buy = buy_intent

        if _buy:

            details = details_predictor(user_input)
            if details.product_name_present and details.quantity_present:
                product_name = details.product_name
                quantity = details.quantity

                # Check if the product is available in the database
                product_available = is_product_available(product_name, quantity)
                if product_available:
                    print(product_available)
                    product_id, price = product_available
                    # Calculate total price
                    total_price = price * quantity
                    placing_order(quantity, total_price, product_id, price)
                    return "Your Order has been placed successfully!"
                else:
                    table_details = fetch_product_details()
                    
                    return table_details

                
            else:
                return "Please provide the product name and quantity"
        
        else:

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