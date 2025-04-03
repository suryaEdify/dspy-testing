'''
If you notice here, we are omitting context and history

If context is {} and history is [], DSPy will learn to ignore them when optimizing answers.
For other queries, context and history can still be used when necessary.
This ensures faster responses for direct questions without schema lookups.

'''


from dspy import Example

train_data = [
    # ❌ Unauthorized Operations (Standard Warning) - Empty context
    Example(
        question="Place an order for 2 Copper Wire 10m",
        context={},
        sql_query="",
        answer="Sorry, but you are not allowed to perform this operation."
    ).with_inputs("question", "context"),
    
    Example(
        question="I want to purchase LED Light Bulb 10W",
        context={},
        sql_query="",
        answer="Sorry, but you are not allowed to perform this operation."
    ).with_inputs("question", "context"),
    
    Example(
        question="Add 5 new products to the database.",
        context={},
        sql_query="",
        answer="Sorry, but you are not allowed to perform this operation."
    ).with_inputs("question", "context"),
    
    Example(
        question="Change the price of all products under $10 to $9.99.",
        context={},
        sql_query="",
        answer="Sorry, but you are not allowed to perform this operation."
    ).with_inputs("question", "context"),
    
    Example(
        question="Delete all customers who haven't placed an order in the last 6 months.",
        context={},
        sql_query="",
        answer="Sorry, but you are not allowed to perform this operation."
    ).with_inputs("question", "context"),
    
    # ✅ Allowed Queries (SELECT-based) - With proper context
    # Example(
    #     question="How many Mini Transformer 220V-110V are available?",
    #     context={
    #         "tables": ["products"],
    #         "columns": [("products", "name, stock_quantity")],
    #         "relationships": []
    #     },
    #     sql_query="SELECT stock_quantity FROM products WHERE name = 'transformer'",
    #     answer="There are 50 transformers in stock."
    # ).with_inputs("question", "context"),
    # 
    # Example(
    #     question="What is the cost of a 10W LED bulb?",
    #     context={
    #         "tables": ["products"],
    #         "columns": [("products", "name, price")],
    #         "relationships": []
    #     },
    #     sql_query="SELECT price FROM products WHERE name = '10W LED bulb'",
    #     answer="The price of a 10W LED bulb is $5.49."
    # ).with_inputs("question", "context"),
    # 
    # Example(
    #     question="List all categories of electrical parts.",
    #     context={
    #         "tables": ["categories"],
    #         "columns": [("categories", "name")],
    #         "relationships": []
    #     },
    #     sql_query="SELECT name FROM categories",
    #     answer="The categories are: Wires & Cables, Switches & Sockets, Lighting, Transformers."
    # ).with_inputs("question", "context"),
    # 
    # Example(
    #     question="How many orders did customer John Doe place?",
    #     context={
    #         "tables": ["customers", "orders"],
    #         "columns": [
    #             ("customers", "customer_id, name"),
    #             ("orders", "order_id, customer_id")
    #         ],
    #         "relationships": [
    #             ("orders", "customers", "foreign_key")
    #         ]
    #     },
    #     sql_query="SELECT COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE customers.name = 'John Doe'",
    #     answer="Customer John Doe has placed 3 orders."
    # ).with_inputs("question", "context"),
    # 
    # Example(
    #     question="Show me all products supplied by 'ElectroSupply Inc.'.",
    #     context={
    #         "tables": ["products", "suppliers"],
    #         "columns": [
    #             ("products", "product_id, name, supplier_id"),
    #             ("suppliers", "supplier_id, name")
    #         ],
    #         "relationships": [
    #             ("products", "suppliers", "foreign_key")
    #         ]
    #     },
    #     sql_query="SELECT p.name FROM products p JOIN suppliers s ON p.supplier_id = s.supplier_id WHERE s.name = 'ElectroSupply Inc.'",
    #     answer="Here are the products supplied by 'ElectroSupply Inc.': Copper Wire 10m, LED Bulb 10W, Electrical Tape"
    # ).with_inputs("question", "context"),
    # 
    # Example(
    #     question="What is the total revenue from orders this month?",
    #     context={
    #         "tables": ["orders"],
    #         "columns": [("orders", "order_date, total_amount")],
    #         "relationships": []
    #     },
    #     sql_query="SELECT SUM(total_amount) FROM orders WHERE strftime('%Y-%m', order_date) = strftime('%Y-%m', 'now')",
    #     answer="The total revenue from orders this month is $10,240.75."
    # ).with_inputs("question", "context"),
    # 
    # # More examples with JOINs and complex queries
    # Example(
    #     question="Find all products in the 'Lighting' category with less than 20 in stock",
    #     context={
    #         "tables": ["products", "categories"],
    #         "columns": [
    #             ("products", "product_id, name, category_id, stock_quantity"),
    #             ("categories", "category_id, name")
    #         ],
    #         "relationships": [
    #             ("products", "categories", "foreign_key")
    #         ]
    #     },
    #     sql_query="""SELECT p.name, p.stock_quantity 
    #                 FROM products p 
    #                 JOIN categories c ON p.category_id = c.category_id 
    #                 WHERE c.name = 'Lighting' AND p.stock_quantity < 20""",
    #     answer="Products in Lighting category with low stock: LED Bulb 10W (15), LED Tube Light (10)"
    # ).with_inputs("question", "context"),
    # 
    # Example(
    #     question="Which customers haven't placed any orders?",
    #     context={
    #         "tables": ["customers", "orders"],
    #         "columns": [
    #             ("customers", "customer_id, name"),
    #             ("orders", "order_id, customer_id")
    #         ],
    #         "relationships": [
    #             ("orders", "customers", "foreign_key")
    #         ]
    #     },
    #     sql_query="""SELECT c.name 
    #                 FROM customers c 
    #                 LEFT JOIN orders o ON c.customer_id = o.customer_id 
    #                 WHERE o.order_id IS NULL""",
    #     answer="Customers without orders: Jane Smith, Robert Johnson"
    # ).with_inputs("question", "context")
]