# ---------------- Loading Python Packages ---------------- 

# Importing Libraries
import os
import sys
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

# Setting up Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "etl.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.FileHandler(log_file, mode="a"),
                        logging.StreamHandler()
                    ])

# Adding utils to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Loading Environment File
load_dotenv()

# Loading Database Credentials from Environment File
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Creating SQLAlchemy Engine to upload the Data to Neon PostgreSQL Database
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?sslmode=require&channel_binding=require", pool_pre_ping=True)
logging.info("SQLAlchemy Engine created successfully")

# ---------------- Check if Table Exists ----------------

# Check if tables already exists in the Neon PostgreSQL Database
inspector = inspect(engine)
existing_tables = inspector.get_table_names()

# If all required tables exist, skip ETL Setup
if set(["customers", "products", "orders"]).issubset(existing_tables):
    logging.info("Tables already exist. Skipping ETL setup.")
else:
    logging.info("Tables do not exist. Running ETL setup...")

    # Reading CSV File
    from utils.read_data import load_csv
    df = load_csv("data", "sales_data.csv")
    logging.info("CSV file loaded successfully")

    # ---------------- Data Cleaning ---------------- 

    # Handling Duplicate Data
    from utils.drop_duplicates import remove_duplicates
    df = remove_duplicates(df)

    # Standardizing Column Names
    from utils.standardize_columns import standardize_column_names
    df = standardize_column_names(df)

    # Optimizing DataFrame
    from utils.data_preprocessor import optimize_dataframe
    df = optimize_dataframe(df)

    logging.info("Data cleaned and optimized successfully")

    # ---------------- Schema Creation ----------------

    # SQL Query to Create Schema
    schema_sql = [
    """
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        customer_name TEXT,
        segment TEXT,
        city TEXT,
        state TEXT,
        country TEXT,
        postal_code NUMERIC,
        region TEXT
    );
    """
    ,
    """
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT,
        category TEXT,
        sub_category TEXT
    );
    """
    ,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        order_date DATE,
        customer_id TEXT,
        product_id TEXT,
        ship_mode TEXT,
        ship_date DATE,
        sales NUMERIC,
        quantity INTEGER,
        discount NUMERIC,
        profit NUMERIC,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """]

    # Creating Schema in Neon PostgreSQL Database
    with engine.connect() as conn:
        for stm in schema_sql:
            conn.execute(text(stm))
        conn.commit()
    logging.info("Database schema created successfully")

    # Customers Table
    customers_cols = ["customer_id", "customer_name", "segment", "city", "state", "country", "postal_code", "region"]
    customers = df[customers_cols].drop_duplicates(subset="customer_id")

    # Products Table
    products_cols = ["product_id", "product_name", "category", "sub_category"]
    products = df[products_cols].drop_duplicates(subset="product_id")

    # Orders Table
    orders_cols = ["order_id", "order_date", "customer_id", "product_id", "ship_mode", "ship_date", "sales", "quantity", "discount", "profit"]
    orders = df[orders_cols].drop_duplicates(subset="order_id")