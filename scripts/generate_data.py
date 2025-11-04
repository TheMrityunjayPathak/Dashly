# ---------------- Loading Python Packages ---------------- 

# Importing Libraries
import os
import sys
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ---------------- Configuration ---------------- 

# Setting up Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "generate_data.log")
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

# ---------------- Generating Random Customers Data ---------------- 

# Generating New Customers Data Randomly
from utils.generate_random_data import random_customers_data
customers = random_customers_data(20, 40)
logging.info("Random customers data generated successfully")

# ---------------- Generating Random Orders Data ---------------- 

# Generating New Orders Data Randomly
from utils.generate_random_data import random_orders_data
orders = random_orders_data(60, 80)
logging.info("Random orders data generated successfully")

# ---------------- Data Cleaning ---------------- 

# Handling Duplicate Data
from utils.drop_duplicates import remove_duplicates
customers = remove_duplicates(customers)
orders = remove_duplicates(orders)

# Standardizing Column Names
from utils.standardize_columns import standardize_column_names
customers = standardize_column_names(customers)
orders = standardize_column_names(orders)

# Optimizing DataFrame
from utils.data_preprocessor import optimize_customers, optimize_orders
customers = optimize_customers(customers)
orders = optimize_orders(orders)

logging.info("Data cleaned and optimized successfully")

# ---------------- Appending New Data to Neon PostgreSQL Database ---------------- 

# Appending new and unique data to Neon PostgreSQL Database
from utils.append_data import append_unique_data_to_db
append_unique_data_to_db("customers", customers, "customer_id", engine)
append_unique_data_to_db("orders", orders, "order_id", engine)
logging.info("New data uploaded to database successfully")