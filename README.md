<h2 align="center">Dashly : Live Sales Dashboard</h2>

<a href="https://app.powerbi.com/view?r=eyJrIjoiZTgxODBhYmMtYjc1Zi00YjVkLWIyZDItZDYxY2RjZmIwNGY5IiwidCI6ImZhYjAyYzVkLTYxYjYtNGIxMi05ZTY2LTdhMDhkOWY0ZmNjMSJ9&pageName=5b9aaf645951a59cacdc">
<img title="Dashly" src="https://github.com/user-attachments/assets/be84d954-9d52-4692-8a77-d40853a91d61">
</a>

## Table of Contents
- [Problem Statement](#problem-statement)
- [Project Objective](#project-objective)
- [Setup](#setup)
- [ER Diagram](#er-diagram)
- [Database Schema](#database-schema)
- [SQL Views](#sql-views)
- [ETL Pipeline](#etl-pipeline)
- [GitHub Actions](#github-actions)
- [Power BI Dashboard](#power-bi-dashboard)
- [Results & Insights](#results--insights)
- [How To](#how-to)
- [Folder Structure](#folder-structure)
- [License](#license)

<hr>

## Problem Statement
**Quick Buy** is a leading superstore operating across the United States, sells thousands of products across regions.

Currently, the store relies heavily on multiple spreadsheets and manual SQL queries to track business performance.
- This manual process makes it :
  - Hard to keep data structured and consistent.
  - Time-consuming to prepare reports.
  - Difficult to update daily transaction data manually.
  - Challenging for non-technical users to extract and understand key business insights.
    
- To solve these challenges, **Quick Buy** aims to build an automated system that :
    - Collects and stores all data in a structured database.
    - Cleans and updates data automatically using an ETL Pipeline.
    - Uses SQL to analyze data and extract meaningful business insights.
    - Displays insights through an interactive Power BI Dashboard.
    - Helps managers and executives quickly understand business performance and make better decisions.

<hr>

## Project Objective
Build an automated system for **Quick Buy** that provides accurate and up-to-date business insights.

To achieve this, the project focuses on the following objectives :
- **Centralized Data Storage**
    - Store all business data (orders, customers and products) in a single SQL database.
    - Ensure data is well-structured, accurate and easy to access.
- **Automated ETL Pipeline**
    - Build an ETL (Extract, Transform, Load) pipeline using Python and SQLAlchemy.
    - Automatically load new data into the database on a daily schedule.
    - Reduce manual reporting work and save time for the analytics.
- **Data Cleaning and Transformation**
    - Automatically clean raw data by fixing missing, duplicate or inconsistent values.
    - Ensure that data is clean and ready for the analysis.
- **Secure and Stable Database Connection**
    - Use a secure connection between Python and the database.
    - Store credentials safely using configuration files or environment variables.
    - Keep the system stable with low chances of connection failure.
- **SQL Views for Business Insights**
    - Create SQL views to simplify complex queries and calculations.
    - Use these views as a consistent data source for the dashboard.
- **Interactive Power BI Dashboard**
    - Build dashboard to visualize different aspects of the business like orders, customers, etc.
    - Make dashboard simple, clear and useful for both technical and non-technical users.
- **Daily Report Refresh & Monitoring**
    - Schedule automatic daily dashboard refreshes to display the most up-to-date data.
    - Add basic logging to monitor ETL runs and quickly identify failures.

<hr>

## Setup

### 1. Clone the Repository
First, you need to clone the project from GitHub to your local system.
```bash
git clone https://github.com/TheMrityunjayPathak/Dashly.git
```

### 2. Setup a Virtual Environment
To avoid version conflicts and keep your project isolated, create a virtual environment.

On Windows :
```bash
python -m venv .venv
```

On macOS/Linux :
```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment
After setting up the virtual environment, activate it to begin installing dependencies.

On Windows :
```bash
.\.venv\Scripts\activate
```

On macOS/Linux :
```bash
source .venv/bin/activate
```

### 4. Install the Project Dependencies
Now, install all the required libraries inside your virtual environment using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

> [!TIP]
> It's a good idea to upgrade `pip` before installing dependencies to avoid compatibility issues.
```bash
pip install --upgrade pip
```

> [!NOTE]
> Use the same Python version as in `.github/workflows/etl_pipeline.yml` to avoid compatibility issues.

### 5. Setup Environment Variables
This project uses a `.env` file to store database credentials like `DB_USER`, `DB_PASS`, `DB_NAME`, etc.

It stores environment variables in plain text.
```python
# .env
DB_HOST=host_name
DB_NAME=database_name
DB_USER=user_name
DB_PASS=password
```

> [!IMPORTANT]
> Make sure not to commit your `.env` file to GitHub or any public repositories.
> 
> You can add it to `.gitignore` to ensure it's excluded from version control.

> [!NOTE]
> If you want to create a free Database in Neon and connect it with Python, go to [How To](#how-to) section.

### 6. Database Connectivity Check
To confirm that the PostgreSQL connection works before running ETL scripts.

This avoids script crashes due to invalid credentials or blocked ports.

```python
# Importing Libraries
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Loading Environment File
load_dotenv()

# Loading Database Credentials from Environment File
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Creating SQLAlchemy Engine to upload the Data to Neon PostgreSQL Database
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?sslmode=require&channel_binding=require", pool_pre_ping=True)
```

### 7. Run ETL Script
This initializes the database and :
- Cleans raw CSV data
- Creates tables (`customers`, `orders`, `products`)
- Loads data into Neon PostgreSQL Database.
```bash
python scripts/etl.py
```

> [!NOTE]
> Run this only once initially or when you want a full database refresh.

### 8. Create SQL Views
This script builds reusable SQL views that summarize business metrics for Power BI Dashboard.

It simplifies queries, ensures consistent logic and improves performance.
```bash
python scripts/create_views.py
```

### 9. Generate New Data (Optional)
Simulates daily transactions by generating new random data for testing pipeline automation.

Helps verify how dashboards respond to new data over time.
```bash
python scripts/generate_data.py
```

### 10. Export Views as CSVs (Optional)
Exports SQL views results to CSV files inside the `views/` folder.

It is useful for sharing datasets or validating dashboard data without connecting to the database.
```bash
python scripts/export_views.py
```

### 11. Simulate Full Workflow Locally
Running all scripts manually helps confirm that everything works locally before enabling the scheduled GitHub run.
```bash
python scripts/etl.py
python scripts/create_views.py
python scripts/generate_data.py
python scripts/export_views.py
```

### 12. Check Logs
- Check log files inside `logs/` folder :
  - `etl.log` → Initial data loading.
  - `create_views.log` → SQL views creation.
  - `generate_data.log` → Daily data generation.
- Logs help you monitor pipeline performance and troubleshoot errors quickly.

<hr>

## ER Diagram
The ER (Entity-Relationship) Diagram visually represents how different tables in the database are related.

### Relationships
- **One Customer → Many Orders**
  - Each customer can place multiple orders.
- **One Product → Many Orders**
  - Each product can appear in multiple orders.
- **Orders Table → Central Table**
  - Serves as the main transactional table, linking customers and products.

<img width="500px" title="ER Diagram" src="https://github.com/user-attachments/assets/e8767755-356f-4f92-9820-cf4438c8cbc5">

<hr>

## Database Schema
The database is designed to store and organize **Quick Buy's** orders, customers and products data.

It ensures that all business data is centralized, consistent and easy to query for analysis and dashboarding.

### SQL Schema Definition 
```sql
/* Customers Table */
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
```

```sql
/* Products Table */
CREATE TABLE IF NOT EXISTS products (
  product_id TEXT PRIMARY KEY,
  product_name TEXT,
  category TEXT,
  sub_category TEXT
);
```

```sql
/* Orders Table */
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
  shipping_duration INTEGER,
  profit_margin NUMERIC,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

<hr>

## SQL Views
SQL Views are used to make data analysis easier and keep all business metrics consistent for dashboards.

Instead of running complex queries every time, Power BI connects directly to these views to fetch clean data.

<details>
  <summary>Click Here for detailed SQL Views</summary>
  <br>

  ```sql
  /* segment_wise_sales_and_profit */
  /* Calculates total sales and profit for each customer segment. */
  CREATE OR REPLACE VIEW segment_wise_sales_and_profit AS
  SELECT 
      c.segment,
      SUM(o.sales) AS total_sales,
      SUM(o.profit) AS total_profit
  FROM orders AS o
  JOIN customers AS c
  ON o.customer_id = c.customer_id
  GROUP BY c.segment;
  ```

  ```sql
  /* region_wise_sales_and_profit */
  /* Summarizes total sales and profit across all regions. */
  CREATE OR REPLACE VIEW region_wise_sales_and_profit AS
  SELECT 
      c.region,
      SUM(o.sales) AS total_sales,
      SUM(o.profit) AS total_profit
  FROM orders AS o
  JOIN customers AS c
  ON o.customer_id = c.customer_id
  GROUP BY c.region;
  ```

  ```sql
  /* month_wise_sales_and_profit */
  /* Shows monthly trends of total sales and profit. */
  CREATE OR REPLACE VIEW month_wise_sales_and_profit AS 
  SELECT
      TO_CHAR(order_date, 'Mon') AS month, 
      SUM(sales) AS total_sales,
      SUM(profit) AS total_profit
  FROM orders
  GROUP BY month;
  ```

  ```sql
  /* top_customers_by_sales */
  /* Lists customers with their total sales and profit to identify top performers. */
  CREATE OR REPLACE VIEW top_customers_by_sales AS
  SELECT
      c.customer_id, 
      c.customer_name,
      SUM(o.sales) AS total_sales,
      SUM(o.profit) AS total_profit
  FROM orders AS o
  JOIN customers AS c
  ON o.customer_id = c.customer_id
  GROUP BY c.customer_id, c.customer_name;
  ```

  ```sql
  /* shipping_performance */
  /* Analyzes sales and profit performance by shipping mode. */
  CREATE OR REPLACE VIEW shipping_performance AS 
  SELECT 
      ship_mode,
      SUM(sales) AS total_sales,
      SUM(profit) AS total_profit
  FROM orders
  GROUP BY ship_mode;
  ```

  ```sql
  /* overall_sales_performance */
  /* Provides overall business KPIs like total sales, profit, orders and customers. */
  CREATE OR REPLACE VIEW overall_sales_performance AS
  SELECT
      SUM(sales) AS total_sales,
      SUM(profit) AS total_profit,
      COUNT(DISTINCT order_id) AS total_orders,
      COUNT(DISTINCT customer_id) AS total_customers,
      COUNT(DISTINCT product_id) AS total_products,
      SUM(quantity) AS total_quantity_sold
  FROM orders;
  ```

  ```sql
  /* state_wise_sales_and_customer_base */
  /* Displays total sales and customer count by U.S. states. */
  CREATE OR REPLACE VIEW state_wise_sales_and_customer_base AS 
  SELECT
      c.state,
      SUM(o.sales) AS total_sales,
      COUNT(DISTINCT c.customer_id) AS total_customers
  FROM orders AS o
  JOIN customers AS c
  ON o.customer_id = c.customer_id
  GROUP BY c.state;
  ````

  ```sql
  /* segment_wise_monthly_sales_and_profit */
  /* Tracks monthly sales and profit performance for each customer segment. */
  CREATE OR REPLACE VIEW segment_wise_monthly_sales_and_profit AS
  SELECT
      c.segment,
      TO_CHAR(o.order_date, 'Mon') AS month_name,
      SUM(o.sales) AS total_sales,
      SUM(o.profit) AS total_profit
  FROM orders AS o
  JOIN customers AS c
  ON o.customer_id = c.customer_id
  GROUP BY c.segment, month_name;
  ```

  ```sql
  /* region_wise_monthly_sales */
  /* Shows monthly sales trends for each region. */
  CREATE OR REPLACE VIEW region_wise_monthly_sales AS
  SELECT
      c.region,
      TO_CHAR(o.order_date, 'Mon') AS month_name,
      SUM(o.sales) AS total_sales
  FROM orders AS o
  JOIN customers AS c
  ON o.customer_id = c.customer_id
  GROUP BY c.region, month_name;
  ```

  ```sql
  /* overall_customers_performance */
  /* Calculates average sales, profit, orders and quantity per customer. */
  CREATE OR REPLACE VIEW overall_customers_performance AS
  SELECT 
      ROUND(SUM(o.sales)/COUNT(DISTINCT o.customer_id)) AS avg_sales_per_customer,
      ROUND(SUM(o.profit)/COUNT(DISTINCT o.customer_id)) AS avg_profit_per_customer,
      ROUND(COUNT(DISTINCT order_id)/COUNT(DISTINCT customer_id)) AS avg_orders_per_customer,
      ROUND(SUM(o.quantity)/COUNT(DISTINCT o.customer_id)) AS avg_quantity_per_customer
  FROM orders AS o;
  ```

  ```sql
  /* avg_discount_per_order_per_customer */
  /* Computes the average discount per customer across all orders. */
  CREATE OR REPLACE VIEW avg_discount_per_order_per_customer AS
  SELECT
      ROUND(AVG(customer_avg), 2) AS avg_discount_per_customer
  FROM (
      SELECT customer_id, AVG(discount) AS customer_avg
      FROM orders
      GROUP BY customer_id
  ) AS sub;
  ```

  ```sql
  /* category_wise_monthly_sales_and_profit */
  /* Tracks monthly sales and profit for each product category. */
  CREATE OR REPLACE VIEW category_wise_monthly_sales_and_profit AS
  SELECT
      p.category,
      TO_CHAR(o.order_date, 'Mon') AS month,
      SUM(o.sales) AS total_sales,
      SUM(o.profit) AS total_profit
  FROM orders AS o
  JOIN products AS p
  ON o.product_id = p.product_id
  GROUP BY p.category, month;
  ```

  ```sql
  /* sub_category_wise_sales_and_profit */
  /* Summarizes total sales and profit by product sub-category. */
  CREATE OR REPLACE VIEW sub_category_wise_sales_and_profit AS
  SELECT
      p.sub_category,
      SUM(o.sales) AS total_sales,
      SUM(o.profit) AS total_profit
  FROM orders AS o
  JOIN products AS p
  ON o.product_id = p.product_id
  GROUP BY p.sub_category;
  ```

  ```sql
  /* category_wise_sales_profit_and_orders */
  /* Shows total sales, profit and order count by product category. */
  CREATE OR REPLACE VIEW category_wise_sales_profit_and_orders AS
  SELECT
      p.category,
      SUM(o.sales) AS total_sales,
      SUM(o.profit) AS total_profit,
      COUNT(DISTINCT o.order_id) AS total_orders
  FROM orders AS o
  JOIN products AS p
  ON o.product_id = p.product_id
  GROUP BY p.category;
  ```

  ```sql
  /* state_wise_most_purchased_sub_category */
  /* Identifies the most purchased sub-category in each U.S. state. */
  CREATE OR REPLACE VIEW state_wise_most_purchased_sub_category AS
  SELECT
      c.state,
      p.sub_category, 
      SUM(o.quantity) AS quantity_sold,
      RANK() OVER (PARTITION BY c.state ORDER BY SUM(o.quantity) DESC) AS sub_category_rank
  FROM orders AS o
  JOIN products AS p
  ON o.product_id = p.product_id
  JOIN customers AS c
  ON o.customer_id = c.customer_id
  GROUP BY c.state, p.sub_category;
  ```
  
</details>

<hr>

## ETL Pipeline
- The ETL (Extract, Transform, Load) Pipeline is the core part of this project.
- It automatically cleans and loads validated sales data into a PostgreSQL database for Power BI dashboard.
- It is built with Python using SQLAlchemy and is reliable and securely configured via environment variables.

### ETL Pipeline Structure
| **Script Name**    | **Purpose**                                                                                    |
| :----------------- | :--------------------------------------------------------------------------------------------- |
| `etl.py`           | Sets up the database schema, cleans the dataset and loads initial data into the database.      |
| `create_views.py`  | Creates multiple SQL views that summarizes and aggregates data for Power BI Dashboard.         |
| `generate_data.py` | Generates random synthetic transactions data to simulate daily updates in the database.        |

### How the ETL Pipeline works?
#### 1. `etl.py`
- This script handles the first step of the process, getting your database ready.

<details>
<summary>Click Here to view the entire Script</summary>
<br>
  
```python
# ---------------- Loading Python Packages ---------------- 

# Importing Libraries
import os
import sys
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

# ---------------- Configuration ---------------- 

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

    # Truncate Tables to avoid uploading Duplicate Data
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE customers CASCADE;"))
        conn.execute(text("TRUNCATE TABLE products CASCADE;"))
        conn.execute(text("TRUNCATE TABLE orders CASCADE;"))
    logging.info("Tables truncated successfully")

    # Appending data to the tables in Neon PostgreSQL Database
    customers.to_sql("customers", engine, if_exists="append", index=False)
    products.to_sql("products", engine, if_exists="append", index=False)
    orders.to_sql("orders", engine, if_exists="append", index=False)
    logging.info("Initial data uploaded to Neon PostgreSQL Database successfully")
```

</details>

#### What is does?
- **Load Configuration**
  - Reads environment variables (like `DB_HOST`, `DB_NAME`) from a `.env` file for secure database access.
- **Logging Setup**
  - Creates a `logs/etl.log` file to track all ETL activity and errors.
- **Extract Data**
  - Loads raw data from a CSV file using a custom `load_csv()` utility function.
- **Transform Data**
  - Removes duplicates, standardizes column names and optimizes datatypes.
- **Load Data**
  - Creates tables in Neon PostgreSQL Database and loads the cleaned data using `to_sql()` function.
- **Schema Management**
  - Ensures relationships between tables using foreign keys and maintains data integrity.
  
---

#### 2. `create_views.py`
- This script builds SQL views in the PostgreSQL Database to simplify analysis and reporting in Power BI.

<details>
<summary>Click Here to view the entire Script</summary>
<br>
  
```python
# ---------------- Loading Python Packages ---------------- 

# Importing Libraries
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------- Configuration ---------------- 

# Setting up Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "create_views.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.FileHandler(log_file, mode="a"),
                        logging.StreamHandler()
                    ])

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

# ---------------- Creating Views in Neon PostgreSQL Database ----------------

# Dictionary of SQL Views
sql_views = {
    "segment_wise_sales_and_profit":"""
        DROP VIEW IF EXISTS segment_wise_sales_and_profit;
        CREATE OR REPLACE VIEW segment_wise_sales_and_profit AS
        SELECT 
            c.segment,
            SUM(o.sales) AS total_sales,
            SUM(o.profit) AS total_profit
        FROM orders AS o
        JOIN customers AS c
        ON o.customer_id = c.customer_id
        GROUP BY c.segment;
    """
    ,
    "region_wise_sales_and_profit":"""
        DROP VIEW IF EXISTS region_wise_sales_and_profit;
        CREATE OR REPLACE VIEW region_wise_sales_and_profit AS
        SELECT 
            c.region,
            SUM(o.sales) AS total_sales,
            SUM(o.profit) AS total_profit
        FROM orders AS o
        JOIN customers AS c
        ON o.customer_id = c.customer_id
        GROUP BY c.region;
    """
    ,
    "month_wise_sales_and_profit":"""
        DROP VIEW IF EXISTS month_wise_sales_and_profit;
        CREATE OR REPLACE VIEW month_wise_sales_and_profit AS 
        SELECT
            TO_CHAR(order_date, 'Mon') AS month, 
            SUM(sales) AS total_sales,
            SUM(profit) AS total_profit
        FROM orders
        GROUP BY month;
    """
    ,
    "top_customers_by_sales":"""
        DROP VIEW IF EXISTS top_customers_by_sales;
        CREATE OR REPLACE VIEW top_customers_by_sales AS
        SELECT
            c.customer_id, 
            c.customer_name,
            SUM(o.sales) AS total_sales,
            SUM(o.profit) AS total_profit
        FROM orders AS o
        JOIN customers AS c
        ON o.customer_id = c.customer_id
        GROUP BY c.customer_id, c.customer_name;
    """
    ,
    "shipping_performance":"""
        DROP VIEW IF EXISTS shipping_performance;
        CREATE OR REPLACE VIEW shipping_performance AS 
        SELECT 
            ship_mode,
            SUM(sales) AS total_sales,
            SUM(profit) AS total_profit
        FROM orders
        GROUP BY ship_mode;
    """
    ,
    "overall_sales_performance":"""
        DROP VIEW IF EXISTS overall_sales_performance;
        CREATE OR REPLACE VIEW overall_sales_performance AS
        SELECT
            SUM(sales) AS total_sales,
            SUM(profit) AS total_profit,
            COUNT(DISTINCT order_id) AS total_orders,
            COUNT(DISTINCT customer_id) AS total_customers,
            COUNT(DISTINCT product_id) AS total_products,
            SUM(quantity) AS total_quantity_sold
        FROM orders;
    """
    ,
    "state_wise_sales_and_customer_base":"""
        DROP VIEW IF EXISTS state_wise_sales_and_customer_base;
        CREATE OR REPLACE VIEW state_wise_sales_and_customer_base AS 
        SELECT
            c.state,
            SUM(o.sales) AS total_sales,
            COUNT(DISTINCT c.customer_id) AS total_customers
        FROM orders AS o
        JOIN customers AS c
        ON o.customer_id = c.customer_id
        GROUP BY c.state;
    """
    ,
    "segment_wise_monthly_sales_and_profit":"""
        DROP VIEW IF EXISTS segment_wise_monthly_sales_and_profit;
        CREATE OR REPLACE VIEW segment_wise_monthly_sales_and_profit AS
        SELECT
            c.segment,
            TO_CHAR(o.order_date, 'Mon') AS month_name,
            SUM(o.sales) AS total_sales,
            SUM(o.profit) AS total_profit
        FROM orders AS o
        JOIN customers AS c
        ON o.customer_id = c.customer_id
        GROUP BY c.segment, month_name;
    """
    ,
    "region_wise_monthly_sales":"""
        DROP VIEW IF EXISTS region_wise_monthly_sales;
        CREATE OR REPLACE VIEW region_wise_monthly_sales AS
        SELECT
            c.region,
            TO_CHAR(o.order_date, 'Mon') AS month_name,
            SUM(o.sales) AS total_sales
        FROM orders AS o
        JOIN customers AS c
        ON o.customer_id = c.customer_id
        GROUP BY c.region, month_name;
    """
    ,
    "overall_customers_performance":"""
        DROP VIEW IF EXISTS overall_customers_performance;
        CREATE OR REPLACE VIEW overall_customers_performance AS
        SELECT 
            ROUND(SUM(o.sales)/COUNT(DISTINCT o.customer_id)) AS avg_sales_per_customer,
            ROUND(SUM(o.profit)/COUNT(DISTINCT o.customer_id)) AS avg_profit_per_customer,
            ROUND(COUNT(DISTINCT order_id)/COUNT(DISTINCT customer_id)) AS avg_orders_per_customer,
            ROUND(SUM(o.quantity)/COUNT(DISTINCT o.customer_id)) AS avg_quantity_per_customer
        FROM orders AS o;
    """
    ,
    "avg_discount_per_order_per_customer":"""
        DROP VIEW IF EXISTS avg_discount_per_order_per_customer;
        CREATE OR REPLACE VIEW avg_discount_per_order_per_customer AS
        SELECT
            ROUND(AVG(customer_avg), 2) AS avg_discount_per_customer
        FROM (
            SELECT customer_id, AVG(discount) AS customer_avg
            FROM orders
            GROUP BY customer_id
        ) AS sub;
    """
    ,
    "category_wise_monthly_sales_and_profit":"""
        DROP VIEW IF EXISTS category_wise_monthly_sales_and_profit;
        CREATE OR REPLACE VIEW category_wise_monthly_sales_and_profit AS
        SELECT
            p.category,
            TO_CHAR(o.order_date, 'Mon') AS month,
            SUM(o.sales) AS total_sales,
            SUM(o.profit) AS total_profit
        FROM orders AS o
        JOIN products AS p
        ON o.product_id = p.product_id
        GROUP BY p.category, month;
    """
    ,
    "sub_category_wise_sales_and_profit":"""
        DROP VIEW IF EXISTS sub_category_wise_sales_and_profit;
        CREATE OR REPLACE VIEW sub_category_wise_sales_and_profit AS
        SELECT
            p.sub_category,
            SUM(o.sales) AS total_sales,
            SUM(o.profit) AS total_profit
        FROM orders AS o
        JOIN products AS p
        ON o.product_id = p.product_id
        GROUP BY p.sub_category;
    """
    ,
    "category_wise_sales_profit_and_orders":"""
        DROP VIEW IF EXISTS category_wise_sales_profit_and_orders;
        CREATE OR REPLACE VIEW category_wise_sales_profit_and_orders AS
        SELECT
            p.category,
            SUM(o.sales) AS total_sales,
            SUM(o.profit) AS total_profit,
            COUNT(DISTINCT o.order_id) AS total_orders
        FROM orders AS o
        JOIN products AS p
        ON o.product_id = p.product_id
        GROUP BY p.category;
    """
    ,
    "state_wise_most_purchased_sub_category":"""
        DROP VIEW IF EXISTS state_wise_most_purchased_sub_category;
        CREATE OR REPLACE VIEW state_wise_most_purchased_sub_category AS
        SELECT
            c.state,
            p.sub_category, 
            SUM(o.quantity) AS quantity_sold,
            RANK() OVER (PARTITION BY c.state ORDER BY SUM(o.quantity) DESC) AS sub_category_rank
        FROM orders AS o
        JOIN products AS p
        ON o.product_id = p.product_id
        JOIN customers AS c
        ON o.customer_id = c.customer_id
        GROUP BY c.state, p.sub_category;
    """
}

# Creating Views in Neon PostgreSQL Database
with engine.connect() as conn:
    for view_name, sql_query in sql_views.items():
        conn.execute(text(sql_query))
    conn.commit()
logging.info("Views created successfully in Neon PostgreSQL Database")
```
</details>

#### What it does?
- **Database Connection**
  - Connects to the database securely using environment variables.
- **Define SQL Views**
  - Creates multiple SQL views to extract key business insights.
- **Execute & Commit**
  - Executes each `CREATE OR REPLACE VIEW` statement and commit changes.
- **Logging**
  - Stores execution logs in `logs/create_views.log`.

---

#### 3. `generate_data.py`
- This script keeps the database updated with new transaction data for schedule data refresh in Power BI.

<details>
<summary>Click Here to view the entire Script</summary>
<br>
  
```python
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
```
</details>

#### What it does?
- **Generate Random Data**
  - Uses custom utility functions to create synthetic customers and orders data.
- **Data Cleaning**
  - Removes duplicates and optimizes datatypes before uploading to Neon Database.
- **Append Unique Data**
  - Inserts only new records into the database, avoiding duplicates.
- **Logging**
  - Saves process logs in `logs/generate_data.log`.

Once this cycle is complete, the process repeats automatically :

```
generate_data.py → create_views.py → Power BI Refresh → New Insights
```

This ensures that the Power BI Dashboard always display the latest insights automatically.

<img title="ETL Pipeline" src="https://github.com/user-attachments/assets/a4a54ad4-eace-4f3c-b90d-53bb0983de00">

<hr>

## Results & Insights
It highlights the key outcomes and insights generated from the project's ETL pipeline and Power BI dashboards.

### Pipeline Performance
It includes pipeline performance metrics like runtime, automation frequency and reliability.

#### 1. Data Loading Overview
| **Parameter**                  | **Value**                         |
| :----------------------------- | :-------------------------------- |
| **Dataset Size**               | ~50,000 rows                      |
| **Tables Used**                | `customers`, `orders`, `products` |
| **Avg. Daily Inserts**         | ~100 records                      |
| **Database**                   | Neon PostgreSQL (Cloud-hosted)    |

> [!NOTE]
> This setup mimics real-time superstore sales with daily updates to orders and customers tables.

#### 2. Automation & Scheduling
| **Attribute**           | **Details**                                         |
| ----------------------- | :-------------------------------------------------- |
| **Automation Tool**     | GitHub Actions                                      |
| **Execution Frequency** | Daily                                               |
| **Scheduled Time**      | 10:00 AM                                            |
| **Trigger Type**        | `cron` (automated) and `workflow_dispatch` (manual) |
| **Runner Environment**  | `ubuntu-latest` (Linux VM)                          |

> [!NOTE]
> This setup ensures that the latest data is always available for Power BI dashboards, no manual refresh required.

<img title="Automation & Scheduling" src="https://github.com/user-attachments/assets/0708e2eb-6511-4962-bb36-2c77752a8f47">

#### 3. Runtime Performance
| **Workflow Step**                    | **Description**                                   | **Runtime (sec)** |
| :----------------------------------- | :------------------------------------------------ | :--------------- |
| **Set up job**                       | Initializes GitHub Actions Environment            |                1s |
| **Checkout Repository**              | Pulls repository code into the Runner             |                1s |
| **Set up Python**                    | Installs Python Environment (v3.10)               |                0s |
| **Install dependencies**             | Installs libraries from `requirements.txt`        |               22s |
| **Run ETL Script**                   | Extracts, transforms & loads data into PostgreSQL |                3s |
| **Run Generate Data Script**         | Generates new synthetic customers & orders data   |                4s |
| **Run Views Script**                 | Creates/refreshes analytical SQL Views            |                3s |
| **Run Export Views Script**          | Exports SQL views as CSV files                    |                2s |
| **Upload Exported CSV as Artifacts** | Uploads exported CSVs to GitHub Actions Artifacts |                2s |
| **Commit and Push CSVs**             | Commits CSV files to the Repository               |                0s |
| **Upload Logs as Artifacts**         | Uploads log files for debugging and tracking      |                1s |
| **Post Setup/Cleanup Steps**         | Cleans Environment Post-run                       |              0-2s |

> [!NOTE]
> Total Runtime : ~45 to ~50 seconds per pipeline run
> 
> Scheduling : The workflow is scheduled via cron (30 4 * * *), meaning it runs daily at 10:00 AM.
> 
> The ETL pipeline runs within a minute, automatically refreshing the dashboard data daily without manual effort.

<img title="Runtime Performance" src="https://github.com/user-attachments/assets/8dcf62d2-9bb8-4ef4-b0c4-292115dc5079">

#### 4. Error Handling and Logging
| **Aspect**           | **Implementation Details**                                            |
| :------------------- | :-------------------------------------------------------------------- |
| **Error Tracking**   | `try-except` blocks in each Script                                    |
| **Log Files**        | `etl.log`, `generate_data.log`, `create_views.log`                    |
| **Log Storage**      | Uploaded as GitHub Actions Artifacts                                  |
| **Security**         | All credentials stored in GitHub Secrets (`DB_USER`, `DB_PASS`, etc.) |

> [!TIP]
>
> Automated logging and secret handling remove the need for manual checks and ensure a smooth workflow run.

#### 5. Reliability and Stability
| **Metric**                  | **Value**              | **Remarks**                                |
| :-------------------------- | :--------------------- | :----------------------------------------- |
| **Total Runtime**           | ~40 seconds            | Fast for Daily Automated ETL               |
| **Success Rate**            | 100% (last 8 runs)     | Verified via GitHub Actions Workflow Panel |
| **Avg. Records Inserted**   | ~100 rows/day          | Lightweight Daily Updates                  |
| **Resource Utilization**    | Low CPU/Memory         | Efficient for Cloud Runners                |

> [!IMPORTANT]
>
> The pipeline runs fully unattended, providing consistent daily data updates and automatic Power BI refreshes.

<img title="Reliability and Stability" src="https://github.com/user-attachments/assets/20dbee9f-b68e-4a1a-b512-4ed7f443d587">

<hr>

### Dashboard Metrics

#### 1. Shipping Mode Performance
| **Shipping Mode** | **Total Sales (₹)** | **% of Total Sales** | **Total Profit (₹)** | **% of Total Profit** | **Profit Margin** |
| ----------------- | ------------------: | -------------------: | -------------------: | --------------------: | ----------------: |
| Standard Class    |           5,099,197 |            **59.7%** |              897,360 |             **59.8%** |         **17.6%** |
| Second Class      |           1,650,059 |            **19.3%** |              292,629 |             **19.5%** |         **17.7%** |
| First Class       |           1,343,959 |            **15.7%** |              230,784 |             **15.4%** |         **17.2%** |
| Same Day          |             440,836 |             **5.2%** |               78,981 |              **5.3%** |         **17.9%** |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- Standard Class drives ~60% of total sales (~₹5.1M) and profit (~₹897K), making it the most profitable and preferred shipping mode.
- Second Class accounts for ~19% of total sales (~₹1.65M) and profit (~₹293K), showing moderate usage.
- First Class brings in ~16% of revenue (~₹1.34M) and ~15% of profit (~₹231K), preferred for premium deliveries.
- Same Day shipping makes up only ~5% of sales (~₹0.44M) and profit (~₹79K), least used but fastest option.
- Profit margins stay steady across modes (~17–18%), indicating balanced logistics and pricing control.

</details>

#### 2. Customer Segment Performance
| **Segment** | **Total Sales (₹)** | **% of Total Sales** | **Total Profit (₹)** | **% of Total Profit** | **Profit Margin** |
| ----------- | ------------------: | -------------------: | -------------------: | --------------------: | ----------------: |
| Consumer    |           4,263,570 |            **49.9%** |              757,416 |             **50.5%** |         **17.8%** |
| Corporate   |           2,742,160 |            **32.1%** |              475,855 |             **31.7%** |         **17.4%** |
| Home Office |           1,528,321 |            **17.9%** |              266,483 |             **17.8%** |         **17.4%** |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- Consumer Segment is the main revenue driver, generating ~50% of total sales (~₹4.26M) and profit (~₹757K).
- Corporate Segment provides ~32% of total sales (~₹2.74M) and profit (~₹476K), showing steady B2B growth.
- Home Office Segment delivers ~18% of revenue (~₹1.53M) and profit (~₹266K), smaller but reliable segment.
- Profit margins are consistent across all segments (~17–18%), highlighting balanced pricing and operational control.
- Potential growth opportunity lies in expanding Home Office sales through targeted marketing or discount strategies.

</details>

#### 3. Monthly Sales & Profit Performance
| **Month** | **Total Sales (₹)** | **% of Total Sales** | **Total Profit (₹)** | **% of Total Profit** | **Profit Margin** |
| :-------- | ------------------: | -------------------: | -------------------: | --------------------: | ----------------: |
| Jan       |             731,193 |             **8.6%** |              130,941 |              **8.7%** |         **17.9%** |
| Feb       |             666,688 |             **7.8%** |              119,076 |              **7.9%** |         **17.9%** |
| Mar       |             765,028 |             **9.0%** |              131,184 |              **8.7%** |         **17.1%** |
| Apr       |             705,858 |             **8.3%** |              123,398 |              **8.2%** |         **17.5%** |
| May       |             734,328 |             **8.6%** |              131,017 |              **8.7%** |         **17.8%** |
| Jun       |             639,560 |             **7.5%** |              115,182 |              **7.7%** |         **18.0%** |
| Jul       |             653,572 |             **7.7%** |              118,166 |              **7.9%** |         **18.1%** |
| Aug       |             695,246 |             **8.2%** |              124,125 |              **8.3%** |         **17.8%** |
| Sep       |             646,965 |             **7.6%** |              110,729 |              **7.4%** |         **17.1%** |
| Oct       |             789,443 |             **9.2%** |              137,346 |              **9.2%** |         **17.4%** |
| Nov       |             712,799 |             **8.3%** |              120,187 |              **8.0%** |         **16.9%** |
| Dec       |             793,371 |             **9.3%** |              138,403 |              **9.2%** |         **17.4%** |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- Q4 (Oct–Dec) drives ~27% of yearly sales and profit, making it the best-performing quarter, great time for offers and stock planning.
- Profit margins stay steady (17–18%) throughout the year, showing good cost control and consistent pricing.
- March and December are strong sales months, ideal for marketing campaigns and promotions.
- Overall, sales and profits stay balanced across months, reflecting smooth and stable business performance.

</details>

#### 4. Regional Performance Insights
| **Region**  | **Total Sales (₹)** | **% of Total Sales** | **Total Profit (₹)** | **% of Total Profit** |
| ----------- | ------------------: | -------------------: | -------------------: | --------------------: |
| **West**    |           2,484,870 |                 ~29% |              440,814 |                  ~29% |
| **East**    |           2,456,014 |                 ~29% |              427,689 |                  ~28% |
| **Central** |           1,986,280 |                 ~23% |              352,400 |                  ~23% |
| **South**   |           1,606,887 |                 ~19% |              278,851 |                  ~19% |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- West Region accounts for ~29% of total revenue (~₹2.48M) and profit (~₹441K), making it the top-performing region.
- East Region generates ~29% of total sales (~₹2.46M) and ~28% of total profit (~₹428K), showing strong and balanced growth.
- Central Region adds ~23% of total revenue (~₹1.99M) and profit (~₹352K), indicating steady mid-range performance.
- South Region contributes ~19% of total revenue (~₹1.61M) and profit (~₹279K), lowest among all regions, suggesting scope for sales expansion.
- Overall, profit margins remain consistent (~17–18%), showing healthy operations across all regions.

</details>

#### 5. Sub-Category Performance
| **Sub-Category**          | **Total Sales (₹)** | **% of Total Sales** | **Total Profit (₹)** | **% of Total Profit** |
| ------------------------- | ------------------: | -------------------: | -------------------: | --------------------: |
| Paper                     |           1,260,080 |                 ~15% |              223,147 |                  ~15% |
| Binders                   |             943,376 |                 ~11% |              162,084 |                  ~11% |
| Phones                    |             879,975 |                 ~10% |              153,018 |                  ~10% |
| Furnishings               |             836,955 |                 ~10% |              151,623 |                  ~10% |
| Art                       |             697,965 |                  ~8% |              120,843 |                   ~8% |
| Storage                   |             651,268 |                  ~8% |              112,673 |                   ~8% |
| Accessories               |             643,638 |                  ~7% |              112,007 |                   ~7% |
| Appliances                |             495,030 |                  ~6% |               90,786 |                   ~6% |
| Others (8 sub-categories) |                   — |                    — |                    — |                     — |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- Paper leads in products sub-categories, driving ~15% of total sales (~₹1.26M) and profit (~₹223K).
- Binders, Phones and Furnishings together contribute ~31% of revenue (~₹2.66M), forming the core profit cluster.
- Art, Storage and Accessories act as strong mid-tier performers, keeping profit margins steady around 17–18%.
- Lower-performing items like Copiers and Supplies together make <5% of total profit, indicating limited business impact.
- Expanding high-value categories (Paper, Phones, Furnishings) can help grow revenue without affecting margin stability.

</details>

#### 6. State-wise Sales Performance (Top 5 States)
| **State**    | **Total Sales (₹)** | **Total Customers** | **% of Total Sales** |
| :----------- | ------------------: | ------------------: | -------------------: |
| California   |           1,792,545 |                 188 |            **21.4%** |
| New York     |             924,728 |                 104 |            **11.0%** |
| Texas        |             903,046 |                  92 |            **10.8%** |
| Pennsylvania |             472,741 |                  45 |             **5.6%** |
| Ohio         |             451,732 |                  51 |             **5.4%** |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- California dominates with ~21% of total sales (~1.8M) and the largest customer base (188), store's strongest regional market.
- New York and Texas together contribute ~22% of total sales (~1.8M), showing strong demand in high-density regions.
- Top 5 states account for ~54% of total revenue (~4.5M), showing sales concentration in key urban and economically active areas.

</details>

<hr>

## How To

### 1. How to create a free PostgreSQL Database on Neon?
- Open [Neon](https://neon.com/) in your browser
<img title="Neon" src="https://github.com/user-attachments/assets/4a5b08e6-290b-4fb7-9820-3afd0b599e03">
&nbsp;

- Login with GitHub/Google/Microsoft
<img title="Login" src="https://github.com/user-attachments/assets/657ccfaa-3c26-492a-9a5f-537f773c763e">
&nbsp;

- Create a New Project
<img title="New Project" src="https://github.com/user-attachments/assets/dd9271db-e210-4ed0-aeaf-9ce337d06474">
&nbsp;

- Fill in the Project Details
<img title="Project Details" src="https://github.com/user-attachments/assets/fb2cf22c-6af9-4598-b207-351e4cd66f45">
&nbsp;

- New PostgreSQL Project is created in Neon Console
<img title="Neon Console" src="https://github.com/user-attachments/assets/eb7f3663-9cda-49ae-94ff-0c866a1d9898">

<hr>

### 2. How to connect Neon Database with Python via SQLAlchemy?
- Open the newly created Project and Click Connect
<img title="Project Console" src="https://github.com/user-attachments/assets/17a7a0bb-ce46-4496-9f35-b6cce062c791">
&nbsp;

- Get the Connection String of your Database
<img title="Connection String" src="https://github.com/user-attachments/assets/bd0d5636-38e3-4b3f-abd8-0bc0589f3577">
<p align="center">Image taken from Neon Documentation</p>

- Understand the Connection String
```
postgresql://alex:AbC123dEf@ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech/dbname?sslmode=require&channel_binding=require
             ^    ^         ^                         ^                              ^
       role -|    |         |- hostname               |- pooler option               |- database
                  |
                  |- password
```
You can use this to configure your database connection.

You can place the connection details in an `.env` file for secure access.
```bash
# .env file
DB_HOST=ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech
DB_NAME=dbname
DB_USER=alex
DB_PASS=AbC123dEf
```
After placing connection details in an `.env` file, you can read it via Python using SQLAlchemy.
```python
# Importing Libraries
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Loading Environment File
load_dotenv()

# Loading Database Credentials from Environment File
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Creating SQLAlchemy Engine to upload the Data to Neon PostgreSQL Database
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?sslmode=require&channel_binding=require", pool_pre_ping=True)
```

<hr>

### 3. How to connect Power BI with Neon Database for live data refresh?
- Launch Power BI Desktop on your computer.
- Start with a new, empty report.
- In the Home tab, click on Get Data then More...
- This opens a list of all available data connectors.
<img title="Getting Data in Power BI" src="https://github.com/user-attachments/assets/a4c5d051-f4b7-4e45-a77c-06e4f13f23af">
&nbsp;

- In the list of connectors, scroll down and find PostgreSQL database.
- Select it and click Connect.
- This tells Power BI that you want to connect to a PostgreSQL database (like Neon).
<img title="Selecting PostgreSQL Database" src="https://github.com/user-attachments/assets/f10c2f7a-40c8-4eb1-b475-0217e09b5c3d">
&nbsp;

- Power BI will prompt you for connection information.
- You'll need the Host and Database name from your Neon connection string.
```bash
DB_HOST=ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech
DB_NAME=dbname
```
- Fill in Host and Database name and Click ok.
- This step connects Power BI directly to your Neon PostgreSQL Database.
<img title="Connecting to PostgreSQL Database" src="https://github.com/user-attachments/assets/65f2b1c5-7759-41c4-8243-6f4b4879358e">
&nbsp;

<details>
  <summary>Click Here to know more about Connection Modes in Power BI</summary>
  &nbsp;
  
  - **Import Mode**
    - Power BI copies the data from your source system into its internal, highly optimized storage engine.
    - That means all the data is loaded and stored within the .pbix file (or in Power BI Service once published).
    - All reports, visuals and calculations run on this cached data, not on the live data source.
  - **DirectQuery Mode**
    - In DirectQuery Mode, Power BI does not import or stores data.
    - Instead, it sends real-time queries to the data source every time a user interacts with a report.
    - For live data refresh, choose DirectQuery Mode.
    
</details>
&nbsp;

- Once connected, you'll see a list of all available tables and views from your Neon database.
- Select the ones you want to use in your report and Click Load.
<img title="Loading Required Data" src="https://github.com/user-attachments/assets/5fd32b9e-826c-4351-95f0-34ff712460fa">
&nbsp;

- Power BI is now connected to your Neon PostgreSQL database.
- It can refresh live data automatically using DirectQuery Mode.

<hr>

### 4. How to configure GitHub Actions?
- Open [GitHub](https://github.com/) in your browser.
- Navigate to the repository that contains your workflow file (`.github/workflows/etl_pipeline.yml`).
- This is where your ETL (Extract-Transform-Load) automation is defined.
- On your repository's main page click Settings from the top navigation bar.
<img title="Repository's Settings Page" src="https://github.com/user-attachments/assets/55764fbc-2ec2-43e1-befa-3895c2c05d84">
&nbsp;

- In the left-hand sidebar of the Settings page, scroll down to find Secrets and Variables.
- Click Actions to open the section where you can manage secrets used in GitHub Actions workflows.
<img title="Secrets and Variables Section" src="https://github.com/user-attachments/assets/72378122-db0e-446d-992f-3e6a6acfe1eb">
&nbsp;

- In the Actions section, click the New repository secret button.
- This opens a window to add a new secret key-value pair.
<img title="New Repository Secret" src="https://github.com/user-attachments/assets/3bbf4e6b-a834-40c9-af7e-316f72921ebf">
&nbsp;

- Open your local `.env` file (this file contains environment variables like API keys, tokens, etc).
- For each variable, copy :
  - Name : the key (`DATABASE_URL`)
  - Value : the corresponding value (`postgres://user:pass@host/db`)
- Paste these into the GitHub form fields :
  - Name → enter the variable name
  - Secret → enter the variable value
<img title="Add New Secret" src="https://github.com/user-attachments/assets/1e701c66-ec31-4c0e-8fb8-21220b4f551f">
&nbsp;

- Your secrets are now securely stored in the repository.
- You can use them inside your workflow `.github/workflows/etl_pipeline.yml` like this :
```bash
DB_USER: ${{ secrets.DB_USER }}
DB_PASS: ${{ secrets.DB_PASS }}
```

<hr>

## GitHub Actions
- The GitHub Actions workflow automates the entire ETL pipeline, without any manual effort.
- It runs every day at 10:00 AM and ensures that the latest data are always updated.

<details>
  <summary>Click Here to view the entire Script</summary>
  <br>

```yaml
# ---------------- Name of the Workflow ----------------
name: ETL Pipeline Automation

# ---------------- When should it run? ----------------
on:
  schedule:
    - cron: "30 4 * * *"   # Run the workflow daily at 10:00 AM
  workflow_dispatch:       # Allow manual run from GitHub UI

# ---------------- Authority to update Repository ----------------
permissions:
  contents: write

# ---------------- Set of steps to run ----------------
jobs:
  data-pipeline:
    runs-on: ubuntu-latest   # Use a Linux VM for the Job

    env: # Shared env variables available to all scripts
      DB_USER: ${{ secrets.DB_USER }}
      DB_PASS: ${{ secrets.DB_PASS }}
      DB_HOST: ${{ secrets.DB_HOST }}
      DB_NAME: ${{ secrets.DB_NAME }}

    steps:
      # ---------------- Step 1 : Checkout Code ----------------
      # This pulls your repository into the GitHub Runner VM
      - name: Checkout repository
        uses: actions/checkout@v3   
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      # ---------------- Step 2 : Set up Python ----------------
      # Installs Python 3.13.0 so GitHub can run your Scripts
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.13.0"

      # ---------------- Step 3 : Install Dependencies ----------------
      # Installs all Python libraries listed in requirements.txt
      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      # ---------------- Step 4 : Run ETL Script ----------------
      # Runs your etl.py script to setup database
      - name: Run ETL Script
        run: |
          python scripts/etl.py

      # ---------------- Step 5 : Generate New Data and Append ----------------
      # Creates new data of customers and orders to ingest in the Neon Database
      - name: Run Generate Data Script
        run: |
          python scripts/generate_data.py

      # ---------------- Step 6 : Run Views Script ----------------
      # Creates or refreshes SQL views in Neon Database
      - name: Run Views Script
        run: |
          python scripts/create_views.py

      # ---------------- Step 7 : Export Views to CSV ----------------
      # Exports SQL views as CSVs inside the `views` folder
      - name: Run Export Views Script
        run: |
          python scripts/export_views.py

      # ---------------- Step 8 : Save CSVs as Artifacts ----------------
      # Stores CSVs in the workflow run, downloadable from GitHub
      - name: Upload Exported CSV as Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: exported-views
          path: views/*.csv

      # ---------------- Step 9 : Commit CSVs to Repository ----------------
      # Updates repository with the latest CSVs so they are visible directly
      - name: Commit and Push CSVs
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add -A views/
          git commit -m "chore: update csv export from views on $(date -u +'%Y-%m-%d')" -m "[skip ci]" || echo "No changes to commit"
          git push

      # ---------------- Step 10 : Save Logs as Artifacts ----------------
      # Stores Logs in the workflow run, downloadable from GitHub
      - name: Upload Logs as Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: etl-logs
          path: logs/*.log
```
</details>

### Workflow Overview

### 1. Name of the Workflow
```yaml
name: ETL Pipeline Automation
```
- This gives your workflow a name.
- You'll see **ETL Pipeline Automation** in the GitHub Actions tab whenever it runs.

<hr>

### 2. When should it run?
```yaml
on:
  schedule:
    - cron: "30 4 * * *"
  workflow_dispatch:
```
- **schedule :** Runs every day at 10:00 AM automatically (based on the cron expression).
- **workflow_dispatch :** Allows you to manually trigger the workflow from the GitHub Actions tab.

<hr>

### 3. Permissions
```yaml
permissions:
  contents: write
```
- This gives your workflow permission to update files in your repository.

<hr>

### 4. Jobs
```yaml
jobs:
  data-pipeline:
    runs-on: ubuntu-latest

    env: # Shared env variables available to all scripts
        DB_USER: ${{ secrets.DB_USER }}
        DB_PASS: ${{ secrets.DB_PASS }}
        DB_HOST: ${{ secrets.DB_HOST }}
        DB_NAME: ${{ secrets.DB_NAME }}
```
- **jobs :** Defines what tasks (jobs) the workflow will perform.
- **data-pipeline :** The name of your main job.
- **runs-on :** Tells GitHub to use a Linux virtual machine (Ubuntu) to run your scripts.

<hr>

### 5. Main Workflow Tasks
- Each step defines a task GitHub Actions performs in sequence :

#### Step 1 : Checkout Repository
```yaml
- name: Checkout repository
  uses: actions/checkout@v3
```
- This pulls your GitHub Repository files into the VM so the workflow can access your scripts, data and folders.

#### Step 2 : Set Up Python
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: "3.13.0"
```
- Installs Python 3.13.0 on the VM, the version your ETL scripts uses.

#### Step 3 : Install Dependencies
```yaml
- name: Install dependencies
  run: |
    pip install -r requirements.txt
```
- Installs all the Python libraries listed in `requirements.txt` file.

#### Step 4 : Run ETL Script
```yaml
- name: Run ETL Script
  run: |
    python scripts/etl.py
```
- Runs your `etl.py` script which :
  - Cleans and transforms raw data.
  - Loads it into the Neon PostgreSQL Database.
  - Uses GitHub Secrets to securely access database credentials (so they're never expose in code).

#### Step 5 : Generate New Data
```yaml
- name: Run Generate Data Script
  run: |
    python scripts/generate_data.py
```
- Runs `generate_data.py` which creates and updates new random data to simulate daily data refresh.

#### Step 6 : Create SQL Views
```yaml
- name: Run Views Script
  run: |
    python scripts/create_views.py
```
- Executes `create_views.py` which builds or refreshes SQL Views in the database.
- These views are directly used by Power BI dashboards.

#### Step 7 : Export Views to CSV
```yaml
- name: Run Export Views Script
  run: |
    python scripts/export_views.py
```
- Exports your latest SQL Views as CSV files into the `views/` folder.
- These can be used for testing, sharing or backups.

#### Step 8 : Upload CSVs as Artifacts
```yaml
- name: Upload Exported CSV as Artifacts
  uses: actions/upload-artifact@v4
  with:
    name: exported-views
    path: views/*.csv
```
- Stores your exported CSV files as downloadable artifacts in GitHub Actions.
- This means you can view and download them directly from the workflow run.

#### Step 9 : Commit CSVs to Repository
```yaml
- name: Commit and Push CSVs
  run: |
    git config --global user.name "github-actions[bot]"
    git config --global user.email "github-actions[bot]@users.noreply.github.com"
    git add -A views/
    git commit -m "chore: update csv export from views on $(date -u +'%Y-%m-%d')" -m "[skip ci]" || echo "No changes to commit"
    git push
```
- Commits updated CSVs to the repository, so anyone visiting the GitHub project can see the latest analysis.

#### Step 10 : Upload Logs
```yaml
- name: Upload Logs as Artifacts
  uses: actions/upload-artifact@v4
  with:
    name: etl-logs
    path: logs/*.log
```
- Uploads all log files as artifacts, useful for debugging or checking workflow results.

<hr>

## Power BI Dashboard
- The Power BI Dashboard is designed to turn data into insights with a clean and interactive interface.
- It connects directly to the database, ensuring that the dashboard always reflects the most recent data.
- The dashboard consists of four main pages : Home, Overview, Customers and Products.
- All are connected through page navigation and drill-through features.

### 1. Home Page
- This page serves as a entry point and navigation hub for the dashboard.
  
### Key Components
- **Navigation Buttons**
    - Interactive page navigator that links to Overview, Customers and Products page.
- **Dashboard Branding**
    - Displays a themed background image for a mordern look.
- **User Experience**
    - Designed for clarity and smooth navigation so users can access key insights in one click.

<img title="Home" src="https://github.com/user-attachments/assets/a15b8d82-a886-492f-af40-3d8d25ea22ad">

<hr>

### 2. Overview
- This page focuses on a high-level summary of overall business performance like orders, customers and products.
  
### Key Components
- **Multi-Row Cards - Key Business Metrics**
  - Data View : `overall_sales_performance`
  - Purpose : Gives an instant snapshot of business performance at a glance.
- **Filled Map - Sales by State**
  - Data View : `state_wise_sales_and_customer_base`
  - Purpose : Visualizes geographic sales distribution, showing which states drive the most revenue.
- **Donut Chart - Shipping Performance**
  - Data View : `shipping_performance`
  - Purpose : Compares sales and profit by shipping mode, helping identify cost-effective delivery methods.
- **Stacked Bar Chart - Segment-wise Sales & Profit**
  - Data View : `segment_wise_sales_and_profit`
  - Purpose : Shows sales and profit across customer segments.
- **Bar Chart - Top Customers by Sales**
  - Data View : `top_customers_by_sales`
  - Purpose : Highlights the top-performing customers, helping identify key contributors to revenue.
- **Filled Area Chart - Monthly Sales & Profit Trend**
  - Data View : `month_wise_sales_and_profit`
  - Purpose : Displays the month-wise trend of sales and profit to track seasonal performance and growth.

<img title="Overview" src="https://github.com/user-attachments/assets/5584efcc-c10a-40c0-b18c-18e907ea64df">

<hr>

### 3. Customers
- This page focuses on understanding customer behavior, performance and geographic distribution.

### Key Components
- **Multi-Row Card - Customer Performance Summary**
  - Data View : `overall_customers_performance`
  - Purpose : Provides an overview of how much each customer contributes on average to sales and profits.
- **Filled Map - State-wise Customer Base**
  - Data View : `state_wise_sales_and_customer_base`
  - Purpose : Visualizes customer distribution across states, helping identify regions with the highest customers.
- **Stacked Column Chart - Region-wise Monthly Sales**
  - Data View : `region_wise_monthly_sales`
  - Purpose : Tracks how sales vary across different regions and months, helping spot seasonal/regional trends.
- **Stacked Column Chart - Region-wise Sales & Profit**
  - Data View : `region_wise_sales_and_profit`
  - Purpose : Compares overall sales and profit across regions to identify high and low-performing areas.
- **Dual Area Charts - Segment-wise Monthly Sales & Profit**
  - Data View : `segment_wise_monthly_sales_and_profit`
  - Purpose : Visualizes monthly trends for sales and profit across different customer segments.

<img title="Customers" src="https://github.com/user-attachments/assets/fd6944a8-59af-43e3-a78d-18b1f92386a4">

<hr>

### 4. Products
- This page focuses on analyzing product performance, category trends and geographical purchasing behavior.
  
### Key Components
- **Filled Map - State-wise Most Purchased Sub-category**
  - Data View : `state_wise_most_purchased_sub_category`
  - Purpose : Helps identify regional preferences and popular product types across states.
- **Treemap - Category-wise Orders**
  - Data View : `category_wise_sales_profit_and_orders`
  - Purpose : Visualizes which categories dominate in orders and their contribution to total business.
- **Stacked Column Chart - Category-wise Sales and Profit**
  - Data View : `category_wise_sales_profit_and_orders`
  - Purpose : Visualizes which categories dominate in sales and profit and their contribution to total business.
- **Stacked Bar Chart - Sub-category-wise Sales & Profit**
  - Data View : `sub_category_wise_sales_and_profit`
  - Purpose : Compares performance of business across all product sub-categories.
- **Dual Ribbon Charts - Category-wise Monthly Sales & Profit Trends**
  - Data View : `category_wise_monthly_sales_and_profit`
  - Purpose : Tracks how each product category performs across months, identify growth patterns.
- **Gauge Chart - Profit Margin Tracker**
  - Data Source
    - Calculated using total sales and profit.
  - Configuration 
    - Minimum : 0%
    - Maximum : 40%
    - Target : 30%
  - Purpose
    - Tracks the current profit margin against the target (30%).
  - Color Logic
    - Red (0-10%) → Low profit margin, needs improvement.
    - Yellow (10-20%) → Moderate margin, progressing toward target.
    - Green (20-40%) → Healthy profit margin, close to or exceeding target.

<img title="Products" src="https://github.com/user-attachments/assets/ccf6f4ad-4498-4d0b-9284-0c60feeb19c2">

<hr>

## Folder Structure

```
├── .github/
│   └── workflows/
│       └── etl_pipeline.yml         # GitHub Actions Workflow File
│
├── scripts/
│   ├── etl.py                       # Extract, Transform & Load Initial Data
│   ├── generate_data.py             # Script to generate and append New Random Data
│   ├── create_views.py              # Script to create or refresh SQL Views in PostgreSQL Database
│   └── export_views.py              # Script to export SQL Views results as CSV File
│
├── data/
│   └── sales_data.csv               # Original Raw Dataset
|
├── database/
│   ├── schema.sql                   # Database Schema
│   └── views.sql                    # SQL Views
│
├── views/
│   ├── shipping_performance.csv     # Exported Views as CSV Files
│   └── ...
|
├── images/
│   ├── banner.png                   # Banner for Home Page
│   ├── shopping-cart.png            # Multi-Row Card Image
|   └── ...
│
├── utils/                           # Reusable Python Functions (utils Package)
│   ├── __init__.py
│   ├── data_preprocessor.py
│   ├── generate_random_data.py
|   └── ...
|
├── Dashboard.pbix                   # Power BI Dashboard File
├── .gitignore                       # All files and folders ignored by Git
├── requirements.txt                 # List of required libraries for the Project
├── LICENSE                          # License specifying permissions and usage rights
└── README.md                        # Detailed documentation of the Project
```

<hr>

## License

This project is licensed under the [MIT License](LICENSE). You are free to use and modify the code as needed.

<div align='left'>
  
**[`^        Scroll to Top       ^`](#dashly--live-sales-dashboard)**

</div>
