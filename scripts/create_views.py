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