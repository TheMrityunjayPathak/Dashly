<h3 align="center" id="top">Dashly : Live Sales Dashboard</h3>

<div align="center">

[![Power BI](https://img.shields.io/badge/Live-Dashboard-F2C811?style=flat&logo=codeforces&logoColor=white)](https://app.powerbi.com/view?r=eyJrIjoiZTgxODBhYmMtYjc1Zi00YjVkLWIyZDItZDYxY2RjZmIwNGY5IiwidCI6ImZhYjAyYzVkLTYxYjYtNGIxMi05ZTY2LTdhMDhkOWY0ZmNjMSJ9&pageName=5b9aaf645951a59cacdc)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-08872B?style=flat&logo=githubactions&logoColor=white)](https://github.com/themrityunjaypathak/Dashly/actions/workflows/etl_pipeline.yaml)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-4420C7?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat&logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-689DC8?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Git](https://img.shields.io/badge/Git-F05133?style=flat&logo=git&logoColor=white)](https://git-scm.com/)

</div>

<a href="https://app.powerbi.com/view?r=eyJrIjoiZTgxODBhYmMtYjc1Zi00YjVkLWIyZDItZDYxY2RjZmIwNGY5IiwidCI6ImZhYjAyYzVkLTYxYjYtNGIxMi05ZTY2LTdhMDhkOWY0ZmNjMSJ9&pageName=5b9aaf645951a59cacdc">
  <img title="Dashly" src="https://github.com/user-attachments/assets/519b5a9e-7e56-4ab4-a9fb-8dfd5b18063c">
</a>

## Table of Contents
- [Problem Statement](#problem-statement)
- [Overview](#overview)
- [Workflow](#workflow)
- [ER Diagram](#er-diagram)
- [Database Schema](#database-schema)
- [SQL Views](#sql-views)
- [Setup](#setup)
- [ETL Pipeline](#etl-pipeline)
- [Results & Insights](#results--insights)
- [Impact](#impact)
- [Key Insights](#key-insights)
- [How To](#how-to)
- [GitHub Actions](#github-actions)
- [Power BI Dashboard](#power-bi-dashboard)
- [Folder Structure](#folder-structure)
- [License](#license)

<hr>

## Problem Statement
- Quick Buy is a superstore chain operating across the United States.
- Performance tracking relied heavily on manual spreadsheets and ad-hoc SQL queries.
- This slowed down decision-making and made it harder to spot growth opportunities.
- The goal was to automate the data workflow and deliver an up-to-date sales dashboard for informed decisions.

<hr>

## Overview
- Designed an ETL pipeline with Python and SQLAlchemy to load 20K+ sales records into a PostgreSQL database.
- Simulated \~70 new transactions daily to replicate ongoing business activity and validate pipeline reliability.
- Connected Power BI to PostgreSQL to deliver an auto-refreshing dashboard with no manual updates required.

<hr>

## Workflow

<details>
<summary>Click Here to view Workflow Diagram</summary>
&nbsp;

<img title="Workflow Diagram" src="https://github.com/user-attachments/assets/a623868e-ad97-4e6d-8e74-2aefdf62a2bb">

</details>

<hr>

## ER Diagram
The ER (Entity-Relationship) diagram visually represents how different tables in the database are related.

### Relationships
- **One Customer ➜ Many Orders**
  - Each customer can place multiple orders.
- **One Product ➜ Many Orders**
  - Each product can appear in multiple orders.
- **Orders Table ➜ Central Table**
  - Serves as the main transactional table, linking customers and products.

<img width="500px" title="ER Diagram" src="https://github.com/user-attachments/assets/acebc61b-0730-4434-ba6a-409bf71c7762">

<hr>

## Database Schema
The database is designed to store and organize Quick Buy's orders, customers, and product data.

It ensures that all business data is centralized, consistent, and easy to query for analysis and dashboarding.

<details>
<summary>Click Here to view Schema Definition</summary>
<br>

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
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```
</details>

<hr>

## SQL Views
SQL views are used to make data analysis easier and keep business metrics consistent for dashboards.

Instead of running complex queries every time, Power BI connects directly to these views to fetch clean data.

<details>
<summary>Click Here to view SQL Views</summary>
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
```

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
    ROUND(COUNT(DISTINCT order_id)::NUMERIC/COUNT(DISTINCT customer_id), 2) AS avg_orders_per_customer,
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

## Setup

### 1. Clone the Repository
First, clone the project from GitHub to your local system.
```bash
git clone https://github.com/themrityunjaypathak/Dashly.git
```

### 2. Set Up a Virtual Environment
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
> Use the same Python version as in `.github/workflows/etl_pipeline.yaml` to avoid compatibility issues.

### 5. Setup Environment Variables
This project uses a `.env` file to store database credentials like `DB_USER`, `DB_PASS`, `DB_NAME`, etc.

```ini
# .env
DB_HOST=host_name
DB_NAME=database_name
DB_USER=user_name
DB_PASS=password
KAGGLE_USERNAME=kaggle_username
KAGGLE_KEY=kaggle_api_key
```

> [!IMPORTANT]
> Make sure not to commit your `.env` file to GitHub or any public repositories.
>
> You can add it to `.gitignore` to ensure it's excluded from version control.

> [!NOTE]
> If you want to create a free Database in Neon and connect it with Python, go to [How To](#how-to) section.

### 6. Database Connectivity Check
Confirm that the PostgreSQL connection works before running ETL scripts.

This avoids script crashes due to invalid credentials or blocked ports.

<details>
<summary>Click Here to view Code Snippet</summary>
<br>

```python
# Importing Libraries
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Loading Environment File
load_dotenv()

# Loading Database Credentials from Environment File
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Creating SQLAlchemy Engine
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}?sslmode=require&channel_binding=require", pool_pre_ping=True)
```
</details>

### 7. Run ETL Script
This initializes the database and :
- Cleans raw CSV data
- Creates tables (`customers`, `orders`, `products`)
- Loads data into the Neon PostgreSQL database
```bash
python scripts/etl.py
```

> [!NOTE]
> Run this only once initially or when you want a full database refresh.

### 8. Generate New Data
Simulates daily transactions by generating new random data for testing pipeline automation.

Helps verify how dashboards respond to new data over time.
```bash
python scripts/generate_data.py
```

### 9. Create SQL Views
This script builds reusable SQL views that summarize business metrics for the Power BI dashboard.

It simplifies queries, ensures consistent logic, and improves performance.
```bash
python scripts/create_views.py
```

### 10. Export Views as CSVs
Exports SQL view results to CSV files inside the `views/` folder.

This is useful for sharing datasets or validating dashboard data without connecting to the database.
```bash
python scripts/export_views.py
```

### 11. Check Logs
- Check log files inside the `logs/` folder :
  - `etl.log` : Initial data loading
  - `generate_data.log` : Daily data generation
  - `create_views.log` : SQL views creation
  - `export_views.log` : CSV export from views
- Logs help you monitor pipeline performance and troubleshoot errors quickly.

<hr>

## ETL Pipeline
- The ETL (Extract, Transform, Load) pipeline is the core part of this project.
- It automatically cleans and loads sales data into a PostgreSQL database for the Power BI dashboard.
- It is built with Python using SQLAlchemy and is securely configured via environment variables.

### ETL Pipeline Structure
| **Script Name**    | **Purpose**                                                                                  |
| :----------------- | :------------------------------------------------------------------------------------------- |
| `etl.py`           | Sets up the database schema, cleans the dataset, and loads initial data into the database.   |
| `generate_data.py` | Generates random synthetic transaction data to simulate daily updates in the database.       |
| `create_views.py`  | Creates multiple SQL views that summarize and aggregate data for the Power BI dashboard.     |

### How does the ETL pipeline work?
#### 1. `etl.py`
- This script handles the first step of the process by preparing the database.

<details>
<summary>Click Here to View Details</summary>

#### What does it do?
- **Load Configuration**
  - Reads environment variables (like `DB_HOST`, `DB_NAME`) from a `.env` file for secure database access.
- **Logging Setup**
  - Creates a `logs/etl.log` file to track all ETL activity and errors.
- **Extract Data**
  - Loads raw data from a CSV file using a custom `load_csv()` utility function.
- **Transform Data**
  - Removes duplicates, standardizes column names, and optimizes data types.
- **Load Data**
  - Creates tables in the Neon PostgreSQL database and loads the cleaned data using the `to_sql()` function.
- **Schema Management**
  - Ensures relationships between tables using foreign keys and maintains data integrity.

</details>

---

#### 2. `generate_data.py`
- This script keeps the database updated with new transaction data for scheduled data refresh in Power BI.

<details>
<summary>Click Here to View Details</summary>

#### What does it do?
- **Generate Random Data**
  - Uses custom utility functions to create synthetic customer and order data.
- **Data Cleaning**
  - Removes duplicates and optimizes data types before uploading to the Neon database.
- **Append Unique Data**
  - Inserts only new records into the database, avoiding duplicates.
- **Logging Setup**
  - Saves process logs in `logs/generate_data.log`.

</details>

---

#### 3. `create_views.py`
- This script builds SQL views in the PostgreSQL database to simplify analysis and reporting in Power BI.

<details>
<summary>Click Here to View Details</summary>

#### What does it do?
- **Database Connection**
  - Connects to the database securely using environment variables.
- **Define SQL Views**
  - Creates multiple SQL views to summarize and aggregate key business insights.
- **Execute & Commit**
  - Executes each `CREATE OR REPLACE VIEW` statement and commits changes.
- **Logging Setup**
  - Stores execution logs in `logs/create_views.log`.

</details>

---

Once this cycle is complete, the process repeats automatically.

```
   +----------------------+     +----------------------+     +----------------------+
   |     Raw CSV Data     |     |      ETL Script      |     |      PostgreSQL      |
   |   (sales_data.csv)   | --> |       (etl.py)       | --> |       Database       |
   +----------------------+     +----------------------+     +----------------------+
                                                                          |
                                                                          v
   +----------------------+     +----------------------+     +----------------------+
   | Generate Data Script |     |       Power BI       |     |     Views Script     |
   |  (generate_data.py)  | <-- |      Dashboard       | <-- |  (create_views.py)   |
   +----------------------+     +----------------------+     +----------------------+
               |
               |
               +--> Back into PostgreSQL Database (next day's 10:00 AM IST cron run)
```

This ensures that the Power BI Dashboard always displays the latest insights automatically.

<hr>

## Results & Insights
This section highlights the key outcomes and insights generated from the ETL pipeline and Power BI dashboards.

### Pipeline Performance
This section summarizes pipeline performance metrics such as runtime, automation frequency, and reliability.

### 1. Data Loading Overview
| **Parameter**          | **Value**                         |
| :--------------------- | :-------------------------------- |
| **Dataset Size**       | 20K+ sales records                |
| **Tables Used**        | `customers`, `orders`, `products` |
| **Avg. Daily Inserts** | \~70 new orders                   |
| **Database**           | Neon PostgreSQL (cloud-hosted)    |

> [!NOTE]
> This setup simulates ongoing business activity with daily updates to the `orders` and `customers` tables.

<hr>

### 2. Automation & Scheduling
| **Attribute**           | **Details**                                         |
| :---------------------- | :-------------------------------------------------- |
| **Automation Tool**     | GitHub Actions                                      |
| **Execution Frequency** | Daily                                               |
| **Scheduled Time**      | 10:00 AM IST                                        |
| **Trigger Type**        | `cron` (automated) and `workflow_dispatch` (manual) |
| **Runner Environment**  | `ubuntu-latest` (GitHub-hosted Ubuntu runner)       |

> [!NOTE]
> This setup ensures that the latest data is always available for Power BI dashboards, with no manual effort.

<hr>

### 3. Runtime Performance
| **Workflow Step**                    | **Description**                                      |
| :----------------------------------- | :--------------------------------------------------- |
| **Set up job**                       | Initializes GitHub Actions environment               |
| **Checkout repository**              | Pulls repository code into the runner                |
| **Set up Python**                    | Installs Python environment (v3.12)                  |
| **Install dependencies**             | Installs libraries from `requirements.txt`           |
| **Run ETL Script**                   | Extracts, transforms, and loads data into PostgreSQL |
| **Run Generate Data Script**         | Generates new synthetic customer and order data      |
| **Run Views Script**                 | Creates / Refreshes analytical SQL views             |
| **Run Export Views Script**          | Exports SQL views as CSV files                       |
| **Upload Exported CSV as Artifacts** | Uploads exported CSVs to GitHub Actions artifacts    |
| **Commit and Push CSVs**             | Commits CSV files to the repository                  |
| **Upload Logs as Artifacts**         | Uploads log files for debugging and tracking         |
| **Post Setup / Cleanup Steps**       | Cleans the environment post-run                      |

> [!NOTE]
> **Total runtime :** \~47 seconds per pipeline run
> 
> **Scheduling :** The workflow runs daily at **10:00 AM IST** using a `cron` schedule (`30 4 * * *`).
>
> `cron` is in UTC (04:30 UTC = 10:00 AM IST)
> 
> The ETL pipeline runs within a minute, automatically refreshing dashboard data daily with no manual effort.

<hr>

### 4. Error Handling and Logging
| **Aspect**         | **Implementation Details**                                                     |
| :----------------- | :----------------------------------------------------------------------------- |
| **Error Tracking** | Structured `try–except` error handling in each script                          |
| **Log Files**      | `etl.log`, `generate_data.log`, `create_views.log`, `export_views.log`         |
| **Log Storage**    | Uploaded as GitHub Actions run artifacts                                       |
| **Security**       | All credentials securely stored in GitHub Secrets (`DB_USER`, `DB_PASS`, etc.) |

> [!TIP]
> Automated logging and secret handling remove the need for manual checks and ensure smooth workflow runs.

<hr>

### 5. Reliability and Stability
| **Metric**                | **Value**                 | **Remarks**                                |
| :------------------------ | :------------------------ | :----------------------------------------- |
| **Total Runtime**         | \~47 seconds              | Fast for a daily automated ETL pipeline    |
| **Success Rate**          | 100%                      | Verified via GitHub Actions workflow panel |
| **Avg. Records Inserted** | \~70 orders/day           | Lightweight daily incremental updates      |
| **Resource Utilization**  | Low CPU and memory usage  | Efficient for cloud runners                |

> [!IMPORTANT]
> The pipeline runs fully unattended, ensuring consistent daily data updates and automatic Power BI refreshes.

<hr>

### Dashboard Metrics
This section highlights key business insights and trends derived from the Power BI dashboard visualizations.

### 1. Shipping Mode Performance
| **Shipping Mode** | **Total Sales ($)** | **% of Total Sales** | **Total Profit ($)** | **% of Total Profit** | **Profit Margin** |
| :---------------- | :------------------ | :------------------- | :------------------- | :-------------------- | :---------------- |
| Standard Class    |           5,099,197 |            **59.7%** |              897,360 |             **59.8%** |         **17.6%** |
| Second Class      |           1,650,059 |            **19.3%** |              292,629 |             **19.5%** |         **17.7%** |
| First Class       |           1,343,959 |            **15.7%** |              230,784 |             **15.4%** |         **17.2%** |
| Same Day          |             440,836 |             **5.2%** |               78,981 |              **5.3%** |         **17.9%** |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- Standard Class drives \~60% of total sales (\~$5.1M) and profit (\~$897K), making it the dominant shipping mode.
- Second Class contributes \~19% of sales (\~$1.65M) and profit (\~$293K), showing steady usage.
- First Class accounts for \~16% of sales (\~$1.34M) and \~15% of profit (\~$231K), preferred for faster delivery.
- Same Day contributes \~5% of sales (\~$0.44M) and profit (\~$79K), lowest usage but fastest option.
- Profit margins remain consistent across modes (\~17-18%), indicating stable logistics and pricing control.

</details>

<hr>

### 2. Customer Segment Performance
| **Segment** | **Total Sales ($)** | **% of Total Sales** | **Total Profit ($)** | **% of Total Profit** | **Profit Margin** |
| :---------- | :------------------ | :------------------- | :------------------- | :-------------------- | :---------------- |
| Consumer    |           4,263,570 |            **49.9%** |              757,416 |             **50.5%** |         **17.8%** |
| Corporate   |           2,742,160 |            **32.1%** |              475,855 |             **31.7%** |         **17.4%** |
| Home Office |           1,528,321 |            **17.9%** |              266,483 |             **17.8%** |         **17.4%** |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- Consumer Segment is the main revenue driver, generating \~50% of total sales (\~$4.26M) and profit (\~$757K).
- Corporate Segment contributes \~32% of sales (\~$2.74M) and profit (\~$476K), indicating steady performance.
- Home Office delivers \~18% of revenue (\~$1.53M) and profit (\~$266K), smaller but reliable.
- Potential growth opportunity lies in expanding Home Office sales through targeted marketing.

</details>

<hr>

### 3. Monthly Sales & Profit Performance
| **Month** | **Total Sales ($)** | **% of Total Sales** | **Total Profit ($)** | **% of Total Profit** | **Profit Margin** |
| :-------- | :------------------ | :------------------- | :------------------- | :-------------------- | :---------------- |
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

- Q4 (Oct-Dec) drives \~27% of annual sales and profit, making it the strongest quarter for promotions.
- Profit margins stay steady at \~17-18% throughout the year, indicating consistent pricing and cost control.
- December and October are peak months by sales contribution, ideal for marketing campaigns and seasonal offers.
- Sales and profit are relatively evenly distributed across months, reflecting stable and predictable performance.

</details>

<hr>

### 4. Regional Performance Insights
| **Region**  | **Total Sales ($)** | **% of Total Sales** | **Total Profit ($)** | **% of Total Profit** |
| :---------- | :------------------ | :------------------- | :------------------- | :-------------------- |
| **West**    |           2,484,870 |                \~29% |              440,814 |                 \~29% |
| **East**    |           2,456,014 |                \~29% |              427,689 |                 \~28% |
| **Central** |           1,986,280 |                \~23% |              352,400 |                 \~23% |
| **South**   |           1,606,887 |                \~19% |              278,851 |                 \~19% |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- West leads with \~29% of revenue (\~$2.48M) and profit (\~$441K), making it the top-performing region.
- East contributes \~29% of sales (\~$2.46M) and \~28% of profit (\~$428K), showing strong and balanced growth.
- Central adds \~23% of revenue (\~$1.99M) and profit (\~$352K), reflecting steady mid-level performance.
- South accounts for \~19% of revenue (\~$1.61M) and profit (\~$279K), highlighting a key growth opportunity.

</details>

<hr>

### 5. Sub-Category Performance
| **Sub-Category**          | **Total Sales ($)** | **% of Total Sales** | **Total Profit ($)** | **% of Total Profit** |
| :------------------------ | :------------------ | :------------------- | :------------------- | :-------------------- |
| Paper                     |           1,260,080 |                \~15% |              223,147 |                 \~15% |
| Binders                   |             943,376 |                \~11% |              162,084 |                 \~11% |
| Phones                    |             879,975 |                \~10% |              153,018 |                 \~10% |
| Furnishings               |             836,955 |                \~10% |              151,623 |                 \~10% |
| Art                       |             697,965 |                 \~8% |              120,843 |                  \~8% |
| Storage                   |             651,268 |                 \~8% |              112,673 |                  \~8% |
| Accessories               |             643,638 |                 \~7% |              112,007 |                  \~7% |
| Appliances                |             495,030 |                 \~6% |               90,786 |                  \~6% |
| Others (8 sub-categories) |                   — |                    — |                    — |                     — |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- Paper is the top-performing sub-category, contributing \~15% of total sales (\~$1.26M) and profit (\~$223K).
- Binders, Phones, and Furnishings contribute \~31% of total profit (\~$466K), forming the high-performing cluster.
- Art, Storage, and Accessories are solid mid-tier performers, contributing \~23% of total profit combined.
- Scaling high-value categories (Paper, Phones, Furnishings) can boost revenue while maintaining stable margins.

</details>

<hr>

### 6. State-wise Sales Performance
| **State**    | **Total Sales ($)** | **Total Customers** | **% of Total Sales** |
| :----------- | :------------------ | :------------------ | :------------------- |
| California   |           1,792,545 |                 188 |            **21.0%** |
| New York     |             924,728 |                 104 |            **10.8%** |
| Texas        |             903,046 |                  92 |            **10.6%** |
| Pennsylvania |             472,741 |                  45 |             **5.5%** |
| Ohio         |             451,732 |                  51 |             **5.3%** |

<details>
<summary>Click Here to view Key Insights</summary>
&nbsp;

- California leads with \~21% sales (\~$1.8M) and the highest customer base (188), making it the strongest market.
- New York + Texas contribute \~22% of total sales (\~$1.83M), indicating strong demand in dense regions.
- Top 5 states generate \~54% of total revenue (\~$4.54M), highlighting sales concentration in key markets.

</details>

<hr>

## Impact
- Automated the workflow using GitHub Actions, achieving zero failures across 250+ runs at \~47 seconds each.
- Delivered a Power BI dashboard that auto-refreshes daily with no manual effort, eliminating ad-hoc reporting.
- Surfaced sales insights across regions, segments, and sub-categories to support data-driven decisions.

<hr>

## Key Insights
- Standard Class drives \~60% of sales (\~$5.1M) and profit (\~$897K), making it the top profit-driving shipping mode.
- Consumer Segment generates \~50% of revenue (\~$4.26M) and profit (\~$757K), the primary customer base.
- Q4 (Oct-Dec) contributes \~27% of annual revenue, indicating strong seasonal demand, ideal for promotions.
- Paper, Binders, and Phones are the top-performing sub-categories, together making up \~36% of total revenue.
- West and East regions lead with \~58% of total sales, while the South with \~19% shows strong growth potential.
- Top 5 States (CA, NY, TX, PA, OH) generate \~54% of total sales, with CA alone contributing \~21% of sales.

<hr>

## How To

### 1. How to create a free PostgreSQL database on Neon?
- Open [Neon](https://neon.com/) in your browser.
<img title="Neon" src="https://github.com/user-attachments/assets/4a5b08e6-290b-4fb7-9820-3afd0b599e03">
&nbsp;

- Log in using GitHub / Google / Microsoft.
<img title="Login" src="https://github.com/user-attachments/assets/657ccfaa-3c26-492a-9a5f-537f773c763e">
&nbsp;

- Create a New Project.
<img title="New Project" src="https://github.com/user-attachments/assets/dd9271db-e210-4ed0-aeaf-9ce337d06474">
&nbsp;

- Fill in the Project Details.
<img title="Project Details" src="https://github.com/user-attachments/assets/fb2cf22c-6af9-4598-b207-351e4cd66f45">
&nbsp;

- New PostgreSQL project is created in the Neon Console.
<img title="Neon Console" src="https://github.com/user-attachments/assets/eb7f3663-9cda-49ae-94ff-0c866a1d9898">

<hr>

### 2. How to connect Neon Database with Python via SQLAlchemy?
- Open the newly created Project and Click Connect.
<img title="Project Console" src="https://github.com/user-attachments/assets/17a7a0bb-ce46-4496-9f35-b6cce062c791">
&nbsp;

- Get the Connection String of your Database.
<img title="Connection String" src="https://github.com/user-attachments/assets/bd0d5636-38e3-4b3f-abd8-0bc0589f3577">
<p align="center">Image taken from Neon Documentation</p>

- Understand the Connection String.
```
postgresql://alex:AbC123dEf@ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech/dbname?sslmode=require&channel_binding=require
             ^    ^         ^                         ^                              ^
       role -|    |         |- hostname               |- pooler option               |- database
                  |
                  |- password
```
You can use this to configure your database connection.

You can place the connection details in an `.env` file for secure access.

<details>
  <summary>Click Here to view Sample ENV File</summary>
  &nbsp;
  
```ini
# .env file
DB_HOST=ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech
DB_NAME=dbname
DB_USER=alex
DB_PASS=AbC123dEf
```

</details>

After placing connection details in an `.env` file, you can read it via Python using SQLAlchemy.

<details>
  <summary>Click Here to view Code Snippet</summary>
  &nbsp;

```python
# Importing Libraries
import os
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
</details>

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
    - In DirectQuery Mode, Power BI does not import or store data.
    - Instead, it sends real-time queries to the data source every time a user interacts with a report.
    - For live data refresh, choose DirectQuery Mode.
    
</details>

- Once connected, you'll see a list of all available tables and views from your Neon database.
- Select the ones you want to use in your report and Click Load.
<img title="Loading Required Data" src="https://github.com/user-attachments/assets/6f582a78-4eff-45b1-9170-acdccbb08cac">
&nbsp;

- Power BI is now connected to your Neon PostgreSQL database.
- It can refresh live data automatically using DirectQuery Mode.

<hr>

### 4. How to configure GitHub Actions?
- Open [GitHub](https://github.com/) in your browser.
- Navigate to the repository that contains your workflow file (`.github/workflows/etl_pipeline.yaml`).
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

- Open your local `.env` file (this file contains environment variables like API keys, tokens, etc.).
- For each variable, copy :
  - Name : the key (`DB_HOST`)
  - Value : the corresponding value (`ep-cool-darkness-a1b2c3d4-pooler.us-east-2.aws.neon.tech`)
- Paste these into the GitHub form fields :
  - Name ➜ enter the variable name
  - Secret ➜ enter the variable value
<img title="Add New Secret" src="https://github.com/user-attachments/assets/1e701c66-ec31-4c0e-8fb8-21220b4f551f">
&nbsp;

- Your secrets are now securely stored in the repository.
- You can use them inside your workflow `.github/workflows/etl_pipeline.yaml` like this :
```bash
DB_USER: ${{ secrets.DB_USER }}
DB_PASS: ${{ secrets.DB_PASS }}
```

<hr>

## GitHub Actions
- The GitHub Actions workflow automates the entire ETL pipeline, without any manual effort.
- It runs every day at 10:00 AM and ensures that the latest data is always updated.

<details>
<summary>Click Here to view GitHub Actions YAML File</summary>
<br>

```yaml
# ---------------- Name of the Workflow ----------------
name: ETL Pipeline Automation

# ---------------- When should it run? ----------------
on:
  schedule:
    - cron: "30 4 * * *" # Run the workflow daily at 10:00 AM
  workflow_dispatch: # Allow manual run from GitHub UI

# ---------------- Authority to update Repository ----------------
permissions:
  contents: write

# ---------------- Run Only One Pipeline at a Time ----------------
# Queues overlapping runs (cron + manual) instead of letting them race on the DB
concurrency:
  group: etl-pipeline
  cancel-in-progress: false # Don't kill an in-progress run, let it finish, queue the next one instead

# ---------------- Set of steps to run ----------------
jobs:
  data-pipeline:
    runs-on: ubuntu-latest # Use a Linux VM for the Job
    timeout-minutes: 30 # Prevents stuck workflows from running forever

    env: # Shared env variables available to all scripts
      DB_USER: ${{ secrets.DB_USER }}
      DB_PASS: ${{ secrets.DB_PASS }}
      DB_HOST: ${{ secrets.DB_HOST }}
      DB_NAME: ${{ secrets.DB_NAME }}
      KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
      KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}

    steps:
      # ---------------- Step 1 : Checkout Code ----------------
      # This pulls your repository into the GitHub Runner VM
      - name: Checkout repository
        uses: actions/checkout@v4   
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      # ---------------- Step 2 : Set up Python ----------------
      # Installs Python 3.12 so GitHub can run your Scripts
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip" # Speeds up installs by caching pip packages between runs

      # ---------------- Step 3 : Install Dependencies ----------------
      # Installs all Python libraries listed in requirements.txt
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install --no-cache-dir -r requirements.txt

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

<details>
<summary>Click Here to view more Details</summary>

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

### 4. Concurrency
```yaml
concurrency:
  group: etl-pipeline
  cancel-in-progress: false
```
- Ensures only one run of this workflow executes at a time.
- If automated `cron` and manual `workflow_dispatch` overlap, the second run queues instead of running in parallel.
- `cancel-in-progress: false` means the in-progress run finishes fully before the queued run starts.
- Prevents two runs from racing on the same database truncate/insert or pushing conflicting commits.

<hr>

### 5. Jobs
```yaml
jobs:
  data-pipeline:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    env: # Shared env variables available to all scripts
        DB_USER: ${{ secrets.DB_USER }}
        DB_PASS: ${{ secrets.DB_PASS }}
        DB_HOST: ${{ secrets.DB_HOST }}
        DB_NAME: ${{ secrets.DB_NAME }}
        KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
        KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
```
- **jobs :** Defines what tasks (jobs) the workflow will perform.
- **data-pipeline :** The name of your main job.
- **runs-on :** Tells GitHub to use a Linux virtual machine (Ubuntu) to run your scripts.

<hr>

### 6. Main Workflow Tasks
- Each step defines a task GitHub Actions performs in sequence :

#### Step 1 : Checkout Repository
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
```
- This pulls your GitHub Repository files into the VM so the workflow can access your scripts, data and folders.

#### Step 2 : Set Up Python
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.12"
    cache: "pip"
```
- Installs Python 3.12 on the VM, version used by the ETL scripts.
- Caches pip dependencies between runs so `Install dependencies` doesn't cold-install every time.

#### Step 3 : Install Dependencies
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
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
  - Uses GitHub Secrets to securely access database credentials (so they're never exposed in code).

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

</details>

<hr>

## Power BI Dashboard
- The Power BI dashboard is designed to turn data into insights with a clean and interactive interface.
- It connects directly to the database, ensuring the dashboard always reflects the most recent data.
- The dashboard consists of four main pages : Home, Overview, Customers, and Products.
- All pages are connected through page navigation and drill-through features.

### 1. Home Page
- This page serves as an entry point and navigation hub for the dashboard.

<img title="Home" src="https://github.com/user-attachments/assets/a15b8d82-a886-492f-af40-3d8d25ea22ad">

### 2. Overview
- This page focuses on a high-level summary of overall business performance like orders, customers, and products.

<img title="Overview" src="https://github.com/user-attachments/assets/5584efcc-c10a-40c0-b18c-18e907ea64df">

### 3. Customers
- This page focuses on understanding customer behavior, performance, and geographic distribution.

<img title="Customers" src="https://github.com/user-attachments/assets/fd6944a8-59af-43e3-a78d-18b1f92386a4">

### 4. Products
- This page focuses on analyzing product performance, category trends, and geographic purchasing behavior.

<img title="Products" src="https://github.com/user-attachments/assets/ccf6f4ad-4498-4d0b-9284-0c60feeb19c2">

<hr>

## Folder Structure

```
Dashly/
|
├── .github/
│   └── workflows/
│       └── etl_pipeline.yaml        # GitHub Actions Workflow File
|
├── data/
│   └── sales_data.csv               # Original Raw Dataset
|
├── database/
│   ├── schema.sql                   # Database Schema
│   └── views.sql                    # SQL Views
|
├── images/
│   ├── banner.png                   # Banner for Home Page
│   ├── shopping-cart.png            # Multi-Row Card Image
|   └── ...
|
├── scripts/
│   ├── etl.py                       # Script to extract, transform, and load Initial Data
│   ├── generate_data.py             # Script to generate and append New Random Data
│   ├── create_views.py              # Script to create or refresh SQL Views in PostgreSQL Database
│   └── export_views.py              # Script to export SQL Views results as CSV File
│
├── utils/                           # Reusable Python Functions (utils Package)
│   ├── __init__.py
│   ├── data_preprocessor.py
│   ├── generate_random_data.py
|   └── ...
|
├── views/
│   ├── shipping_performance.csv     # Exported SQL Views as CSV Files
│   └── ...
|
├── .gitignore                       # All files and folders ignored by Git
├── Dashboard.pbix                   # Power BI Dashboard File
├── LICENSE                          # License specifying permissions and usage rights
├── README.md                        # Detailed documentation of the Project
└── requirements.txt                 # List of required libraries for the Project
```

<hr>

## License
This project is licensed under the [MIT License](LICENSE). You are free to use and modify the code as needed.

<div align='left'>
  
**[`^        Scroll to Top       ^`](#top)**

</div>
