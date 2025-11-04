# Importing Libraries
import os
import random
import kagglehub
import numpy as np
import pandas as pd
from faker import Faker
from functools import reduce
from datetime import timedelta

# Setting up Faker to generate Fake Data
faker = Faker("en_US")

# Downloading the latest version of the Dataset from Kaggle
path = kagglehub.dataset_download("vivek468/superstore-dataset-final")

# Creating CSV File Path
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file_path = os.path.join(path, file)

# Reading the CSV File
df = pd.read_csv(csv_file_path, encoding_errors="ignore")

# Standardizing Column Names
df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")

# Customers DataFrame
customers_cols = ["customer_id", "customer_name", "segment", "city", "state", "country", "postal_code", "region"]
customers = df[customers_cols].drop_duplicates(subset="customer_id")

# Products DataFrame
products_cols = ["product_id", "product_name", "category", "sub_category"]
products = df[products_cols].drop_duplicates(subset="product_id")
products.columns = products.columns.str.title().str.replace("_"," ")

# Orders DataFrame
orders_cols = ["order_id", "order_date", "customer_id", "product_id", "ship_mode", "ship_date", "sales", "quantity", "discount", "profit"]
orders = df[orders_cols].drop_duplicates(subset="order_id")

def random_customers_data(lower_limit, upper_limit):
    """
    Generate a randomized customers DataFrame based on existing customer distribution.

    This function creates a synthetic dataset of customers using Faker and weighted
    sampling based on the frequency distribution of existing customer attributes
    (segment, city, state, region). It ensures that each generated customer_id is unique.

    Parameters:
        lower_limit (int): Minimum number of random customers to generate.
        upper_limit (int): Maximum number of random customers to generate.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic customer records with the following columns:
                      - Customer Id
                      - Customer Name
                      - Segment
                      - City
                      - State
                      - Country
                      - Postal Code
                      - Region

    Notes:
        The function uses a global variable `customers_df` to store the final DataFrame.
        Unique `Customer Id` values are generated using initials, random digits, and postal code.
        Attribute selection follows the same statistical distribution as the original dataset.
        Country is always set to "United States".
    """
    # Data and Weights to generate Random Values
    segment_data = np.sort(customers["segment"].unique())
    segment_weight = customers["segment"].value_counts(normalize=True).sort_index().values

    city_data = np.sort(customers["city"].unique())
    city_weight = customers["city"].value_counts(normalize=True).sort_index().values

    state_data = np.sort(customers["state"].unique())
    state_weight = customers["state"].value_counts(normalize=True).sort_index().values

    region_data = np.sort(customers["region"].unique())
    region_weight = customers["region"].value_counts(normalize=True).sort_index().values

    # Generating Random Customers Data
    customers_data = []
    used_ids = set()

    for _ in range(random.randint(lower_limit, upper_limit)):
        while True:
            customer_name = faker.name()
            postal_code = faker.numerify("%####")
            customer_id = f"{reduce(lambda x, y: x[0] + y[0], customer_name.split())}-{faker.numerify('%####')}-{postal_code}"
            
            if customer_id not in used_ids:
                used_ids.add(customer_id)
                break

        segment = random.choices(segment_data, weights=segment_weight, k=1)[0]
        city = random.choices(city_data, weights=city_weight, k=1)[0]
        state = random.choices(state_data, weights=state_weight, k=1)[0]
        country = "United States"
        postal_code = postal_code
        region = random.choices(region_data, weights=region_weight, k=1)[0]

        customers_data.append({
            "Customer Id": customer_id,
            "Customer Name": customer_name,
            "Segment": segment,
            "City": city,
            "State": state,
            "Country": country,
            "Postal Code": int(postal_code),
            "Region": region
        })

    global customers_df
    customers_df = pd.DataFrame(customers_data)
    return customers_df

def random_orders_data(lower_limit, upper_limit):
    """
    Generate a randomized orders DataFrame based on existing order distribution.

    This function creates synthetic order records using weighted sampling from existing
    order data, including realistic patterns for order date, shipping mode, shipping
    duration, sales ranges, quantity, discount and profit margin.

    Parameters:
        lower_limit (int): Minimum number of random orders to generate.
        upper_limit (int): Maximum number of random orders to generate.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic order records with the following columns:
                      - Order Id
                      - Order Date
                      - Customer Id
                      - Product Id
                      - Ship Mode
                      - Ship Date
                      - Sales
                      - Quantity
                      - Discount
                      - Profit

    Notes:
        Uses weighted distributions taken from original orders dataset for realism.
        Order Ids are generated using sampled prefixes + year + random digits.
        Shipping duration is calculated using the mean duration per shipping mode.
        Profit is computed using a randomized profit margin (5%-30%).
        Requires that `customers_df` and `products` DataFrames already exist.
        Ensures uniqueness of generated order IDs.

    Dependencies:
        `customers_df` (global DataFrame) must already be created using random_customers_data()
        `products` DataFrame is expected to be preloaded from original dataset
    """
    # Data and Weights to generate Random Values
    order_id_data = orders["order_id"].str.split("-").str.get(0).unique()
    order_id_weight = orders["order_id"].str.split("-").str.get(0).value_counts(normalize=True).values

    ship_mode_data = np.sort(orders["ship_mode"].unique())
    ship_mode_weight = orders["ship_mode"].value_counts(normalize=True).sort_index().values

    sales_data = [range(50, 500), range(500, 1000), range(1000, 2000), range(2000, int(orders["sales"].max()))]
    sales_weight = pd.cut(orders["sales"], bins=[0, 500, 1000, 2000, int(orders["sales"].max())]).value_counts(normalize=True).values

    quantity_data = np.sort(orders["quantity"].unique())
    quantity_weight = orders["quantity"].value_counts(normalize=True).sort_index().values

    discount_data = np.sort(orders["discount"].unique())
    discount_weight = orders["discount"].value_counts(normalize=True).sort_index().values

    # Shipping Duration Dictionary
    orders["shipping_duration"] = (pd.to_datetime(orders["ship_date"]) - pd.to_datetime(orders["order_date"])).dt.days
    shipping_duration_map = np.round(orders.groupby("ship_mode")["shipping_duration"].mean()).to_dict()

    # Generating Orders Data
    orders_data = []
    used_ids = set()

    for _ in range(random.randint(lower_limit, upper_limit)):
        while True:
            order_date = faker.date_between(start_date="-5y", end_date="today")
            order_id = f"{random.choices(order_id_data, weights=order_id_weight, k=1)[0]}-{order_date.strftime('%Y')}-{faker.numerify('%#####')}"

            if order_id not in used_ids:
                used_ids.add(order_id)
                break

        customer_id = random.choice(customers_df["Customer Id"].to_list())
        product_id = random.choice(products["Product Id"].to_list())
        ship_mode = random.choices(ship_mode_data, weights=ship_mode_weight, k=1)[0]
        ship_date = order_date + timedelta(days=shipping_duration_map.get(ship_mode))
        sales = random.choice(random.choices(sales_data, weights=sales_weight, k=1)[0])
        quantity = random.choices(quantity_data, weights=quantity_weight, k=1)[0]
        discount = random.choices(discount_data, weights=discount_weight, k=1)[0]
        profit_margin = random.uniform(0.05, 0.30)
        profit = round(sales*profit_margin)

        orders_data.append({
            "Order Id": order_id,
            "Order Date": order_date,
            "Customer Id": customer_id,
            "Product Id": product_id,
            "Ship Mode": ship_mode,
            "Ship Date": ship_date,
            "Sales": sales,
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit
        })

    return pd.DataFrame(orders_data)    