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

/* Products Table */
CREATE TABLE IF NOT EXISTS products (
  product_id TEXT PRIMARY KEY,
  product_name TEXT,
  category TEXT,
  sub_category TEXT
);

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