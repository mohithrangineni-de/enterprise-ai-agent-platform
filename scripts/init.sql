-- scripts/init.sql
-- Sample enterprise analytics schema with realistic data

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10,2) NOT NULL,
    launch_date DATE
);

CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    region TEXT NOT NULL,
    segment TEXT NOT NULL,
    acquisition_date DATE
);

CREATE TABLE IF NOT EXISTS targets (
    period TEXT NOT NULL,  -- e.g. '2024-Q3'
    region TEXT NOT NULL,
    target_revenue NUMERIC(14,2) NOT NULL,
    PRIMARY KEY (period, region)
);

CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    region TEXT NOT NULL,
    product_id INT REFERENCES products(id),
    customer_id INT REFERENCES customers(id),
    revenue NUMERIC(12,2) NOT NULL,
    units_sold INT NOT NULL,
    channel TEXT NOT NULL  -- 'online', 'direct', 'partner'
);

-- Indexes
CREATE INDEX IF NOT EXISTS sales_date_idx ON sales(date);
CREATE INDEX IF NOT EXISTS sales_region_idx ON sales(region);

-- ─── Seed products ────────────────────────────────────────────────────────
INSERT INTO products (name, category, price, launch_date) VALUES
  ('DataSync Pro',     'Software',  1200.00, '2022-01-15'),
  ('CloudVault Basic', 'Cloud',      299.00, '2022-06-01'),
  ('CloudVault Plus',  'Cloud',      799.00, '2023-02-10'),
  ('AnalyticsHub',     'Software',  2400.00, '2023-08-01'),
  ('SecureEdge',       'Security',  1800.00, '2024-01-20')
ON CONFLICT DO NOTHING;

-- ─── Seed targets ─────────────────────────────────────────────────────────
INSERT INTO targets (period, region, target_revenue) VALUES
  ('2024-Q3', 'AMER',  4200000),
  ('2024-Q3', 'EMEA',  3100000),
  ('2024-Q3', 'APAC',  2800000),
  ('2024-Q3', 'LATAM', 1200000),
  ('2024-Q4', 'AMER',  4800000),
  ('2024-Q4', 'EMEA',  3400000),
  ('2024-Q4', 'APAC',  2600000),
  ('2024-Q4', 'LATAM', 1100000)
ON CONFLICT DO NOTHING;

-- ─── Seed sample sales (Q3 + Q4 2024) ─────────────────────────────────────
-- Q3 healthy performance, Q4 APAC drops ~18%
INSERT INTO sales (date, region, product_id, revenue, units_sold, channel)
SELECT
    gs::date,
    r.region,
    (RANDOM() * 4 + 1)::int,
    CASE
        WHEN r.region = 'AMER' THEN 45000 + RANDOM() * 20000
        WHEN r.region = 'EMEA' THEN 32000 + RANDOM() * 15000
        WHEN r.region = 'APAC' AND gs < '2024-10-01' THEN 28000 + RANDOM() * 12000
        WHEN r.region = 'APAC' THEN 16000 + RANDOM() * 8000  -- Q4 drop
        ELSE 11000 + RANDOM() * 5000
    END,
    (RANDOM() * 20 + 5)::int,
    CASE (RANDOM() * 3)::int WHEN 0 THEN 'online' WHEN 1 THEN 'direct' ELSE 'partner' END
FROM generate_series('2024-07-01'::date, '2024-12-31'::date, '1 day') gs
CROSS JOIN (VALUES ('AMER'), ('EMEA'), ('APAC'), ('LATAM')) AS r(region)
WHERE EXTRACT(DOW FROM gs) NOT IN (0, 6);  -- weekdays only
