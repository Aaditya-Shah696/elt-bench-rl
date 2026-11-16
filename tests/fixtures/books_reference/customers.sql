-- every book line a customer ordered, w the price it sold at
WITH order_lines AS (
    SELECT o.customer_id, ol.book_id, ol.price
    FROM books.cust_order o
    JOIN books.order_line ol ON ol.order_id = o.order_id
),
-- % of their books over $13, 0-100 scale like the answers
line_stats AS (
    SELECT customer_id,
           100.0 * COUNT(*) FILTER (WHERE price > 13) / COUNT(*) AS per_books_over_13
    FROM order_lines
    GROUP BY customer_id
),
order_counts AS (
    SELECT customer_id, COUNT(*) AS num_orders
    FROM books.cust_order
    GROUP BY customer_id
),
-- count orders per method per customer, then rank and keep the most used
preferred_method AS (
    SELECT customer_id, method_name AS preferred_shipping_method
    FROM (
        SELECT o.customer_id, sm.method_name,
               ROW_NUMBER() OVER (PARTITION BY o.customer_id
                                  ORDER BY COUNT(*) DESC, sm.method_name COLLATE "C") AS rn
        FROM books.cust_order o
        JOIN books.shipping_method sm ON sm.method_id = o.shipping_method_id
        GROUP BY o.customer_id, sm.method_name
    ) x
    WHERE rn = 1
),
-- oldest could be more than one book if they share a date so use IN
oldest_book_buyers AS (
    SELECT DISTINCT customer_id
    FROM order_lines
    WHERE book_id IN (
        SELECT book_id FROM books.book
        WHERE publication_date::date = (SELECT MIN(publication_date::date) FROM books.book)
    )
),
international AS (
    SELECT o.customer_id,
           100.0 * COUNT(*) FILTER (WHERE sm.method_name = 'International') / COUNT(*)
               AS per_order_shipped_internationally
    FROM books.cust_order o
    JOIN books.shipping_method sm ON sm.method_id = o.shipping_method_id
    GROUP BY o.customer_id
)
-- every customer left joins so people w no orders still get a row
SELECT
    c.customer_id,
    c.first_name || ' ' || c.last_name AS customer_name,
    c.email,
    ls.per_books_over_13,
    COALESCE(oc.num_orders, 0) AS num_orders,  -- task says null -> 0
    pm.preferred_shipping_method,
    CASE WHEN obb.customer_id IS NOT NULL THEN 1 ELSE 0 END AS has_ordered_the_oldest_book,
    i.per_order_shipped_internationally
FROM books.customer c
LEFT JOIN line_stats ls          ON ls.customer_id = c.customer_id
LEFT JOIN order_counts oc        ON oc.customer_id = c.customer_id
LEFT JOIN preferred_method pm    ON pm.customer_id = c.customer_id
LEFT JOIN oldest_book_buyers obb ON obb.customer_id = c.customer_id
LEFT JOIN international i        ON i.customer_id = c.customer_id
