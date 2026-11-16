-- one row per author+book pair, everything else builds off this
WITH author_books AS (
    SELECT ba.author_id, b.book_id, b.title, b.num_pages,
           b.publication_date::date AS published  -- dates came in as text so cast
    FROM books.book_author ba
    JOIN books.book b ON b.book_id = ba.book_id
),
-- number each authors books oldest first and keep #1
first_books AS (
    SELECT author_id, title AS first_book
    FROM (
        SELECT author_id, title,
               -- collate C so ties sort like snowflake (uppercase before lowercase)
               ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY published, title COLLATE "C") AS rn
        FROM author_books
    ) x
    WHERE rn = 1
),
book_stats AS (
    SELECT author_id,
           -- 100.0 not 100 or postgres does integer division and you get 0
           100.0 * COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM published) = 1992) / COUNT(*) AS per_book_published_1992,
           SUM(num_pages) AS total_pages  -- used later to compare against the avg
    FROM author_books
    GROUP BY author_id
),
-- books dont have a price, only order lines do, so price = what it sold for
sales AS (
    SELECT ab.author_id, ab.title, ol.price
    FROM author_books ab
    JOIN books.order_line ol ON ol.book_id = ab.book_id
),
-- same trick as first_books but highest price, ties go to the last title
most_expensive AS (
    SELECT author_id, title AS most_expensive_book
    FROM (
        SELECT author_id, title,
               ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY price DESC, title COLLATE "C" DESC) AS rn
        FROM sales
    ) x
    WHERE rn = 1
),
avg_price AS (
    SELECT author_id, AVG(price) AS avg_price
    FROM sales
    GROUP BY author_id
)
-- start from every author and left join so authors w no books still show up
SELECT
    a.author_id,
    a.author_name,
    fb.first_book,
    bs.per_book_published_1992,
    -- avg is over all authors totals, no books means null which falls to 0
    CASE WHEN bs.total_pages < (SELECT AVG(total_pages) FROM book_stats) THEN 1 ELSE 0 END
        AS wrote_fewer_total_pages_than_the_average,
    me.most_expensive_book,  -- null if none of their books ever sold
    CASE WHEN ap.avg_price > 19 THEN 1 ELSE 0 END AS avg_book_price_over_19
FROM books.author a
LEFT JOIN first_books fb    ON fb.author_id = a.author_id
LEFT JOIN book_stats bs     ON bs.author_id = a.author_id
LEFT JOIN most_expensive me ON me.author_id = a.author_id
LEFT JOIN avg_price ap      ON ap.author_id = a.author_id
