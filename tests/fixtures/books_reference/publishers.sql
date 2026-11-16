-- books w their language name and a real date, everything else uses this
WITH books AS (
    SELECT b.book_id, b.publisher_id, b.title, b.num_pages,
           b.publication_date::date AS published,
           bl.language_name
    FROM books.book b
    LEFT JOIN books.book_language bl ON bl.language_id = b.language_id
),
book_stats AS (
    SELECT publisher_id,
           COUNT(*) AS num_books,
           -- avg is over ALL books not just this publishers
           COUNT(*) FILTER (WHERE num_pages > 0.7 * (SELECT AVG(num_pages) FROM books.book))
               AS num_books_pages_greater_than_70_per_of_avg,
           -- 100.0 so no integer division, 0-100 scale like the answers
           100.0 * COUNT(*) FILTER (WHERE language_name = 'Japanese') / COUNT(*) AS per_japanese_books
    FROM books
    GROUP BY publisher_id
),
-- oldest first, ties go to the first title (collate C so it sorts like snowflake)
oldest AS (
    SELECT publisher_id, title AS title_of_the_oldest_book
    FROM (
        SELECT publisher_id, title,
               ROW_NUMBER() OVER (PARTITION BY publisher_id ORDER BY published, title COLLATE "C") AS rn
        FROM books
    ) x
    WHERE rn = 1
),
-- most pages first, ties go to the last title this time
most_pages AS (
    SELECT publisher_id, title AS book_title_with_the_most_pages
    FROM (
        SELECT publisher_id, title,
               ROW_NUMBER() OVER (PARTITION BY publisher_id ORDER BY num_pages DESC, title COLLATE "C" DESC) AS rn
        FROM books
    ) x
    WHERE rn = 1
)
-- start from publishers and left join so ones w zero books still show up
SELECT
    p.publisher_id,
    p.publisher_name,
    COALESCE(bs.num_books, 0) AS num_books,  -- task says null -> 0
    COALESCE(bs.num_books_pages_greater_than_70_per_of_avg, 0) AS num_books_pages_greater_than_70_per_of_avg,
    bs.per_japanese_books,
    o.title_of_the_oldest_book,
    mp.book_title_with_the_most_pages
FROM books.publisher p
LEFT JOIN book_stats bs ON bs.publisher_id = p.publisher_id
LEFT JOIN oldest o      ON o.publisher_id = p.publisher_id
LEFT JOIN most_pages mp ON mp.publisher_id = p.publisher_id
