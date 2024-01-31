/* Code Exercises from Joining Data in SQL */

-- Chapter 1

-- -- Exercise 1

-- Select all columns from cities
SELECT *
FROM cities;

SELECT * 
FROM cities
-- Inner join to countries
INNER JOIN countries
-- Match on country codes
    ON cities.country_code = countries.code;

-- Select name fields (with alias) and region 
SELECT 
    cities.name AS city,
    countries.name AS country,
    countries.region
FROM 
    cities
INNER JOIN countries
    ON cities.country_code = countries.code;

-- Select fields with aliases
SELECT
    c.code AS country_code,
    c.name,
    e.year,
    e.Inflation_rate
FROM countries AS c
-- Join to economies (alias e)
INNER JOIN economies AS e
-- Match on code field using table aliases
    ON c.code = e.code


-- -- Exercise 2

SELECT 
	c.name AS country,
	l.name AS language,
	official
FROM countries AS c
INNER JOIN languages AS l
-- Match using the code column
    USING(code)


-- -- Exercise 3

-- Select country and language names, aliased
SELECT 
    c.name AS country,
    l.name AS language
-- From countries (aliased)
FROM 
    countries AS c
-- Join to languages (aliased)
INNER JOIN languages as l
-- Use code as the joining field with the USING keyword
    USING(code);

-- Rearrange SELECT statement, keeping aliases
SELECT 
    l.name AS language,
    c.name AS country
FROM countries AS c
INNER JOIN languages AS l
    USING(code)
-- Order the results by language
ORDER BY 
    language;


-- -- Exercise 4

-- Check validity of A (valid)
SELECT
    c.name AS country,
    l.name AS language
FROM countries AS c
INNER JOIN languages AS l 
    USING(code)
WHERE 
    c.name = 'Armenia'
ORDER BY country;

-- Check validity of B (invalid)
SELECT
    c.name AS country,
    l.name AS language
FROM countries AS c
INNER JOIN languages AS l 
    USING(code)
WHERE 
    l.name = 'Alsatlan'
ORDER BY country;


-- Check validity of C (valid)
SELECT
    c.name AS country,
    l.name AS language
FROM countries AS c
INNER JOIN languages AS l 
    USING(code)
WHERE 
    l.name = 'Bhojpuri'
ORDER BY country;


-- -- Exercise 5

-- 1 Select relevant fields
-- 2 Inner join countries and populations, aliased, on code
SELECT
    name,
    year,
    fertility_rate
FROM countries AS c 
INNER JOIN populations AS p
    ON c.code = p.country_code;

-- Select fields
-- Join to economies (as e)
-- Match on country code
SELECT 
    name,
    e.year,
    fertility_rate,
    e.unemployment_rate
FROM countries AS c
INNER JOIN populations AS p
    ON c.code = p.country_code
INNER JOIN economies AS e
    ON e.code = c.code;
    

-- -- Exercise 6





-- -- Exercise 7





-- -- Exercise 8





-- -- Exercise 9





-- Chapter 2

-- -- Exercise 1





-- -- Exercise 2





-- -- Exercise 3





-- -- Exercise 4





-- -- Exercise 5





-- -- Exercise 6





-- -- Exercise 7





-- -- Exercise 8





-- -- Exercise 9





-- Chapter 3

-- -- Exercise 1





-- -- Exercise 2





-- -- Exercise 3





-- -- Exercise 4





-- -- Exercise 5





-- -- Exercise 6





-- -- Exercise 7





-- -- Exercise 8





-- -- Exercise 9





-- Chapter 4

-- -- Exercise 1





-- -- Exercise 2





-- -- Exercise 3





-- -- Exercise 4





-- -- Exercise 5





-- -- Exercise 6





-- -- Exercise 7





-- -- Exercise 8





-- -- Exercise 9





