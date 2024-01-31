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

-- Add an additional joining condition such that you are also joining on year
SELECT 
	name,
	e.year,
	fertility_rate,
	unemployment_rate
FROM 
	countries AS c
INNER JOIN populations AS p
	ON c.code = p.country_code
INNER JOIN economies AS e
	ON c.code = e.code
	AND p.year = e.year;





-- Chapter 2

-- -- Exercise 1

-- Perform an inner join with cities as c1 and countries as c2 on country code
SELECT 
    c1.name AS city,
    code,
    c2.name AS country,
    region,
    city_proper_pop
FROM cities AS c1
INNER JOIN countries AS c2
    ON c1.country_code = c2.code
ORDER BY code DESC;

-- Join right table (with alias)
SELECT 
	c1.name AS city, 
    code, 
    c2.name AS country,
    region, 
    city_proper_pop
FROM cities AS c1
LEFT JOIN countries AS c2
    ON c1.country_code = c2.code
ORDER BY code DESC;


-- -- Exercise 2

SELECT 
    name, region, gdp_percapita
FROM 
    countries AS c
LEFT JOIN economies AS e
-- Match on code fields
    USING(code)
-- Filter for the year 2010
WHERE
    year = 2010;


-- Select region, and average gdp_percapita as avg_gdp
SELECT
    region,
    AVG(gdp_percapita) AS avg_gdp
FROM countries AS c
LEFT JOIN economies AS e
USING(code)
WHERE year = 2010
-- Group by region
GROUP BY region;


SELECT region, AVG(gdp_percapita) AS avg_gdp
FROM countries AS c
LEFT JOIN economies AS e
USING(code)
WHERE year = 2010
GROUP BY region
-- Order by descending avg_gdp
ORDER BY avg_gdp DESC
-- Return only first 10 records
LIMIT(10);


-- -- Exercise 3

-- Modify this query to use RIGHT JOIN instead of LEFT JOIN
SELECT 
    countries.name AS country,
    languages.name AS language,
    percent
FROM languages
RIGHT JOIN countries
    USING(code)
ORDER BY language;



-- -- Exercise 4

SELECT
    name AS country,
    code,
    region,
    basic_unit
FROM countries
-- Join to currencies
FULL JOIN currencies 
    USING(code)
-- Where region is North America or name is null
WHERE
    region = 'North America'
    OR
    name IS NULL
ORDER BY region;


SELECT
	name AS country,
	code,
	region,
	basic_unit
FROM countries
-- Join to currencies
LEFT JOIN currencies 
	USING (code)
WHERE region = 'North America' 
	OR name IS NULL
ORDER BY region;


SELECT 
	name AS country,
	code,
	region,
	basic_unit
FROM countries
-- Join to currencies
INNER JOIN currencies 
	USING (code)
WHERE region = 'North America' 
	OR name IS NULL
ORDER BY region;




-- -- Exercise 5

SELECT 
	c1.name AS country, 
    region, 
    l.name AS language,
	basic_unit, 
    frac_unit
FROM countries as c1 
-- Full join with languages (alias as l)
FULL JOIN languages AS l
    USING(code)
-- Full join with currencies (alias as c2)
FULL JOIN currencies AS c2
    USING(code)
WHERE region LIKE 'M%esia';


-- -- Exercise 6

SELECT
	c.name AS country,
	l.name AS language
-- Inner join countries as c with languages as l on code
FROM countries AS c 
INNER JOIN languages AS l
	USING(code)
WHERE c.code IN ('PAK','IND')
	AND l.code in ('PAK','IND');

SELECT
	c.name AS country,
	l.name AS language
FROM countries AS c        
-- Perform a cross join to languages (alias as l)
CROSS JOIN languages AS l
WHERE c.code in ('PAK','IND')
	AND l.code in ('PAK','IND');


-- -- Exercise 7

/* Determine the names of the five countries 
and their respective regions with the lowest 
life expectancy for the year 2010. */

SELECT
	c.name AS country,
	region,
	life_expectancy AS life_exp
FROM 
	countries AS c
FULL JOIN populations AS p
	ON c.code = p.country_code
WHERE year = 2010
ORDER BY life_expectancy ASC
LIMIT 5;


-- -- Exercise 8

-- Select aliased fields from populations as p1
-- Join populations as p1 to itself, alias as p2, on country code
SELECT
    p1.country_code,
    p1.size AS size2010,
    p2.size AS size2015
FROM populations AS p1
INNER JOIN populations AS p2 
    ON p1.country_code = p2.country_code;

-- Filter such that p1.year is always five years before p2.year
SELECT 
	p1.country_code, 
    p1.size AS size2010, 
    p2.size AS size2015
FROM populations AS p1
INNER JOIN populations AS p2
	ON p1.country_code = p2.country_code
WHERE p1.year = 2010
    AND p1.year = p2.year - 5;


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





