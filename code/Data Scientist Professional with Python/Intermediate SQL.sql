/* Code Exercises from Intermediate SQL */

-- Chapter 1

-- -- Exercise 1

-- Count the number of records in the people table
SELECT COUNT(*) AS count_records
FROM people;

-- Count the number of birthdates in the people table
SELECT COUNT(birthdate) AS count_birthdate
FROM people;

-- Count the records for languages and countries represented in the films table
SELECT 
    COUNT(language) AS count_languages,
    COUNT(country) AS count_countries
FROM films;


-- -- Exercise 2

-- Return the unique countries from the films table
SELECT DISTINCT country
FROM films;

-- Count the distinct countries from the films table
SELECT COUNT(DISTINCT country) AS count_distinct_countries
FROM films;



-- -- Exercise 3

-- Debug this code
SELECT certification
FROM films
LIMIT 5;

-- Debug this code
SELECT 
    film_id,
    imdb_score,
    num_votes
FROM reviews;

-- Debug this code
SELECT COUNT(birthdate) AS count_birthdays
FROM people;


-- -- Exercise 4

-- Rewrite this query
SELECT 
    person_id, 
    role 
FROM roles 
LIMIT 10;



-- Chapter 2

-- -- Exercise 1

-- Select film_ids and imdb_score with an imdb_score over 7.0
SELECT 
    film_id,
    imdb_score
FROM
    reviews
WHERE 
    imdb_score > 7.0;

-- Select film_ids and facebook_likes for ten records with less than 1000 likes 
SELECT 
    film_id,
    facebook_likes
FROM
    reviews
WHERE 
    facebook_likes < 1000
LIMIT 10;

-- Count the records with at least 100,000 votes
SELECT 
    COUNT(*) as films_over_100k_votes
FROM
    reviews
WHERE 
    num_votes >= 100000;


-- -- Exercise 2

-- Count the Spanish-language films
SELECT
    COUNT(*) as count_spanish
FROM films
WHERE 
    language = 'Spanish';



-- -- Exercise 3

-- Select the title and release_year for all German-language films released before 2000
SELECT
    title,
    release_year
FROM 
    films
WHERE 
    release_year < 2000
    AND 
    language = 'German';

-- Select all records for German-language films released after 2000 and before 2010
SELECT *
FROM films
WHERE 
    (release_year > 2000 AND release_year < 2010)
    AND 
    language = 'German';


-- -- Exercise 4

-- Find the title and year of films from the 1990 or 1999
SELECT 
    title,
    release_year
FROM 
    films
WHERE
    release_year = 1990
    OR
    release_year = 1999;


-- Filter to see only English or Spanish-language films
SELECT 
	title, 
	release_year
FROM 
	films
WHERE 
	(release_year = 1990 OR release_year = 1999)
	AND 
	(language = 'English' OR language = 'Spanish');


-- Filter films with more than $2,000,000 gross
SELECT 
	title, 
	release_year
FROM 
	films
WHERE 
	(release_year = 1990 OR release_year = 1999)
	AND 
	(language = 'English' OR language = 'Spanish')
	AND
	(gross > 2000000);


-- -- Exercise 5

-- Select the title and release_year for films released between 1990 and 2000
SELECT
    title,
    release_year
FROM
    films
WHERE 
    release_year BETWEEN 1990 AND 2000;

-- Build on the previous query and select only films with budget over 100 million
SELECT
    title,
    release_year
FROM
    films
WHERE 
    release_year BETWEEN 1990 AND 2000
	AND
	budget > 100000000;

-- Restrict the query to only Spanish-language films
SELECT 
	title, 
	release_year
FROM 
	films
WHERE 
	release_year BETWEEN 1990 AND 2000
	AND 
	budget > 100000000
	AND
	language = 'Spanish';

-- Amend the query to include Spanish or French-language films
SELECT 
	title, 
	release_year
FROM 
	films
WHERE 
	release_year BETWEEN 1990 AND 2000
	AND 
	budget > 100000000
	AND
	(language = 'Spanish' OR language = 'French');


-- -- Exercise 6

-- Select the names that start with B
SELECT name
FROM people
WHERE name LIKE 'B%';

-- Select the names that have r as the second letter
SELECT name
FROM people
WHERE name LIKE '_r%';

-- Select names that don't start with A
SELECT name
FROM people
WHERE name NOT LIKE 'A%';


-- -- Exercise 7

-- Find the title and release_year for all films over two hours in length released in 1990 and 2000
SELECT
    title,
    release_year
FROM
    films
WHERE
    duration > 120
    AND
    (release_year = 1990 OR release_year = 2000);

-- Find the title and language of all films in English, Spanish, and French
SELECT
    title,
    language
FROM 
    films
WHERE 
    language IN ('English', 'Spanish', 'French');

-- Find the title, certification, and language all films certified NC-17 or R that are in English, Italian, or Greek
SELECT
    title,
    certification,
    language
FROM 
    films
WHERE
    certification IN ('NC-17', 'R')
    AND 
    language IN ('English', 'Italian', 'Greek');


-- -- Exercise 8

-- Count the unique titles
SELECT 
	COUNT(DISTINCT title) AS nineties_english_films_for_teens
FROM 
	films
-- Filter to release_years to between 1990 and 1999
WHERE 
	release_year BETWEEN 1990 AND 1999
	AND
-- Filter to English-language films
	language = 'English'
	AND
-- Narrow it down to G, PG, and PG-13 certifications
	certification IN ('G', 'PG', 'PG-13');



-- -- Exercise 9

-- List all film titles with missing budgets
SELECT 
    title AS no_budget_info
FROM 
    films
WHERE 
    budget IS NULL;

-- Count the number of films we have language data for
SELECT
    COUNT(language) AS count_language_known
FROM 
    films;



-- Chapter 3

-- -- Exercise 1

-- Query the sum of film durations
SELECT
    SUM(duration) AS total_duration
FROM
    films;

-- Calculate the average duration of all films
SELECT
    avg(duration) AS average_duration
FROM
    films;

-- Find the latest release_year
SELECT
    MAX(release_year) AS latest_year
FROM
    films;

-- Find the duration of the shortest film
SELECT
    MIN(duration) AS shortest_film
FROM 
    films;


-- -- Exercise 2

-- Calculate the sum of gross from the year 2000 or later
SELECT
    SUM(gross) AS total_gross
FROM 
    films
WHERE 
    release_year >= 2000;

-- Calculate the average gross of films that start with A
SELECT
    AVG(gross) AS avg_gross_A
FROM 
    films
WHERE 
    title LIKE 'A%';

-- Calculate the lowest gross film in 1994
SELECT
    MIN(gross) AS lowest_gross
FROM
    films
WHERE
    release_year = 1994;

-- Calculate the highest gross film released between 2000-2012
SELECT
    MAX(gross) AS highest_gross
FROM
    films
WHERE
    release_year BETWEEN 2000 AND 2012;


-- -- Exercise 3

-- Round the average number of facebook_likes to one decimal place
SELECT
    ROUND(AVG(facebook_likes),1) AS avg_facebook_likes
FROM 
    reviews;


-- -- Exercise 4

-- Calculate the average budget rounded to the thousands
SELECT
    ROUND(AVG(budget),-3) AS avg_budget_thousands
FROM
    films;


-- -- Exercise 5

-- Calculate the title and duration_hours from films
SELECT 
    title,
    (duration / 60.0) as duration_hours
FROM films;

-- Calculate the percentage of people who are no longer alive
SELECT 
    COUNT(deathdate) * 100.0 / COUNT(*) AS percentage_dead
FROM 
    people;

-- Find the number of decades in the films table
SELECT 
    (MAX(release_year) - MIN(release_year)) / 10.0 AS number_of_decades
FROM 
    films;

   
-- -- Exercise 6

-- Round duration_hours to two decimal places
SELECT 
    title, 
    ROUND(duration / 60.0, 2) AS duration_hours
FROM films;




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





