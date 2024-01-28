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





