/* Code for Datacamp Project: "Analyzing Students' Mental Health" */

-- Run this code to save the CSV file as students
SELECT * 
FROM 'students.csv';

-- Filter df to have only internationals
SELECT 
	stay,
	COUNT(*) AS count_int,
	ROUND(AVG(todep),2) AS average_phq,
	ROUND(AVG(tosc),2) AS average_scs,
	ROUND(AVG(toas),2) AS average_as
FROM students
WHERE Inter_dom = 'Inter'
GROUP BY stay
ORDER BY stay DESC;