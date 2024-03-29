**Course roadmap**
- Querying databases
- Count and view specified records
- Understand query execution and style 
- Filtering
- Aggregate functions
- Sorting and grouping

# Querying a database

## `COUNT()`
The `COUNT()` function in SQL is used to count the number of rows in a result set or the number of occurrences of a value in a column. It's a versatile and commonly used aggregate function. 

**Basic Syntax:**

```sql
SELECT COUNT(*)
FROM your_table;
```
This counts all rows in the specified table.

```sql
SELECT COUNT(column_name)
FROM your_table;
```
This counts the number of non-null values in the specified column.

**Note:**
- `COUNT(*)` counts all rows, including those with null values in any column.
- `COUNT(column_name)` counts the number of non-null values in the specified column.

### **Examples:**

1. **Counting All Rows:**
   ```sql
   SELECT COUNT(*)
   FROM employees;
   ```

   This returns the total number of rows in the "employees" table.

2. **Counting Non-Null Values in a Column:**
   ```sql
   SELECT COUNT(employee_id)
   FROM employees;
   ```

   This returns the number of non-null values in the "employee_id" column of the "employees" table.

3. **Counting Distinct Values:**
   ```sql
   SELECT COUNT(DISTINCT department_id)
   FROM employees;
   ```

   This returns the number of distinct department IDs in the "employees" table.

### **Use Cases**

1. **Data Exploration:**
   - `COUNT()` is often used to get a quick overview of the size of a dataset.

2. **Checking for Missing Data:**
   - By counting non-null values in a column, you can identify missing or incomplete data.

3. **Aggregation with GROUP BY:**
   - When combined with `GROUP BY`, `COUNT()` can provide counts for different groups within a dataset.

   ```sql
   SELECT department_id, COUNT(*)
   FROM employees
   GROUP BY department_id;
   ```

4. **Checking Data Quality:**
   - It's useful for verifying data quality by ensuring that certain conditions are met.

   ```sql
   SELECT COUNT(*)
   FROM orders
   WHERE order_date IS NULL;
   ```

   This counts the number of records with a null "order_date."

5. **Distinct Value Counts:**
   - When used with `DISTINCT`, `COUNT()` can provide the number of unique values in a column.

   ```sql
   SELECT COUNT(DISTINCT product_category)
   FROM products;
   ```

   This returns the number of unique product categories.

# Query execution

- SQL is not processed in its written order
```sql
SELECT
	col1     -- processed second
FROM 
	table1   -- processed first
LIMIT 10;    -- processed third
```

It is good to know processing order for debugging and aliasing.

## Debugging SQL

Some error messages in SQL are very helpful and clear:
- Misspelling
- Incorrect capitalization
- Incorrect or missing punctuation

While others are less helpful:
- Comma errors: will give approximate location of the error and call it a syntax error.

# SQL style

In terms of running a query formatting is not required, for example this query will run correctly:
```sql
select title, release_year, country from films limit 3
```
But a more readable and better way to write it would be:
```sql
SELECT
	title,
	release_year.
	country
FROM
	films
LIMIT 3;
```
It is also helpful to follow a SQL style guide such as Holywell's style guide [here](https://www.sqlstyle.guide/).

## Dealing with non-standard field names

Although it is best practice to not include spaces in field or table names, it may be that one has it.
- `release year` instead of `release_year`
In such case we need to use double quotes, like so:
```sql
SELECT
	title,
	"release year",
	country
FROM 
	films
LIMIT 3;
```


# Filtering numbers

## WHERE

The `WHERE` clause in SQL is used to filter the rows returned in a query based on specified conditions. It allows you to narrow down the result set to only include rows that meet specific criteria.

Conditions in the `WHERE` clause can include various operators, such as `=`, `<>` (not equal), `<`, `>`, `<=`, `>=`, `LIKE`, `IN`, `BETWEEN`, and logical operators like `AND`, `OR`, `NOT`.

**Basic Syntax:**

```sql
SELECT column1, column2
FROM your_table
WHERE condition;
```

**Examples:**

1. **Filtering Rows Based on a Single Condition:**
   ```sql
   -- Select employees with a salary greater than 50000
   SELECT employee_id, employee_name, salary
   FROM employees
   WHERE salary > 50000;
   ```

2. **Filtering Rows Based on Multiple Conditions:**
   ```sql
   -- Select employees hired after a certain date in a specific department
   SELECT employee_id, employee_name, hire_date, department_id
   FROM employees
   WHERE hire_date > '2022-01-01' AND department_id = 2;
   ```

3. **Using Comparison Operators:**
   ```sql
   -- Select products with a price between 50 and 100
   SELECT product_id, product_name, price
   FROM products
   WHERE price BETWEEN 50 AND 100;
   ```

**Use Cases:**

1. **Data Filtering:**
   - The primary use of `WHERE` is to filter rows based on specific conditions, allowing you to focus on relevant data.

2. **Data Validation:**
   - Use `WHERE` to validate data against certain criteria, identifying and handling data quality issues.

   ```sql
   -- Identify orders with missing customer IDs
   SELECT *
   FROM orders
   WHERE customer_id IS NULL;
   ```

3. **Time-Based Filtering:**
   - Filter data based on time-related conditions, such as selecting orders from the last month.

   ```sql
   -- Select orders placed in the last 30 days
   SELECT *
   FROM orders
   WHERE order_date >= CURRENT_DATE - INTERVAL '30' DAY;
   ```

4. **Combining with Aggregation:**
   - Combine `WHERE` with aggregate functions like `COUNT` to filter data before performing calculations.

   ```sql
   -- Count the number of employees in a specific department
   SELECT COUNT(*)
   FROM employees
   WHERE department_id = 3;
   ```

5. **Conditional Retrieval:**
   - Retrieve data based on conditional requirements, such as finding products in stock.

   ```sql
   -- Select products with stock greater than 0
   SELECT *
   FROM products
   WHERE stock_quantity > 0;
   ```


# Multiple criteria filtering

## `OR`, `AND`, `BETWEEN`

We can use these three operators to filter multiple criteria at the same time:
- Using `OR`: when you need to satisfy at least one condition.
```SQL
SELECT *
FROM coats
WHERE color = 'yellow' OR length = 'short';
```
- Using `AND`: when you need to satisify all criteria.
```SQL
SELECT *
FROM coats
WHERE color = 'yellow' AND length = 'short';
```
- Using `BETWEEN`:
```SQL
SELECT *
FROM coats
WHERE buttons BETWEEN 1 AND 5;
```

### Combining `OR` and `AND`
- Filter films released in 1994 or 1995, and certified PGorR
```SQL
SELECT *
FROM films
WHERE
	(release_year = 1994 OR release_year = 1995)
	AND (cert = 'PG' OR cert = 'R');
```

If we are working with multiple conditions we need to enclose each condition in parenthesis.

### Combining `BETWEEN` and `AND`
Filter films between 1994 and 2000:
- If we were to use **only `AND`**, we would do something like this:
```sql
SELECT title
FROM films
WHERE release_year >= 1994
	AND release_year <= 2000;
```
- We can also use `BETWEEN` and `AND`:
```sql
SELECT title
WHERE films
WHERE 
	release_year BETWEEN 1994 AND 2000;
```

# Filtering text

## Filtering for a text pattern: `LIKE`, `NOT LIKE`, `IN`


### `LIKE`
- Used to search for a pattern in a field 
- It uses the *wildcards* to accompany it
	- `%` match zero, one, or many characters
	- `_` match a single character

Using `LIKE` with `%`:
```sql 
SELECT name
FROM people
WHERE name LIKE 'Ade%';

/* output:

| name          |
|---------------|
|Adel Karam     |
|Aden Young     |
| ...           |

*/
```

Using `LIKE` with `_`:
```sql 
SELECT name
FROM people
WHERE name LIKE 'Ev_';

/* output:

| name   |
|--------|
|Eve     |
|Eva     |
| ...    |

*/
```
Note that in the above use case, someone called *Eva Mendes* for example would **not** show up, as `_` only allows for one character variation.

### `NOT LIKE`
We use `NOT LIKE` to find records that don't match the specified pattern.
```Sql
SELECT name
FROM people
WHERE name NOT LIKE `A.%`;

/* output:

| name          |
|---------------|
|Adel Karam     |
|Aden Young     |
| ...           |

*/
```

#### Wildcard positioning
- Find records with names ending in r:
```sql
SELECT name
FROM people
WHERE name LIKE `%r`;
```
- Find records with names where the third character is t:
```sql
SELECT name
FROM people
WHERE name LIKE `__t%`;
```


### `IN`
Using the `IN` operator we can filter based on a set/list of values.
```sql
-- get titles of films release in either years
SELECT title
FROM films
WHERE release_year IN (1999,2000,2001);

-- get titles of films in either French or German
SELECT title
FROM films
WHERE language IN ('German', 'French');
```


# NULL values

## Missing values
We have learned earlier that we can include or exclude null values in our counts based on:
- `COUNT(field_name)` includes only non-NAs
- `COUNT(*)` includes NAs too

We can have missing values for many different reasons, such as:
- Human error
- Information not available
- Unknown

## `IS NULL` & `IS NOT NULL`

We can check which records have NAs inside a certain field by using `WHERE` and `IS NULL`:
```SQL
-- Get all the records without a birthdate on file
SELECT name
FROM people
WHERE birthdate IS NULL;
```
We can also make sure to count only records without null values, using `WHERE` and `IS NOT NULL`:
```sql
-- Get all the records with a birthdate on file
SELECT name
FROM people
WHERE birthdate IS NOT NULL;
```


# Summarizing data

To summarize data in SQL we use **aggregate functions.**
- Aggregate functions return a single value
- Most common aggregate functions:
	- `AVG()`, `SUM()`, `MIN()`, `MAX()`, `COUNT()`

We can for example:
- Get the mean budget in our table:
```sql 
SELECT 
	AVG(budget)
FROM 
	films;
```
- Get the total budget of all films in our table:
```sql
SELECT 
	SUM(budget)
FROM 
	films;
```
- Get the lowest budget:
```sql
SELECT
	MIN(budget)
FROM
	films;
```
- Get the highest budget:
```sql
SELECT
	MAX(budget)
FROM
	films;
```

Note that `AVG()` and `SUM()` only work with **numerical data**, while `COUNT()`, `MIN()`, `MAX()` work with various data types.

For **non-numerical data** the logic is as follows:
- `MIN()` <-> `MAX()`; this could be:
	- Minimum <-> Maximum
	- Lowest <-> Highest
	- A <-> Z: running these on strings will be interpreted alphabetically
	- 1715 <-> 2022

# Summarizing subsets

## Using `WHERE` with aggregate functions
- `AVG()`
```SQL
SELECT 
	AVG(budget) AS avg_budget
FROM
	films
WHERE
	release_year >= 2010;
```

## `ROUND()`
```SQL
SELECT
	ROUND(AVG(budget), 2) AS avg_budget
FROM
	films
WHERE
	release_year >= 2010;
```

# Aliasing and arithmetic

In SQL, we can use arithmetic to execute calculations, such as `+`, `-`, `*` and `/`.
```sql
SELECT (3+1);

SELECT (3/3);

SELECT (12*3);

SELECT (2-1);
```

Note that SQL assumes that if we use INT in our operation, we want an INT back. So for example this operation would return 1, not the precise answer:
```sql
SELECT (4/3);
-- returns: |1|
```
If we were to use floats we would get a precise output:
```sql
SELECT (4.0/3.0);
-- returns: |1.333...|
```


# Sorting results

## `ORDER BY`
Used to sort results on one or more fields.
- Sorts in ascending order by default
```sql
SELECT
	title,
	budget
FROM 
	films
ORDER BY budget;
```
- We can use `ASC` to explicitly say it is in ascending order for better code readability:
```sql
SELECT
	title,
	budget
FROM 
	films
ORDER BY budget ASC;
```
- We can also use `DESC` to sort in descending order:
```sql
SELECT
	title,
	budget
FROM 
	films
ORDER BY budget DESC;
```
- We can improve our query by removing `null` values:
```sql
SELECT
	title,
	budget
FROM 
	films
WHERE budget IS NOT NULL
ORDER BY budget;
```

### Ordering by multiple fields

When using `ORDER BY` on multiple fields the order is:
- `ORDER BY field_one, field_two`
- Think of `field_two` as a tie-breaker
- We can utilize `ASC` and `DESC` to sort in different orders per field

### Order of execution
```sql 
SELECT                     -- 3
	title,
	budget
FROM                       -- 1 
	films
WHERE budget IS NOT NULL   -- 2
ORDER BY budget            -- 4
LIMIT 10;                  -- 5
```


# Grouping data

## `GROUP BY`
We can use `GROUP BY` to group our data by a specific field.
```SQL
SELECT
	certification,
	COUNT(title) AS title_count
FROM
	films
GROUP BY certification;
```
- Note we cannot add fields in the `SELECT` which are not either in the `GROUP BY` or in an aggregate function.

We can also `GROUP BY` multiple fields:
```sql
SELECT
	certification,
	language,
	COUNT(title) AS title_count
FROM
	films
GROUP BY
	certification,
	language;
```

We can combine `GROUP BY` with `ORDER BY`:
```SQL
SELECT
	certification,
	COUNT(title) AS title_count
FROM 
	films
GROUP BY 
	certification
ORDER BY
	title_count DESC;
```

### Order of execution
```sql
SELECT                                     -- 3
	certification,
	COUNT(title) AS title_count
FROM                                       -- 1
	films
GROUP BY                                   -- 2
	certification
ORDER BY                                   -- 4
	title_count DESC
LIMIT 10;                                  -- 5
```

# Filtering grouped data

## `HAVING`
We can conditionally filter our data by using `HAVING`
```SQL
SELECT
	release_year,
	COUNT(title) AS title_count
FROM 
	films
GROUP BY
	release_year
HAVING
	COUNT(title) > 10;
```

### Order of execution
```SQL
SELECT                                        -- 5
	certification,
	COUNT(title) AS title_count
FROM                                          -- 1
	films
WHERE                                         -- 2
	certification IN ('G', 'PG', 'PG-13')
GROUP BY                                      -- 3
	certification
HAVING                                        -- 4
	COUNT(title) > 500
ORDER BY                                      -- 6
	title_count DESC
LIMIT 3;                                      -- 7
```

### `HAVING` vs `WHERE`
`WHERE` filters individual records, `HAVING` filters grouped records.

Business questions examples:
- What films were released in the year 2000?
	- To answer this question we do not need grouping.
```SQL
SELECT
	title,
	certification,
	release_year
FROM 
	films
WHERE
	release_year = 2000
ORDER BY
	title;
```
- In what years was the average film duration over two hours?
	- This question does require grouping.
```SQL
SELECT
	release_year,
	AVG(duration) AS avg_duration
FROM
	films
GROUP BY 
	release_year
HAVING 
	AVG(duration) > 120;
```