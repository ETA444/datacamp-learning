
# The ins and outs of INNER JOINs

A **key** is a field that uniquely identifies each records, this is what we use to join tables on.

## `INNER JOIN`

The `INNER JOIN` looks for records in both tables which match on a given field.
- It retrieves **only the rows where there is a match** in both tables based on the specified key in the `ON` clause.
```sql
SELECT 
	column1, 
	column2,
	table1.column3 -- *Note*
FROM 
	table1
INNER JOIN table2
	ON table1.key = table2.key;

```
- **Note:** when selecting fields that exist in both tables we need to specify `table1.column3` as `column3` exists in both tables.

### Aliasing tables

For ease we can alias tables:
```sql
SELECT 
	column1, 
	column2,
	t1.column3
FROM 
	table1 AS t1
INNER JOIN table2 AS t2
	ON t1.key = t2.key;
```
- Aliases can be used in the `table1.column3` syntax in `SELECT` and `ON` clauses.

### `USING`
To make the above query even more concise we utilize `USING`:
```sql
SELECT 
	column1, 
	column2,
	t1.column3
FROM 
	table1 AS t1
INNER JOIN table2 AS t2
	USING(key); -- < --
```
With `USING` we only need to specify the key field in brackets instead of the previous syntax format.

# Defining relationships

In SQL, the relationships between tables are typically classified into three main types based on how data is shared between the tables. These relationships are fundamental concepts in relational databases:

1. **One-to-One Relationship:**
   - In a one-to-one relationship, each row in the first table is associated with at most one row in the second table, and vice versa. This relationship is less common but is used when two tables have a unique and singular connection.

   ```sql
   -- Example: Employees and EmployeeDetails
   CREATE TABLE Employees (
       EmployeeID INT PRIMARY KEY,
       EmployeeName VARCHAR(255)
   );

   CREATE TABLE EmployeeDetails (
       EmployeeID INT PRIMARY KEY,
       Salary DECIMAL(10,2),
       CONSTRAINT FK_EmployeeDetails_Employees
           FOREIGN KEY (EmployeeID) REFERENCES Employees (EmployeeID)
   );
   ```

2. **One-to-Many Relationship:**
   - In a one-to-many relationship, each row in the first table can be associated with multiple rows in the second table, but each row in the second table is associated with at most one row in the first table. This is the most common type of relationship.

   ```sql
   -- Example: Customers and Orders
   CREATE TABLE Customers (
       CustomerID INT PRIMARY KEY,
       CustomerName VARCHAR(255)
   );

   CREATE TABLE Orders (
       OrderID INT PRIMARY KEY,
       OrderDate DATE,
       CustomerID INT,
       CONSTRAINT FK_Orders_Customers
           FOREIGN KEY (CustomerID) REFERENCES Customers (CustomerID)
   );
   ```

3. **Many-to-Many Relationship:**
   - In a many-to-many relationship, each row in the first table can be associated with multiple rows in the second table, and vice versa. This relationship requires an intermediary table, often referred to as a junction or link table, to manage the connections.

   ```sql
   -- Example: Students and Courses (with an intermediary Enrollment table)
   CREATE TABLE Students (
       StudentID INT PRIMARY KEY,
       StudentName VARCHAR(255)
   );

   CREATE TABLE Courses (
       CourseID INT PRIMARY KEY,
       CourseName VARCHAR(255)
   );

   CREATE TABLE Enrollment (
       StudentID INT,
       CourseID INT,
       CONSTRAINT FK_Enrollment_Students
           FOREIGN KEY (StudentID) REFERENCES Students (StudentID),
       CONSTRAINT FK_Enrollment_Courses
           FOREIGN KEY (CourseID) REFERENCES Courses (CourseID),
       PRIMARY KEY (StudentID, CourseID)
   );
   ```

# Multiple joins

## Joins on joins
```SQL
SELECT *
FROM left_table
INNER JOIN right_table
	ON left_table.id = right_table.id
INNER JOIN another_table
	ON left_table.id = another_table.id
```

## Joining on multiple keys
```sql
SELECT *
FROM left_table AS l
INNER JOIN right_table AS r
	ON l.id = r.id
	AND l.date = r.date;
```


# OUTER JOINS
## `LEFT` and `RIGHT JOIN`s

### `LEFT JOIN`

`LEFT JOIN` will return all records in the left table, and those records in the right table that match on the joining field provided.

#### Syntax
```SQL
-- Suppose we want all prime ministers and all presidents for countries that have them:
SELECT
	p1.country,
	prime_minister,
	president
FROM prime_ministers AS p1
LEFT JOIN presidents AS p2
	USING(country);
```
- `LEFT JOIN` can also be written as `LEFT OUTER JOIN`.

### `RIGHT JOIN`

`RIGHT JOIN` will return all records in the right table, and those records in the left table that match on the joining field provided.

#### Syntax
```SQL
-- Suppose we want all prime ministers and all presidents for countries that have them:
SELECT *
FROM table1 AS t1
RIGHT JOIN table2 AS t2
	USING(country);
```
- `RIGHT JOIN` can also be written as `RIGHT OUTER JOIN`.

## `FULL JOIN`s

A `FULL JOIN` combines a `LEFT JOIN` and a `RIGHT JOIN`.

### Syntax
```sql
SELECT
	l.id AS l_id,
	r.id AS r_id,
	l.val AS l_val,
	r.val AS r_val
FROM left_table AS l
FULL JOIN right_table AS r
	USING(id);
```
- `FULL JOIN` can also be written as `FULL OUTER JOIN`.

# `CROSS JOIN`s

`CROSS JOIN` creates all possible combinations of two tables.

## Syntax
In contrast to other type of joins, we do not specify `ON` when using `CROSS JOIN`:
```SQL
SELECT
	id1,
	id2
FROM table1
CROSS JOIN table2;
```


# Self joins
- **Self joins** are tables joined with themselves
- They can be used to compare parts of the same table

## Syntax
There is no syntax called `SELF JOIN`, we use other joins to achieve the self join.
```sql
SELECT
	p1.country AS country1,
	p2.country AS country2,
	p1.continent
FROM prime_ministers AS p1
INNER JOIN prime_minister AS p2
	ON p1.continent = p2.continent
		AND p1.country <> p2.country;
```


# Set theory for SQL Joins

## Venn diagrams and set theory
### `UNION` diagram
- `UNION` takes two tables as input, and returns all records from both tables (without duplicating records occurring in both)
```sql
-- Table 1: TableA
| id | columnA |
|----|---------|
| 1  | ValueA1 |
| 2  | ValueA2 |
| 3  | ValueA3 |
| 6  | ValueA6 |  -- Same record as in TableB

-- Table 2: TableB
| id | columnB |
|----|---------|
| 1  | ValueB1 |
| 4  | ValueB4 |
| 5  | ValueB5 |
| 6  | ValueA6 |  -- Same record as in TableA

-- UNION Result
| id | columnA |
|----|---------|
| 1  | ValueA1 |
| 2  | ValueA2 |
| 3  | ValueA3 |
| 4  | ValueB4 |
| 5  | ValueB5 |
| 6  | ValueA6 | -- Only one is kept
```
#### Syntax: `UNION`
```sql
SELECT *
FROM table1
UNION
SELECT *
FROM table2
```

### `UNION ALL` diagram
- `UNION ALL` takes two tables and returns all records from both tables, **including duplicates**.
```sql
-- Table 1: TableA
| id | columnA |
|----|---------|
| 1  | ValueA1 |
| 2  | ValueA2 |
| 3  | ValueA3 |
| 6  | ValueA6 |  -- Same record as in TableB

-- Table 2: TableB
| id | columnB |
|----|---------|
| 1  | ValueB1 |
| 4  | ValueB4 |
| 5  | ValueB5 |
| 6  | ValueA6 |  -- Same record as in TableA

-- UNION Result
| id | columnA |
|----|---------|
| 1  | ValueA1 |
| 2  | ValueA2 |
| 3  | ValueA3 |
| 4  | ValueB4 |
| 5  | ValueB5 |
| 6  | ValueA6 | -- Both are kept
| 6  | ValueA6 | -- Both are kept
```
#### Syntax: `UNION ALL`
```sql
SELECT *
FROM table1
UNION ALL
SELECT *
FROM table2
```

### Notes
- Both `UNION` and `UNION ALL` are used to **stack** data vertically, this is a fundamental difference from `JOIN`s.
- `UNION` and `UNION ALL` are called set operators.
- When using these we:
	- Need the same field data types
	- Retain field names from the left tables regardless of aliases

#### Example
```sql
SELECT
	monarch as leader,
	country
FROM monarchs
UNION
SELECT
	prime_minister, -- this will go under leader
	country
FROM prime_ministers
ORDER BY
	country,
	leader
LIMIT 10;
```


# At the `INTERSECT`

The `INTERSECT` set operator in SQL retrieves the common rows that appear in the result sets of two or more SELECT statements, effectively returning only the overlapping rows shared between the result sets. It is used to find the intersection of data sets, ensuring that only rows present in all specified SELECT statements are included in the final result.

Here's a simple example using two tables and the `INTERSECT` set operator:

```sql
-- Table 1: TableA
| id | columnA |
|----|---------|
| 1  | ValueA1 |
| 2  | ValueA2 |
| 3  | ValueA3 |

-- Table 2: TableB
| id | columnB |
|----|---------|
| 2  | ValueB2 |
| 3  | ValueA3 |
| 4  | ValueB4 |

-- INTERSECT Result
| id | columnA |
|----|---------|
| 3  | ValueA3 |
```
- `TableA` and `TableB` have commonality on the `id` column.
- The `INTERSECT` operation returns the rows that are common to both tables, based on the `id` column.
- The result includes only the row where `id = 3` because it is the only row present in both `TableA` and `TableB`.

This demonstrates how `INTERSECT` helps find the commonality between two sets of data, allowing you to identify rows that exist in both result sets.

## Syntax: `INTERSECT`
```sql
SELECT id, columnA
FROM TableA
INTERSECT
SELECT id, columnB
FROM TableB;
```

# `EXCEPT`

The `EXCEPT` set operator in SQL is used to retrieve the distinct rows from the result set of the first SELECT statement that do not appear in the result set of the second SELECT statement. It effectively subtracts the rows from the second result set, providing a set difference and highlighting the unique elements in the first result set.

Here's a simple example using two tables and the `EXCEPT` set operator:

```sql
-- Table 1: TableA
| id | columnA |
|----|---------|
| 1  | ValueA1 | -- both id and columnA unique
| 2  | ValueA2 | -- columnA values unique
| 3  | ValueA3 | -- Exists in both tables

-- Table 2: TableB
| id | columnB |
|----|---------|
| 2  | ValueB2 |
| 3  | ValueA3 | -- Exists in both tables
| 4  | ValueB4 | 

-- EXCEPT Result
| id | columnA |
|----|---------|
| 1  | ValueA1 |
| 2  | ValueA2 |
```

This illustrates how `EXCEPT` helps identify the unique rows in the first result set that do not have corresponding entries in the second result set.

## Syntax
```sql
SELECT id, columnA
FROM TableA
EXCEPT
SELECT id, columnB
FROM TableB;
```

This query uses the `EXCEPT` set operator to retrieve the distinct rows from the `TableA` result set that do not appear in the `TableB` result set. It returns the rows that are unique to `TableA`.