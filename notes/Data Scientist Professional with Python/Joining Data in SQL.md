
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