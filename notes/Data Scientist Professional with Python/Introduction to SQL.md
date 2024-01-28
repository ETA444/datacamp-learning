
# Databases

## Relational databases

A relational database is a structured collection of data organized in tables, where information is stored in a way that establishes relationships between different pieces of data. This type of database is based on the principles of relational algebra and is designed to efficiently manage and retrieve information.

**Key Concepts:**

1. **Table:**
   - In a relational database, data is organized into tables, which are similar to spreadsheets. Each table consists of rows and columns, where each row represents a record, and each column represents a specific attribute or field.

2. **Record:**
   - A record is a complete set of information about a particular entity. Each row in a table is a record, and it contains values for each of the attributes defined by the columns.

3. **Column/Attribute:**
   - Columns, also known as attributes, define the specific types of data that can be stored in a table. Each column has a name and a data type that specifies the kind of information it can hold (e.g., integer, text, date).

4. **Primary Key:**
   - A primary key is a unique identifier for each record in a table. It ensures that each row can be uniquely identified and is used to establish relationships between tables.

5. **Foreign Key:**
   - A foreign key is a column in a table that refers to the primary key in another table. It establishes a link between the two tables, creating relationships and enabling data integrity.

6. **Relationships:**
   - Relationships define how tables are connected to each other. Common types include one-to-one, one-to-many, and many-to-many relationships, reflecting how entities in the real world are related.

7. **Normalization:**
   - Normalization is the process of organizing data in a database to eliminate redundancy and improve data integrity. It involves breaking down large tables into smaller, related tables.

8. **SQL (Structured Query Language):**
   - SQL is the language used to interact with relational databases. It allows users to perform operations such as querying, updating, inserting, and deleting data from tables.

## Tables

- Table rows are referred to as **records**
- Table columns are referred to as **fields**

Fields are set at database creation; there is no limit to the number of records

### Good table manners

- **Table names** should:
	- be lowercase
	- have no spaces (use underscores)
	- refer to a collective group or be plural

It is better to have more tables than more fields and records in one table.

### Laying the table

#### Records
A **record** is a row that holds data on an individual observation.

#### Fields
A **field** is a column that holds one piece of information about all records.

Field names should:
- be lowercase
- have no spaces
- be singular
- be different from other field names
- be different than table names

#### Identifiers
- **Unique identifiers** are used to identify records in a table.
	- Also called **primary key**

## Database schema & Data types

### Schema
Databases consist of multiple tables, and a **schema illustrates their structure, relationships, and field data types**. 
- Database schemas show data types for each field in all tables, and they also show relationships between tables.

The schema information is physically stored on a server's hard disk, which can be any computer configured to provide services over a network but is typically a powerful centralized computer. 

### Data types
Data within a database is organized into tables, with **each field assigned a data type** based on its content, such as numbers, text, or dates.

Data types differ in space and storage requirements depending on how long they are. Take a look at patrons table to see the most common data types:

- **VARCHAR:** used for storing "strings" or a sequence of characters such as letters or punctuation like the names field
- **INT:** used for integers or whole numbers, such as the years in the member_year column
- **NUMERIC:** stores "floats" or numbers that include a fractional part, such as the $2.05 Jasmin owes in fines


# Introducing queries 

## Keywords

Keywords are reserved words for operations.

These are some of the most common SQL keywords:

1. **SELECT:**
   - Used to retrieve data from one or more tables.
   ```sql
   SELECT column1, column2 FROM your_table;
   ```

2. **FROM:**
   - Specifies the table from which to retrieve data.
   ```sql
   SELECT column1, column2 FROM your_table;
   ```

3. **WHERE:**
   - Filters data based on specified conditions.
   ```sql
   SELECT column1, column2 FROM your_table WHERE condition;
   ```

4. **ORDER BY:**
   - Sorts the result set in ascending or descending order.
   ```sql
   SELECT column1, column2 FROM your_table ORDER BY column1 ASC;
   ```

5. **GROUP BY:**
   - Groups rows that have the same values into summary rows.
   ```sql
   SELECT column1, COUNT(*) FROM your_table GROUP BY column1;
   ```

6. **HAVING:**
   - Filters the results of a GROUP BY clause based on specified conditions.
   ```sql
   SELECT column1, COUNT(*) FROM your_table GROUP BY column1 HAVING COUNT(*) > 1;
   ```

7. **INSERT INTO:**
   - Adds new records to a table.
   ```sql
   INSERT INTO your_table (column1, column2) VALUES (value1, value2);
   ```

8. **UPDATE:**
   - Modifies existing records in a table.
   ```sql
   UPDATE your_table SET column1 = value1 WHERE condition;
   ```

9. **DELETE:**
   - Removes records from a table based on specified conditions.
   ```sql
   DELETE FROM your_table WHERE condition;
   ```

10. **JOIN:**
    - Combines rows from two or more tables based on related columns.
    ```sql
    SELECT column1, column2 FROM table1 INNER JOIN table2 ON table1.column = table2.column;
    ```

11. **DISTINCT:**
    - Returns unique values in a specified column.
    ```sql
    SELECT DISTINCT column1 FROM your_table;
    ```

12. **AS:**
    - Renames a column or table using an alias.
    ```sql
    SELECT column1 AS new_name FROM your_table;
    ```

13. **COUNT:**
    - Returns the number of rows in a result set or the number of occurrences of a value in a column.
    ```sql
    SELECT COUNT(*) FROM your_table;
    ```


## Writing queries

### Aliasing

In SQL you can use what is called aliasing to rename columns within your query and output.

```sql
SELECT 
	name AS first_name, 
	year_hired
FROM 
	employees;
```


### Selecting DISTINCT records

You can select only unique values in a column by using the `DISTINCT` keyword along with `SELECT`.

```sql
SELECT DISTINCT
	year_hired
FROM
	employees;
```

#### Unique combinations of multiple fields

To select unique combinations of multiple fields you use `DISTINCT` on multiple fields in the query:
```sql
SELECT DISTINCT
	dept_id,
	year_hired
FROM
	employees;
```


## Views

**Views** are virtual tables that are the result of a saved SQL `SELECT` statement.
- When accessed, views automatically update in response to updates in the underlying data.
- Views are not saved in the database but serve as saved queries to specific information.

**To create a view:**
```sql
-- Create a view/virtual table: employee_hire_year
CREATE VIEW employee_hire_year AS
SELECT
	id,
	name,
	year_hired
FROM
	employees;
```


# SQL flavors

There are various variations of SQL.
- Both free and paid
- All used with relational databases
- Vast majority of keywords are the same
- All must follow universal standards
- Only additional keywords define the variations, not a difference in the basic syntax

## Two popular SQL flavors

### PostgreSQL
- Free and open-source relational database system
- Created at the University of California, Berkeley
- "PostgreSQL" refers to both the PostgreSQL database system and the associated SQL variation

### SQL Server / T-SQL
- Has free and paid versions
- Created by Microsoft
- T-SQL is Microsoft's SQL flavor, used with SQL Server databases

### Comparing the two

#### Limiting the output to *n* rows
- PostgreSQL: **LIMIT *n***
```sql
SELECT 
	id,
	name
FROM employees
LIMIT 2;
```
- SQL Server: SELECT **TOP(*n*)**
```sql
SELECT TOP(2)
	id,
	name
FROM employees;
```