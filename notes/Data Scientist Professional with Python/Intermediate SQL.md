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

