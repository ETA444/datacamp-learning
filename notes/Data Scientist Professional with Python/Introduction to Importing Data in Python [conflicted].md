***
## open()

The `open()` function in Python is used to open a file and return a file object. It provides a way to read, write, or manipulate files in different modes.

**Function syntax:**
```python
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
```

**Parameters:**
- `file`: Specifies the path or name of the file to be opened.
- `mode` (optional): Specifies the mode in which the file will be opened. It can be `'r'` for reading (default), `'w'` for writing, `'a'` for appending, `'x'` for exclusive creation, `'b'` for binary mode, `'t'` for text mode, and more.
- `buffering` (optional): Specifies the buffering policy. If set to 0 or negative, buffering is disabled. If set to 1 or greater, it specifies the buffer size.
- `encoding` (optional): Specifies the character encoding to be used for reading or writing text files.
- `errors` (optional): Specifies how encoding and decoding errors should be handled.
- `newline` (optional): Specifies the newline character(s) to use when reading or writing text files.
- `closefd` (optional): Specifies whether to close the underlying file descriptor when the file object is closed. Default is `True`.
- `opener` (optional): Specifies a custom opener function for opening the file.

**Example of use:**
```python
# Open a file for reading
file = open('example.txt', 'r')

# Read the contents of the file
content = file.read()

# Close the file
file.close()
```

In the example, `open('example.txt', 'r')` opens the file `'example.txt'` in read mode. The `read()` method is then used to read the contents of the file into the `content` variable. Finally, the `close()` method is called to close the file and release any system resources associated with it.

The `open()` function is a fundamental tool for file handling in Python. It allows you to interact with files by providing access to read, write, and modify their contents. After opening a file, you can perform various operations such as reading data, writing data, seeking to specific positions, or iterating over the lines. It is important to close the file using the `close()` method to free up system resources and ensure proper file handling.


##### Reading a text file
```python
filename = 'huck_finn.txt'
file = open(filename, mode='r')
text = file.read()
file.close()

# to automatically close files use 'with' statement
```


```python
print(text)
```
![Pasted image 20230719191729](/images/Pasted%20image%2020230719191729.png)

##### Writing to a file
```python
filename = 'huck_finn.txt'
file = open(filename, mode='w')
file.close()

# to automatically close files use 'with' statement
```

### Exercises

##### Exercise 1
```python
# Open a file: file
file = open('moby_dick.txt', mode='r')

# Print it
print(file.read())
```
![Pasted image 20230719194411](/images/Pasted%20image%2020230719194411.png)

```python
# Check whether file is closed
print(file.closed)
```
![Pasted image 20230719194432](/images/Pasted%20image%2020230719194432.png)

```python
# Close file
file.close()

# Check whether file is closed
print(file.closed)
```
![Pasted image 20230719194444](/images/Pasted%20image%2020230719194444.png)

***
## with

The `with` statement in Python is used for efficient and safe management of resources, such as files. It provides a convenient way to work with resources that need to be explicitly opened and closed, ensuring that they are properly cleaned up even if an exception occurs.

**Syntax:**
```python
with expression as target:
    # Code block to work with the resource
```

**Example of use (file handling):**
```python
# Open a file using the 'with' statement
with open('example.txt', 'r') as file:
    # Perform operations on the file within the code block
    content = file.read()
    print(content)
# The file is automatically closed when exiting the 'with' block
```

In the example, the `with open('example.txt', 'r') as file:` statement is used to open the file `'example.txt'` for reading. The file object is assigned to the variable `file`. Within the indented code block, operations can be performed on the file, such as reading its contents. Once the code block is exited, the file is automatically closed, even if an exception occurs.

The `with` statement is useful because it ensures proper resource cleanup, such as closing a file, when the block of code inside it is exited, whether normally or due to an exception. It eliminates the need to explicitly call `.close()` on the file object, reducing the chances of resource leaks and improving code readability. It is commonly used with file handling, database connections, network sockets, and other resources that need to be properly managed and released.

```python
with open('huck_finn.txt', 'r') as file:
	print(file.read())
```
![Pasted image 20230719192017](/images/Pasted%20image%2020230719192017.png)

### Exercises

##### Exercise 1
```python
# Read & print the first 3 lines
with open('moby_dick.txt', 'r') as file:
	print(file.readline())
	print(file.readline())
	print(file.readline())
```
![Pasted image 20230719194844](/images/Pasted%20image%2020230719194844.png)


## .readline()

The `.readline()` method in Python is used to read a single line from a file. It reads the characters from the current position of the file pointer until it encounters a newline character (`'\n'`) or reaches the end of the line.

**Method syntax:**
```python
file.readline(size=-1)
```

**Parameters:**
- `size` (optional): Specifies the maximum number of characters to read from the line. If not specified or set to `-1` (default), it reads the entire line.

**Example of use:**
```python
# Open a file for reading
file = open('example.txt', 'r')

# Read the first line from the file
line = file.readline()

# Close the file
file.close()

# Display the line
print(line)
```

In the example, `file.readline()` reads the first line from the file into the `line` variable. The `print(line)` statement displays the contents of the line.

The `.readline()` method allows you to read individual lines from a file. It can be used to process large text files line by line or extract specific information from a file. Each call to `.readline()` moves the file pointer to the next line, so subsequent calls read subsequent lines from the file.

After reading the lines, remember to close the file using the `.close()` method to release system resources and ensure proper file handling. Alternatively, you can use the `with` statement, which automatically takes care of closing the file for you.
***
## np.loadtxt()

The `np.loadtxt()` function in NumPy is used to load data from a text file into a NumPy array.

**Function syntax:**
```python
np.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
```

**Parameters:**
- `fname`: Specifies the file name or file object from which to load the data.
- `dtype` (optional): Specifies the data type of the resulting array. By default, it is set to `float`.
- `comments` (optional): Specifies the character or characters used to indicate comments in the text file. Lines starting with the comment character(s) will be ignored. By default, it is set to `'#'`.
- `delimiter` (optional): Specifies the character or characters used to separate values in the text file. By default, it is any whitespace.
- `converters` (optional): Specifies a dictionary mapping column indices or names to converter functions, which can be used to perform custom data conversions during loading.
- `skiprows` (optional): Specifies the number of rows at the beginning of the file to skip before loading the data.
- `usecols` (optional): Specifies which columns to load from the file. It can be a single column index, a sequence of column indices, or a sequence of column names.
- `unpack` (optional): Specifies whether to transpose the loaded data. If `True`, the resulting array will be transposed, with each column as a separate array. By default, it is set to `False`.
- `ndmin` (optional): Specifies the minimum number of dimensions of the resulting array. By default, it is set to `0`, which means the array will have the minimum number of dimensions needed to represent the data.

**Example of use:**
```python
import numpy as np

# Load data from a text file
data = np.loadtxt('data.txt', delimiter=',')

# Display the loaded data
print(data)
```

In the example, `np.loadtxt('data.txt', delimiter=',')` is used to load data from the file `'data.txt'`. The data in the file is assumed to be delimited by commas. The resulting array `data` contains the loaded data.

The `np.loadtxt()` function is useful for loading numerical data from text files into NumPy arrays. It supports various options to handle different file formats, delimiters, comments, and conversions. You can customize the loading process by specifying the appropriate parameters according to your specific data file. The resulting array can then be used for various numerical computations and data analysis tasks in NumPy.


```python
import numpy as np
filename = 'MNIST.txt'
data = np.loadtxt(filename, delimeter = ',')
data
```
![Pasted image 20230719193036](/images/Pasted%20image%2020230719193036.png)

##### Customizing the NumPy import
- `skiprows` (optional): Specifies the number of rows at the beginning of the file to skip before loading the data.
```python
import numpy as np
filename = 'MNIST_header.txt'
data = np.loadtxt(filename, delimiter = ',', skiprows = 1)
print(data)
```
![Pasted image 20230719193300](/images/Pasted%20image%2020230719193300.png)

- `usecols` (optional): Specifies which columns to load from the file. It can be a single column index, a sequence of column indices, or a sequence of column names.
```python
import numpy as np
filename = 'MNIST_header.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols[0,2])
print(data)
```
![Pasted image 20230719193453](/images/Pasted%20image%2020230719193453.png)

- `dtype` (optional): Specifies the data type of the resulting array. By default, it is set to `float`.
```python
data = np.loadtxt(filename, delimiter = ',', dtype = str)
```

![Pasted image 20230719193916](/images/Pasted%20image%2020230719193916.png)

### Exercises 

##### Exercise 1
```python
# Import package
import numpy as np

# Assign filename to variable: file
file = 'digits.csv'

# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')

# Print datatype of digits
print(type(digits))

# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()

```
![Pasted image 20230719205725](/images/Pasted%20image%2020230719205725.png)
![Pasted image 20230719205739](/images/Pasted%20image%2020230719205739.png)

##### Exercise 2
```python
# Import numpy
import numpy as np

# Assign the filename: file
file = 'digits_header.txt'

# Load the data: data - tab delimited
data = np.loadtxt(file, delimiter='\t', skiprows=1, usecols=[0,2])

# Print data
print(data)
```
![Pasted image 20230719210013](/images/Pasted%20image%2020230719210013.png)

##### Exercise 3
```python
# Assign filename: file
file = 'seaslug.txt'

# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)

# Print the first element of data
print(data[0])

# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)

# Print the 10th element of data_float
print(data_float[9])

# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()

```
![Pasted image 20230719210225](/images/Pasted%20image%2020230719210225.png)
![Pasted image 20230719210239](/images/Pasted%20image%2020230719210239.png)

##### Exercise 4
```python
# Assign the filename: file
file = 'titanic.csv'

# Import file using np.recfromcsv: d
d = np.recfromcsv(file, delimiter = ',', names=True, dtype=None)

# Print out first three entries of d
print(d[:3])

```
![Pasted image 20230719211247](/images/Pasted%20image%2020230719211247.png)


## np.genfromtxt()

The `np.genfromtxt()` function in NumPy is used to load data from a text file into a NumPy array, providing more flexibility than `np.loadtxt()`. It can handle missing values, different data types, and more complex file formats.

**Function syntax:**
```python
np.genfromtxt(fname, delimiter=None, dtype=float, names=True, skip_header=0, filling_values=None, usecols=None)
```

**Parameters:**
- `fname`: Specifies the file name or file object from which to load the data.
- `delimiter` (optional): Specifies the character or characters used to separate values in the text file. By default, it is any whitespace.
- `dtype` (optional): Specifies the data type of the resulting array. By default, it is `float`.
- `names` (optional): Specifies whether to treat the first row as column names. If `True`, the resulting array will have named columns based on the first row.
- `skip_header` (optional): Specifies the number of rows at the beginning of the file to skip before loading the data.
- `filling_values` (optional): Specifies the values to be used for missing data.
- `usecols` (optional): Specifies which columns to load from the file. It can be a single column index, a sequence of column indices, or a sequence of column names.

**Example of use:**
```python
import numpy as np

# Load data from a text file
data = np.genfromtxt('data.txt', delimiter=',', dtype=float, names=True)

# Display the loaded data
print(data)
```

In the example, `np.genfromtxt('data.txt', delimiter=',', dtype=float, names=True)` is used to load data from the file `'data.txt'`. The data in the file is assumed to be delimited by commas. The resulting array `data` contains the loaded data, with named columns based on the first row.

The `np.genfromtxt()` function is more powerful than `np.loadtxt()` as it can handle various data types, missing values, and more complex file formats. It provides greater flexibility in loading data from text files and is commonly used in data analysis tasks involving structured data.


## np.recfromcsv

The `np.recfromcsv()` function is used to load data from a CSV (comma-separated values) file into a structured NumPy record array.

**Function Syntax:**
```python
np.recfromcsv(fname, delimiter=',', names=True, dtype=None)
```

**Parameters:**
- `fname`: Specifies the file name or file object from which to load the data.
- `delimiter` (optional): Specifies the character or characters used to separate values in the CSV file. The default delimiter is a comma (',').
- `names` (optional): Specifies whether to treat the first row as column names. If `True`, the resulting record array will have named fields based on the column names. The default is `True`.
- `dtype` (optional): Specifies the data type of the resulting record array. If not provided, the data types will be inferred automatically.

**Example Usage:**
```python
import numpy as np

# Load data from a CSV file as a record array
data = np.recfromcsv('data.csv', delimiter=',', names=True)

# Display the loaded data
print(data)
```

In this example, the `np.recfromcsv()` function is used to load data from the CSV file `'data.csv'`. The `names=True` parameter indicates that the first row of the CSV file contains column names, and the resulting record array will have named fields based on those column names.



##### Importing using pandas

```python
import pandas as pd
filename = 'wineequality-red.csv'
data = pd.read_csv(filename)
data.head()
```
![Pasted image 20230719193844](/images/Pasted%20image%2020230719193844.png)

### Exercises

##### Exercise 1
```python
# Import pandas as pd
import pandas as pd

# Assign the filename: file
file = 'titanic.csv'

# Read the file into a DataFrame: df
df = pd.read_csv(file)

# View the head of the DataFrame
print(df.head())
```
![Pasted image 20230719211413](/images/Pasted%20image%2020230719211413.png)

##### Exercise 2
```python
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Assign filename: file
file = 'titanic_corrupt.txt'

# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values=['Nothing'])

# Print the head of the DataFrame
print(data.head())

# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[['Age']])
plt.xlabel('Age (years)')
plt.ylabel('count')
plt.show()
```
![Pasted image 20230719212203](/images/Pasted%20image%2020230719212203.png)
![Pasted image 20230719212211](/images/Pasted%20image%2020230719212211.png)


## type()

The `type()` function in Python is used to determine the type of an object. It returns the type of the object as a Python class.

**Function syntax:**
```python
type(object)
```

**Parameters:**
- `object`: Specifies the object for which you want to determine the type.

**Example of use:**
```python
# Check the type of different objects
x = 5
y = "Hello"
z = [1, 2, 3]

print(type(x))  # Output: <class 'int'>
print(type(y))  # Output: <class 'str'>
print(type(z))  # Output: <class 'list'>
```

In the example, `type(x)`, `type(y)`, and `type(z)` are used to determine the types of the objects `x`, `y`, and `z`, respectively. The `print()` function is used to display the resulting types.

The `type()` function is useful when you need to determine the type of an object at runtime. It can be used to conditionally check the type of an object, perform type-specific operations, or debug and troubleshoot code by verifying the types of objects. It is a fundamental tool for understanding the nature of objects and their behavior in Python.

***
## pickle

The `pickle` module in Python is used for object serialization and deserialization. It allows you to convert complex Python objects, such as data structures, into a byte stream and store them in a file or transmit them over a network. The serialized object can be retrieved later and reconstructed back into its original form.

**Serialization** refers to the process of converting an object into a byte stream, which can be stored or transmitted.

**Deserialization** refers to the process of reconstructing an object from a serialized byte stream.

The `pickle` module provides the following main functions:

- `pickle.dump(obj, file)`: Serializes the object `obj` and writes the byte stream to the file object `file`.
- `pickle.dumps(obj)`: Serializes the object `obj` and returns the byte stream as a string.
- `pickle.load(file)`: Reads a byte stream from the file object `file` and reconstructs the original object.
- `pickle.loads(bytes)`: Reconstructs an object from a serialized byte stream `bytes`.

**Example:**

```python
import pickle

# Serialize an object and write it to a file
data = [1, 2, 3, 4, 5]
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

# Deserialize the object from the file
with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Display the loaded data
print(loaded_data)
```

In this example, a list `data` is serialized using `pickle.dump()` and written to a file named `'data.pkl'`. Later, the object is deserialized using `pickle.load()` to retrieve the original data from the file.

The `pickle` module is useful for preserving the state of Python objects, sharing data between different Python programs, or storing data for later use. However, caution should be exercised when unpickling data from untrusted sources, as it can execute arbitrary code.

```python
import pickle 
with open('pickled_fruit.pkl', 'rb') as file:
	data = pickle.load(file)
print(data)
```
![Pasted image 20230720102203](/images/Pasted%20image%2020230720102203.png)

##### Exercise 1
```python
# Import pickle package
import pickle

# Open pickle file and load data: d
with open('data.pkl', 'rb') as file:
    d = pickle.load(file)

# Print d
print(d)

# Print datatype of d
print(type(d))
```
![Pasted image 20230720105934](/images/Pasted%20image%2020230720105934.png)

***
## pd.ExcelFile()

The `pd.ExcelFile()` function in pandas is used to create an ExcelFile object, which represents an Excel file. This object can then be used to extract data from specific sheets within the Excel file.

**Function syntax:**
```python
pd.ExcelFile(io, engine=None)
```

**Parameters:**
- `io`: Specifies the path, file-like object, or URL of the Excel file to be opened.
- `engine` (optional): Specifies the engine to use for reading the Excel file. By default, it is `None`, which means the engine will be automatically determined based on the available options.

**Example of use:**
```python
import pandas as pd

# Create an ExcelFile object
excel_file = pd.ExcelFile('data.xlsx')

# Get the sheet names in the Excel file
sheet_names = excel_file.sheet_names

# Read data from a specific sheet
data = excel_file.parse('Sheet1')

# Display the loaded data
print(data)
```

In this example, `pd.ExcelFile('data.xlsx')` creates an ExcelFile object `excel_file` representing the Excel file `'data.xlsx'`. The `sheet_names` attribute can be used to obtain a list of sheet names present in the Excel file. The `parse()` method is then used to read data from a specific sheet, such as `'Sheet1'`, into a DataFrame.

The `pd.ExcelFile()` function provides an efficient way to work with Excel files in pandas by allowing selective loading of specific sheets. It provides flexibility in accessing and manipulating data stored in different sheets within an Excel file.

### Importing Spreadsheets
```python
import pandas as pd
filename = 'urbanpop.xlsx'
data = pd.ExcelFile(filename)
print(data.sheet_names)
```
![Pasted image 20230720102552](/images/Pasted%20image%2020230720102552.png)

```python
df1 = data.parse('1960-1966') # sheet name variant (str)
df2 = data.parse(0) # sheet index variant (float)
```

## .sheet_names

The `.sheet_names` attribute in pandas is used to retrieve a list of sheet names present in an Excel file.

**Example:**
```python
import pandas as pd

# Create an ExcelFile object
excel_file = pd.ExcelFile('data.xlsx')

# Get the sheet names
sheet_names = excel_file.sheet_names

# Display the sheet names
print(sheet_names)
```

In this example, `excel_file.sheet_names` retrieves the list of sheet names from the `excel_file` ExcelFile object. The resulting `sheet_names` variable contains the names of all the sheets present in the Excel file.

By accessing the `.sheet_names` attribute, you can obtain a list of available sheets and use them to extract data from specific sheets using methods like `parse()` or `read_excel()`. This feature allows you to selectively load and process data from different sheets within an Excel file using pandas.

## .parse()

The `.parse()` method in pandas is used to extract data from a specific sheet of an Excel file, given the sheet name or sheet index.

**Method syntax:**
```python
excel_file.parse(sheet, header=None, names=None, index_col=None, ...)
```

**Parameters:**
- `sheet`: Specifies the sheet to extract data from. It can be the sheet name (as a string) or the sheet index (as an integer).
- `header` (optional): Specifies the row number (0-indexed) to be used as the column names. By default, `header=None` uses the default column names.
- `names` (optional): Specifies a list of column names to use, instead of the names inferred from the data.
- `index_col` (optional): Specifies the column(s) to be used as the index of the resulting DataFrame. It can be a single column or a list of columns.
- Additional parameters such as `skiprows`, `nrows`, `usecols`, and more can be used to further customize the data extraction process.

**Example of use:**
```python
import pandas as pd

# Create an ExcelFile object
excel_file = pd.ExcelFile('data.xlsx')

# Extract data from a specific sheet
data = excel_file.parse('Sheet1')

# Display the extracted data
print(data)
```

In this example, `excel_file.parse('Sheet1')` extracts data from the sheet named `'Sheet1'` in the Excel file represented by the `excel_file` ExcelFile object. The resulting data is returned as a DataFrame.

By using the `.parse()` method, you can selectively extract data from specific sheets in an Excel file, allowing you to focus on the relevant data for further analysis or processing using pandas.

##### Exercise 1
```python
# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = 'battledeath.xlsx'

# Load spreadsheet: xls
xls = pd.ExcelFile(file)

# Print sheet names
print(xls.sheet_names)
```
![Pasted image 20230720110105](/images/Pasted%20image%2020230720110105.png)

##### Exercise 2
```python
# Load a sheet into a DataFrame by name: df1
df1 = xls.parse('2004')

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2 = xls.parse(0)

# Print the head of the DataFrame df2
print(df2.head())
```
![Pasted image 20230720110228](/images/Pasted%20image%2020230720110228.png)

##### Exercise 3
```python
# Parse the first sheet and rename the columns: df1
df1 = xls.parse(0, skiprows=1, names=['Country', 'AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xls.parse(1, usecols=[0], skiprows=1, names=['Country'])

# Print the head of the DataFrame df2
print(df2.head())
```
![Pasted image 20230720110455](/images/Pasted%20image%2020230720110455.png)

***
## pd.read_sas()

The `pd.read_sas()` function in pandas is used to read data from a SAS dataset file (.sas7bdat) and create a DataFrame.

**Function syntax:**
```python
pd.read_sas(filepath_or_buffer, format=None, index=None, encoding=None)
```

**Parameters:**
- `filepath_or_buffer`: Specifies the path or file-like object representing the SAS dataset file to be read.
- `format` (optional): Specifies the format of the SAS dataset file. By default, it is `None`, which means the format will be automatically inferred based on the file extension.
- `index` (optional): Specifies the column(s) to be used as the index of the DataFrame. It can be a single column or a list of columns.
- `encoding` (optional): Specifies the character encoding to be used when reading the SAS dataset file.

**Example of use:**
```python
import pandas as pd

# Read data from a SAS dataset file into a DataFrame
data = pd.read_sas('data.sas7bdat')

# Display the loaded data
print(data)
```

In this example, `pd.read_sas('data.sas7bdat')` reads data from the SAS dataset file `'data.sas7bdat'` and creates a DataFrame `data` containing the loaded data.

The `pd.read_sas()` function is specifically designed to handle SAS dataset files and provides a convenient way to read such files into pandas DataFrames. It supports various options to customize the reading process, such as specifying the format, index columns, and character encoding. The resulting DataFrame can then be used for data exploration, manipulation, analysis, and visualization using the rich functionality provided by pandas.

##### Importing SAS Files

```python
import pandas as pd
from sas7bdat import SAS7BDAT

with SAS7BDAT('urbanpop.sas7bdat') as file:
	df_sas = file.to_data_frame()
```

##### Exercise 1
```python
# Import sas7bdat package
from sas7bdat import SAS7BDAT

# Save file to a DataFrame: df_sas
with SAS7BDAT('sales.sas7bdat') as file:
    df_sas = file.to_data_frame()

# Print head of DataFrame
print(df_sas.head())

# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()
```
![Pasted image 20230720110739](/images/Pasted%20image%2020230720110739.png)
![Pasted image 20230720110747](/images/Pasted%20image%2020230720110747.png)

***
## pd.read_stata()

The `pd.read_stata()` function in pandas is used to read data from a Stata dataset file (.dta) and create a DataFrame.

**Function syntax:**
```python
pd.read_stata(filepath_or_buffer, convert_dates=True, convert_categoricals=True, encoding=None)
```

**Parameters:**
- `filepath_or_buffer`: Specifies the path or file-like object representing the Stata dataset file to be read.
- `convert_dates` (optional): Specifies whether to convert date variables to pandas' `datetime` data type. By default, it is `True`.
- `convert_categoricals` (optional): Specifies whether to convert categorical variables to pandas' `Categorical` data type. By default, it is `True`.
- `encoding` (optional): Specifies the character encoding to be used when reading the Stata dataset file.

**Example of use:**
```python
import pandas as pd

# Read data from a Stata dataset file into a DataFrame
data = pd.read_stata('data.dta')

# Display the loaded data
print(data)
```

In this example, `pd.read_stata('data.dta')` reads data from the Stata dataset file `'data.dta'` and creates a DataFrame `data` containing the loaded data.

The `pd.read_stata()` function is specifically designed to handle Stata dataset files and provides a convenient way to read such files into pandas DataFrames. It supports various options to customize the reading process, such as converting dates and categorical variables, as well as specifying the character encoding. The resulting DataFrame can then be used for data exploration, manipulation, analysis, and visualization using the rich functionality provided by pandas.

### Importing Stata Files

```python
import pandas as pd
file = pd.read_stata('urbanpop.dta')
```


##### Exercise 1
```python
# Import pandas
import pandas as pd

# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('disarea.dta')

# Print the head of the DataFrame df
print(df.head())

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of countries')
plt.show()
```
![Pasted image 20230720110939](/images/Pasted%20image%2020230720110939.png)
![Pasted image 20230720111003](/images/Pasted%20image%2020230720111003.png)


***
## h5py.File()

The `h5py.File()` function is used to open an HDF5 file and create an `h5py.File` object, which serves as a gateway for accessing the data stored in the HDF5 file.

**Function syntax:**
```python
h5py.File(name, mode='r', driver=None, libver=None, swmr=False, **kwds)
```

**Parameters:**
- `name`: Specifies the path or filename of the HDF5 file to be opened.
- `mode` (optional): Specifies the file mode. The default mode is `'r'` (read-only), but it can also be `'w'` (write, truncating any existing file), `'a'` (read/write if the file exists, create otherwise), or `'r+'` (read/write, without truncating the file).
- `driver` (optional): Specifies the HDF5 driver to be used. By default, it is `None`, which means the best available driver will be used.
- `libver` (optional): Specifies the HDF5 library version to be used. By default, it is `None`, which means the latest available library version will be used.
- `swmr` (optional): Specifies whether to enable single-writer/multiple-readers (SWMR) mode. The default is `False`.
- Additional keyword arguments can be used to pass various options to the underlying HDF5 library.

**Example of use:**
```python
import h5py

# Open an HDF5 file
file = h5py.File('data.hdf5', 'r')

# Access datasets and groups within the file
dataset = file['dataset_name']
group = file['group_name']

# Perform operations on the datasets or groups
data = dataset[:]
attributes = dataset.attrs

# Close the file when finished
file.close()
```

In this example, `h5py.File('data.hdf5', 'r')` opens the HDF5 file `'data.hdf5'` in read-only mode and creates an `h5py.File` object named `file`. You can then use the `file` object to access datasets and groups within the HDF5 file and perform operations on them. Finally, it is important to close the file using the `close()` method to release the resources associated with it.

The `h5py.File()` function is commonly used for reading and manipulating HDF5 files, which are widely used for storing large and complex datasets. The `h5py` library provides a convenient and efficient interface for working with HDF5 files in Python.

##### Importing HDF5 Files
```python
import h5py

filename = 'H-H1_LOSC.hdf5'
data = h5py.File(filename, 'r')

print(type(data))
```
![Pasted image 20230720103823](/images/Pasted%20image%2020230720103823.png)


##### The Structure of HDF5 Files

Data - Main Keys (Meta, Quality, Strain)
```python
for key in data.keys():
	print(key)
```
![Pasted image 20230720104100](/images/Pasted%20image%2020230720104100.png)
```python
print(type(data['meta']))
```
![Pasted image 20230720104241](/images/Pasted%20image%2020230720104241.png)

Each Main Key Has Keys
```python
for key in data['meta'].keys():
	print(key)
```
![Pasted image 20230720104413](/images/Pasted%20image%2020230720104413.png)

```python
print(np.array(data['meta']['Description']), np.array(data['meta']['Detector']))
```
![Pasted image 20230720104544](/images/Pasted%20image%2020230720104544.png)

##### Exercise 1
```python
# Import packages
import numpy as np
import h5py

# Assign filename: file
file = 'LIGO_data.hdf5'

# Load file: data
data = h5py.File(file, 'r')

# Print the datatype of the loaded file
print(type(data))

# Print the keys of the file
for key in data.keys():
    print(key)
```
![Pasted image 20230720111402](/images/Pasted%20image%2020230720111402.png)

##### Exercise 2
```python
# Get the HDF5 group: group
group = data['strain']

# Check out keys of group
for key in group.keys():
    print(key)

# Set variable equal to time series data: strain
strain = np.array(data['strain']['Strain'])

# Set number of time points to sample: num_samples
num_samples = 10000

# Set time vector
time = np.arange(0, 1, 1/num_samples)

# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()
```
![Pasted image 20230720111606](/images/Pasted%20image%2020230720111606.png)
![Pasted image 20230720111614](/images/Pasted%20image%2020230720111614.png)


***
## scipy.io.loadmat()

The `scipy.io.loadmat()` function is used to load data from a MATLAB file (.mat) and return a dictionary-like object containing the data.

**Function syntax:**
```python
scipy.io.loadmat(file_name, mdict=None, appendmat=True, variable_names=None)
```

**Parameters:**
- `file_name`: Specifies the path or name of the MATLAB file to be loaded.
- `mdict` (optional): Specifies an optional dictionary to which MATLAB variables will be copied.
- `appendmat` (optional): Specifies whether to append the MATLAB file's variables to the `mdict` dictionary. By default, it is `True`.
- `variable_names` (optional): Specifies a list of variable names to load from the MATLAB file. If provided, only the specified variables will be loaded.

**Example of use:**
```python
import scipy.io

# Load data from a MATLAB file
data = scipy.io.loadmat('data.mat')

# Access variables from the loaded data
variable1 = data['variable1']
variable2 = data['variable2']

# Display the loaded variables
print(variable1)
print(variable2)
```

In this example, `scipy.io.loadmat('data.mat')` loads the data from the MATLAB file `'data.mat'` and returns a dictionary-like object `data`. You can access the MATLAB variables from the loaded data using the corresponding keys.

The `scipy.io.loadmat()` function is part of the `scipy` library and provides a convenient way to read and load data from MATLAB files into Python. It is commonly used when working with MATLAB data in scientific and engineering applications.


## scipy.io.savemat()

The `scipy.io.savemat()` function is used to save data to a MATLAB file (.mat).

**Function syntax:**
```python
scipy.io.savemat(file_name, mdict, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='column')
```

**Parameters:**
- `file_name`: Specifies the path or name of the MATLAB file to be saved.
- `mdict`: Specifies a dictionary-like object containing the data to be saved. The keys represent variable names, and the corresponding values are the data arrays.
- `appendmat` (optional): Specifies whether to append variables to an existing MATLAB file. By default, it is `True`.
- `format` (optional): Specifies the MATLAB file format version. The default is `'5'`, which corresponds to MATLAB 5 format.
- `long_field_names` (optional): Specifies whether to support field names longer than 31 characters. The default is `False`.
- `do_compression` (optional): Specifies whether to apply compression to reduce the file size. The default is `False`.
- `oned_as` (optional): Specifies how one-dimensional arrays are written to MATLAB. The default is `'column'`.

**Example of use:**
```python
import numpy as np
import scipy.io

# Create a dictionary with data to save
data = {
    'variable1': np.array([1, 2, 3]),
    'variable2': np.array([4, 5, 6])
}

# Save data to a MATLAB file
scipy.io.savemat('data.mat', data)
```

In this example, a dictionary-like object `data` is created, containing two variables `'variable1'` and `'variable2'` along with their corresponding data arrays. The `scipy.io.savemat('data.mat', data)` function is then used to save the data to a MATLAB file named `'data.mat'`.

The `scipy.io.savemat()` function allows you to store data from Python to MATLAB-compatible files, which can be useful for exchanging data between Python and MATLAB environments. It supports various options to control the file format, compression, and array orientation.

##### Importing a MAT file
```python
import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))
```
![Pasted image 20230720105103](/images/Pasted%20image%2020230720105103.png)
![Pasted image 20230720111726](/images/Pasted%20image%2020230720111726.png)
```python
print(type(mat['x']))
```
![Pasted image 20230720105159](/images/Pasted%20image%2020230720105159.png)

##### Exercise 1
```python
# Import package
import scipy.io

# Load MATLAB file: mat
mat = scipy.io.loadmat('albeck_gene_expression.mat')

# Print the datatype type of mat
print(type(mat))
```
![Pasted image 20230720111917](/images/Pasted%20image%2020230720111917.png)

##### Exercise 2
```python
# Print the keys of the MATLAB dictionary
print(mat.keys())

# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))

# Print the shape of the value corresponding to the key 'CYratioCyt'
print(mat['CYratioCyt'].shape)

# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()
```
![Pasted image 20230720112112](/images/Pasted%20image%2020230720112112.png)
![Pasted image 20230720112120](/images/Pasted%20image%2020230720112120.png)


