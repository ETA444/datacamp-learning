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
![[Pasted image 20230719191729.png]]

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
![[Pasted image 20230719194411.png]]

```python
# Check whether file is closed
print(file.closed)
```
![[Pasted image 20230719194432.png]]

```python
# Close file
file.close()

# Check whether file is closed
print(file.closed)
```
![[Pasted image 20230719194444.png]]

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
![[Pasted image 20230719192017.png]]

### Exercises

##### Exercise 1
```python
# Read & print the first 3 lines
with open('moby_dick.txt', 'r') as file:
	print(file.readline())
	print(file.readline())
	print(file.readline())
```
![[Pasted image 20230719194844.png]]


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
![[Pasted image 20230719193036.png]]

##### Customizing the NumPy import
- `skiprows` (optional): Specifies the number of rows at the beginning of the file to skip before loading the data.
```python
import numpy as np
filename = 'MNIST_header.txt'
data = np.loadtxt(filename, delimiter = ',', skiprows = 1)
print(data)
```
![[Pasted image 20230719193300.png]]

- `usecols` (optional): Specifies which columns to load from the file. It can be a single column index, a sequence of column indices, or a sequence of column names.
```python
import numpy as np
filename = 'MNIST_header.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols[0,2])
print(data)
```
![[Pasted image 20230719193453.png]]

- `dtype` (optional): Specifies the data type of the resulting array. By default, it is set to `float`.
```python
data = np.loadtxt(filename, delimiter = ',', dtype = str)
```

![[Pasted image 20230719193916.png]]

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
![[Pasted image 20230719205725.png]]
![[Pasted image 20230719205739.png]]

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
![[Pasted image 20230719210013.png]]

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
![[Pasted image 20230719210225.png]]
![[Pasted image 20230719210239.png]]

##### Exercise 4
```python
# Assign the filename: file
file = 'titanic.csv'

# Import file using np.recfromcsv: d
d = np.recfromcsv(file, delimiter = ',', names=True, dtype=None)

# Print out first three entries of d
print(d[:3])

```
![[Pasted image 20230719211247.png]]


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



### Importing using pandas

```python
import pandas as pd
filename = 'wineequality-red.csv'
data = pd.read_csv(filename)
data.head()
```
![[Pasted image 20230719193844.png]]

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
![[Pasted image 20230719211413.png]]

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
![[Pasted image 20230719212203.png]]
![[Pasted image 20230719212211.png]]


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





