
## `inspect.getdoc()`

In Python, the `inspect.getdoc()` function is used to retrieve the docstring (documentation string) of a Python object such as a module, class, function, method, or any other object that has a docstring. The docstring is a string that typically contains information about the object's purpose, usage, and details.

**Function Syntax:**
```python
inspect.getdoc(object)
```

**Parameters:**
- `object`: The Python object for which you want to retrieve the docstring.

**Return Value:**
- Returns a string containing the docstring of the specified object.

**Examples:**

```python
import inspect

# Define a function with a docstring
def greet(name):
    """
    This function greets the person passed in as a parameter.
    """
    return f"Hello, {name}!"

# Get the docstring of the greet() function
docstring = inspect.getdoc(greet)
```

In this example:
- We define a function `greet()` with a docstring that provides information about the function's purpose.
- We use `inspect.getdoc(greet)` to retrieve the docstring of the `greet()` function, and the result is stored in the variable `docstring`.

The `inspect.getdoc()` function is helpful when you want to access and display the documentation of Python objects dynamically, making it easier to understand and use various functions and modules in your code.

--- 
## `try-except-finally` blocks
In Python, you can define different types of `except` clauses to handle specific exceptions or categories of exceptions. Here are some commonly used `except` clauses and their purposes:

1. **`except` (Generic Exception Handling)**:
   - This is the most general `except` clause and can catch any exception. It's often used as a fallback when you want to handle any exception that may occur.

   ```python
   try:
       # Code that may raise an exception
   except Exception as e:
       # Handle the exception
   ```

2. **`except` followed by a Specific Exception**:
   - You can specify a particular exception class to catch and handle. This allows you to provide specialized handling for specific types of exceptions.

   ```python
   try:
       # Code that may raise a specific exception
   except ValueError:
       # Handle ValueError
   except FileNotFoundError:
       # Handle FileNotFoundError
   ```

3. **`except` with Multiple Exceptions**:
   - You can use a single `except` block to catch multiple exceptions by specifying them as a tuple.

   ```python
   try:
       # Code that may raise exceptions
   except (ValueError, FileNotFoundError):
       # Handle ValueError or FileNotFoundError
   ```

4. **`except` with an `as` Clause**:
   - You can use the `as` keyword to assign the caught exception to a variable, allowing you to access details about the exception.

   ```python
   try:
       # Code that may raise an exception
   except ValueError as ve:
       # Handle the ValueError and access it using 've'
   ```

5. **`except` (Catch-All) at the End**:
   - You can place a generic `except` clause at the end of a series of specific `except` clauses to catch any exceptions not caught by the specific handlers.

   ```python
   try:
       # Code that may raise exceptions
   except ValueError:
       # Handle ValueError
   except FileNotFoundError:
       # Handle FileNotFoundError
   except:
       # Handle any other exceptions
   ```

6. **`except` (Exception Base Class)**:
   - You can use the base class `Exception` to catch all exceptions that inherit from it. This is a broader catch-all than `except Exception`.

   ```python
   try:
       # Code that may raise exceptions
   except Exception as e:
       # Handle any exception (including subclasses of Exception)
   ```

These are some common `except` clauses you can use for exception handling in Python. You can create more specialized exception handling based on the specific needs of your code. Keep in mind that it's generally a good practice to handle exceptions as specifically as possible to provide clear and appropriate error handling for different scenarios.

You can write as many `except` clauses as you need to handle different types of exceptions in your code. There is no strict limit to the number of `except` clauses you can include in a `try`-`except` block.

However, it's important to consider the following guidelines when working with multiple `except` clauses:

1. **Order Matters**: Exception handling is processed from top to bottom in the order that `except` clauses appear in your code. Python will use the first `except` block that matches the raised exception. Therefore, if you have more specific exception handlers, they should appear before more generic ones to ensure that the specific exceptions are caught before the more general ones.

   ```python
   try:
       # Code that may raise exceptions
   except SpecificException:
       # Handle SpecificException
   except AnotherSpecificException:
       # Handle AnotherSpecificException
   except Exception:
       # Handle all other exceptions (generic)
   ```

2. **Catch-All Exception**: It's common to include a generic `except` clause at the end to catch any exceptions that were not caught by the more specific handlers. This is a good practice for providing a fallback error handling mechanism.

   ```python
   try:
       # Code that may raise exceptions
   except SpecificException:
       # Handle SpecificException
   except AnotherSpecificException:
       # Handle AnotherSpecificException
   except Exception:
       # Handle all other exceptions (generic)
   ```

3. **Use Specific Exceptions**: Whenever possible, handle specific exceptions that are relevant to your code. This allows you to provide more accurate error messages and tailored error handling for different scenarios.

In summary, you can have as many `except` clauses as needed to handle various types of exceptions in your code. Just remember to order them appropriately and prioritize specific exception handlers over generic ones to ensure that exceptions are caught and handled correctly.

---
## `global`

In Python, the `global` keyword is used to declare a variable as global inside a function. When you define a variable as global within a function, it means that the variable is accessible and modifiable from anywhere in the code, both inside and outside the function.

**Usage:**
```python
global variable_name
```

**Example:**

```python
x = 10  # Global variable

def modify_global_variable():
    global x  # Declare 'x' as a global variable
    x = 20   # Modify the global variable 'x' inside the function

modify_global_variable()

print(x)  # Output: 20
```

In this example:
- We define a global variable `x` with an initial value of `10`.
- Inside the `modify_global_variable()` function, we use the `global` keyword to declare that we want to modify the global variable `x`.
- We change the value of `x` to `20` within the function.
- After calling the function, the value of the global variable `x` has been modified to `20`, and it can be accessed and modified from any part of the code.

It's important to use the `global` keyword with caution, as excessive use of global variables can make code harder to understand and maintain. In many cases, it's recommended to use function parameters and return values to pass data between functions instead of relying heavily on global variables.

---
## `nonlocal`

In Python, the `nonlocal` keyword is used to indicate that a variable is a "nonlocal variable" within a nested function. Nonlocal variables are different from local variables in that they are not defined in the current function's scope but in an enclosing (usually outer) function's scope. The `nonlocal` keyword allows you to access and modify variables in the nearest enclosing scope that are not global.

Here's an example of how `nonlocal` is used:

```python
def outer_function():
    x = 10  # This is a variable in the outer function's scope

    def inner_function():
        nonlocal x  # Declare 'x' as a nonlocal variable
        x = 20  # Modify the nonlocal variable 'x' inside the inner function

    inner_function()
    print("Value of 'x' in outer function:", x)

outer_function()
```

In this example:

- We define an `outer_function()` that contains a variable `x` with an initial value of `10`.
- Inside the `outer_function()`, we define an `inner_function()` where we declare `x` as a nonlocal variable using `nonlocal x`.
- We modify the value of the nonlocal variable `x` to `20` inside the `inner_function()`.
- After calling the `inner_function()` within the `outer_function()`, we print the value of `x` from the outer function's perspective.

The output of this code will be:

```
Value of 'x' in outer function: 20
```

In this case, the `nonlocal` keyword allows us to modify the variable `x` in the outer function's scope from within the inner function. This is particularly useful when you have nested functions and want to work with variables from an enclosing scope without making them global.

---
## `closure`

Closures are a powerful and important concept in programming, especially in languages like Python where they are commonly used. A closure occurs when a nested function references a value from its containing (enclosing) function's scope, even after the containing function has finished executing. In Python, functions are first-class objects, which means they can be returned from other functions, passed as arguments, and enclosed within other functions.

Here's how closures work in Python:

1. **Nested Functions:** In Python, you can define functions within other functions. The inner function is referred to as a nested function.

   ```python
   def outer_function(x):
       def inner_function(y):
           return x + y
       return inner_function
   ```

2. **Access to Enclosing Scope:** The inner function (`inner_function` in this example) has access to variables in its containing function's scope (`outer_function` in this case). This means it can use and reference variables from `outer_function`.

   ```python
   add_five = outer_function(5)
   ```

3. **Closure Creation:** When you call `outer_function(5)`, it returns the `inner_function`. This is where the closure is created. The `inner_function` maintains a reference to the `x` variable from the `outer_function`'s scope, even after `outer_function` has finished executing.

4. **Use of Closure:** You can now use `add_five` as a function that adds `5` to any value.

   ```python
   result = add_five(10)  # Result is 15
   ```

   In this example, `add_five` is a closure because it "closes over" the `x` variable from the `outer_function`. It can remember and use the value of `x` even though `outer_function` has completed.

Common use cases for closures include:

- Creating functions with behavior that depends on some external context or configuration.
- Implementing data hiding or encapsulation by keeping data private within a function's scope.
- Implementing decorators, which are used to modify or enhance the behavior of functions or methods.

Closures are a powerful tool in Python and can make your code more modular, maintainable, and flexible by encapsulating behavior and data within functions while still allowing access to that data when needed.

---
Of course! Here's the information about the `.cell_contents` attribute in the previous format:

## `.cell_contents`

In Python, the `.cell_contents` attribute is used to access the values of variables within a closure. A closure is a nested function that "closes over" or captures variables from its enclosing function's scope. The `.cell_contents` attribute allows you to retrieve the values of these captured variables within the closure.

**Usage:**
```python
cell_object.cell_contents
```

**Example:**

```python
def return_a_func(arg1, arg2):
    def new_func():
        print('arg1 was {}'.format(arg1))
        print('arg2 was {}'.format(arg2))
    return new_func

my_func = return_a_func(2, 17)

# Get the values of the variables in the closure
closure_values = [
    cell.cell_contents for cell in my_func.__closure__
]

print(closure_values == [2, 17])  # Output: True
```

In this example:
- We define a closure `new_func` within the `return_a_func` function.
- The values of `arg1` and `arg2` from the outer function are captured by the closure.
- We access the values of these captured variables using the `.cell_contents` attribute within the `my_func.__closure__` object.

The `.cell_contents` attribute is a crucial mechanism for working with closures in Python, allowing you to access and utilize variables from enclosing scopes within your functions.

---
## `@decorators`

1. **`@staticmethod` and `@classmethod`:
   - `@staticmethod` defines a method that doesn't depend on the instance and can be called on the class itself.
   - `@classmethod` defines a method that takes the class itself as its first argument.

```python
class MyClass:
    class_variable = 10

    @staticmethod
    def static_method(x, y):
        return x + y

    @classmethod
    def class_method(cls, x):
        return cls.class_variable + x

# Using static method
result1 = MyClass.static_method(5, 3)

# Using class method
result2 = MyClass.class_method(7)

print(result1)  # Output: 8 (5 + 3)
print(result2)  # Output: 17 (10 + 7)
```

2. **`@property`**:
   - `@property` decorator is used to define a method as a "getter" for a class attribute.
   - It allows you to access an attribute like a regular attribute, even though it's computed via a method.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @property
    def area(self):
        return 3.14 * self._radius ** 2

circle = Circle(5)
print(circle.radius)  # Accessing radius as a property
print(circle.area)    # Accessing area as a property
```

3. **`@<decorator>` (Custom Decorators)**:
   - Custom decorators are functions that take a function as an argument and return a new function that typically wraps the original function.

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

4. **`@lru_cache`**:
   - `@lru_cache` decorator is used for caching the results of a function to improve performance.
   - It stores the results of expensive function calls and reuses them when the same inputs occur again.

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))  # Calculates Fibonacci(10) and caches results
```

Certainly! Here are the remaining common decorators in Python along with code examples and comments:

5. **`@abstractmethod`**:
   - `@abstractmethod` decorator is used in abstract base classes (defined using the `abc` module) to indicate that a method must be implemented by concrete subclasses.
   - It enforces the implementation of specific methods in derived classes.

```python
from abc import ABC, abstractmethod

class MyAbstractClass(ABC):
    @abstractmethod
    def abstract_method(self):
        pass

class ConcreteClass(MyAbstractClass):
    def abstract_method(self):
        print("Concrete implementation of abstract_method")

obj = ConcreteClass()
obj.abstract_method()
```

6. **`@contextmanager`**:
   - `@contextmanager` decorator is used to define context managers, which allow you to set up and tear down resources within a `with` block.
   - It simplifies resource management and cleanup.

```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("Entering the context")
    yield
    print("Exiting the context")

with my_context():
    print("Inside the context")
```

7. **`@functools.wraps`**:
   - `@functools.wraps` decorator is used when creating custom decorators to preserve the metadata (such as docstrings) of the wrapped function.
   - It ensures that the wrapper function behaves like the original function.

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Before calling the function")
        result = func(*args, **kwargs)
        print("After calling the function")
        return result
    return wrapper

@my_decorator
def my_function():
    """This is a docstring."""
    print("Inside the function")

print(my_function.__name__)  # Output: "my_function" (preserves function name)
print(my_function.__doc__)   # Output: "This is a docstring." (preserves docstring)
```

### Custom example: ``@double_args`

The `@double_args` decorator can be a custom decorator that doubles the arguments passed to a function. Here's an example of how to create and use the `@double_args` decorator:

```python
# Define a custom decorator
def double_args(func):
    def wrapper(*args, **kwargs):
        # Double each argument
        doubled_args = [arg * 2 for arg in args]
        # Call the original function with doubled arguments
        return func(*doubled_args, **kwargs)
    return wrapper

# Apply the @double_args decorator to a function
@double_args
def add(a, b):
    return a + b

# Use the decorated function
result = add(3, 4)
print(result)  # Output: 14 (3 * 2 + 4 * 2)
```

In this example, the `@double_args` decorator takes a function `func` as its argument, and the `wrapper` function inside the decorator doubles each argument passed to the wrapped function. The original function `func` is then called with the doubled arguments.

When you apply the `@double_args` decorator to the `add` function, any arguments passed to `add` will be doubled before the addition is performed.

You can create custom decorators like `@double_args` to modify the behavior of functions in various ways based on your specific needs.

In Python, `*args` and `**kwargs` are special syntax used to pass a variable number of arguments to a function. They allow you to work with a flexible number of positional arguments and keyword arguments within a function. Here's an explanation of both:

1. **`*args`** (Positional Arguments):
   - The `*args` syntax allows a function to accept a variable number of positional arguments.
   - It collects all the positional arguments into a tuple within the function.

   ```python
   def example_function(*args):
       for arg in args:
           print(arg)

   example_function(1, 2, 3, 4)
   ```

   In this example, `*args` collects all the arguments passed to `example_function` into a tuple, which can then be iterated through.

2. **`**kwargs`** (Keyword Arguments):
   - The `**kwargs` syntax allows a function to accept a variable number of keyword arguments.
   - It collects all the keyword arguments into a dictionary within the function.

   ```python
   def example_function(**kwargs):
       for key, value in kwargs.items():
           print(key, value)

   example_function(a=1, b=2, c=3)
   ```

   In this example, `**kwargs` collects all the keyword arguments and their values into a dictionary, which can then be iterated through.

The use of `*args` and `**kwargs` is a convention, and you can choose other variable names, but it's a widely adopted convention that makes the code more readable and clear. The asterisk (*) syntax is used to indicate that the function should collect multiple arguments into a tuple (`*args`) or a dictionary (`**kwargs`). These names are not fixed; you can use any valid variable names, but using `*args` and `**kwargs` is a common and accepted practice in Python.

So, in the code example provided:
- `*args` in the `wrapper(*args, **kwargs)` line collects all positional arguments passed to the wrapped function.
- `**kwargs` in the `wrapper(*args, **kwargs)` line collects all keyword arguments passed to the wrapped function.

These collected arguments can then be used within the `wrapper` function to call the original function with the adjusted arguments.

---
## `magic` or `dunder` operators

In Python, there are several built-in attributes (often referred to as "magic" or "dunder" methods) and attributes that are commonly used to access metadata or perform special operations on objects. Here are some of the most common ones:

1. **`__doc__`**: Provides the docstring (documentation string) of an object, which is typically a string containing information about the object's purpose and usage.

2. **`__name__`**: Returns the name of a module, function, class, or method as a string.

3. **`__module__`**: Indicates the name of the module where the object is defined. Useful for determining the module of a function or class.

4. **`__class__`**: Returns the class to which an instance belongs. Useful for checking the class of an object.

5. **`__dict__`**: Contains a dictionary of an object's attributes and their values. Useful for introspection and dynamic attribute access.

6. **`__getattr__`**: Allows custom handling of attribute access. It's called when an attribute is not found through the usual lookup process.

7. **`__setattr__`**: Allows custom handling of attribute assignment. It's called when an attribute is assigned a value.

8. **`__delattr__`**: Allows custom handling of attribute deletion. It's called when an attribute is deleted using the `del` statement.

9. **`__init__`**: A special method used for initializing instances of a class. It's called automatically when an object is created.

10. **`__call__`**: Allows an object to be called as if it were a function. It's invoked when you use parentheses `()` to call an object.

11. **`__str__`**: Returns a string representation of an object. It's called by the built-in `str()` function and when using `print`.

12. **`__repr__`**: Returns a string representation of an object that can be used to recreate the object. It's called by the built-in `repr()` function.

13. **`__eq__`**: Defines the behavior of the equality (`==`) operator for custom objects. It's used to compare objects for equality.

14. **`__lt__`, `__le__`, `__gt__`, `__ge__`**: Define comparison methods for custom objects to support ordering operations like `<`, `<=`, `>`, `>=`.

15. **`__len__`**: Defines the behavior of the built-in `len()` function for custom objects. It's used to determine the length of an object.

16. **`__getitem__`, `__setitem__`, `__delitem__`**: Define methods for indexing and slicing custom objects (used with square brackets `[]`).

These are some of the most commonly used metadata attributes and special methods in Python. They allow you to customize the behavior of your classes and objects, perform introspection, and access important information about your code.

---
## `__wrapped__`

1. **`@functools.wraps` Decorator**:
   - The `@functools.wraps` decorator is used to create custom decorators that wrap or modify the behavior of a function.
   - One of its main purposes is to ensure that the wrapped function retains its original metadata (e.g., docstring, name) so that it behaves like the original function.

2. **`__wrapped__` Attribute**:
   - When you use `@functools.wraps` to decorate a function, the decorator sets the `__wrapped__` attribute of the wrapper function (the decorated function) to refer to the original (wrapped) function.
   - This allows you to access the original function through the `__wrapped__` attribute, even though you are working with the decorated version.

Here's an example:

```python
import functools

def my_decorator(func):
    @functools.wraps(func)  # Preserve metadata of the original function
    def wrapper(*args, **kwargs):
        print("Before calling the function")
        result = func(*args, **kwargs)
        print("After calling the function")
        return result
    return wrapper

@my_decorator
def my_function():
    """This is a docstring."""
    print("Inside the function")

print(my_function.__name__)  # Output: "my_function" (preserves function name)
print(my_function.__doc__)   # Output: "This is a docstring." (preserves docstring)

# Access the original function using the __wrapped__ attribute
original_function = my_function.__wrapped__
print(original_function.__name__)  # Output: "my_function"
```

In this example, `@functools.wraps(func)` ensures that the decorated `wrapper` function retains the metadata of the original `my_function`, including its name and docstring. You can access the original function using the `__wrapped__` attribute.

---
## `signal.signal()`

In Python, the `signal.signal()` function is used to set a handler function for a specific signal. Signals are a form of inter-process communication used to notify a process that a specific event has occurred. You can use `signal.signal()` to associate a custom handler function with a signal, allowing your program to respond to various events, such as keyboard interrupts (`SIGINT`) or termination requests (`SIGTERM`).

**Function Syntax:**
```python
signal.signal(signalnum, handler)
```

**Parameters:**
- `signalnum`: An integer representing the signal number (e.g., `signal.SIGINT` for Ctrl+C).
- `handler`: A callable (usually a function) that will be called when the specified signal is received. It takes two arguments: the signal number and the current stack frame (frame object).

**Return Value:**
- The return value of `signal.signal()` is the previous signal handler function (if one existed) for the specified signal.

**Examples:**

```python
import signal
import time

# Define a custom handler function for SIGINT (Ctrl+C)
def sigint_handler(signum, frame):
    print(f"Received SIGINT (Ctrl+C). Exiting gracefully.")
    exit(0)

# Set the custom handler for SIGINT
signal.signal(signal.SIGINT, sigint_handler)

# Wait for SIGINT (Ctrl+C)
print("Press Ctrl+C to trigger the custom handler.")
while True:
    time.sleep(1)
```

In this example:
- We define a custom handler function `sigint_handler` to handle the `SIGINT` signal (Ctrl+C).
- We use `signal.signal(signal.SIGINT, sigint_handler)` to set the custom handler for the `SIGINT` signal.
- When you press Ctrl+C while running the program, it triggers the custom handler function.

The `signal.signal()` function is a powerful tool for managing signal handling in your Python programs, allowing you to define custom behavior when specific events or interrupts occur.

---
## `signal.alarm()`

In Python, the `signal.alarm()` function is used to schedule an alarm signal (specifically, `SIGALRM`) to be sent to the calling process after a specified number of seconds. This function is often used for setting timers and creating timeouts within your Python programs.

**Function Syntax:**
```python
signal.alarm(seconds)
```

**Parameters:**
- `seconds`: An integer representing the number of seconds after which the alarm signal (`SIGALRM`) will be triggered. If `seconds` is `0`, any previously scheduled alarm is canceled.

**Return Value:**
- The return value of `signal.alarm()` is the number of seconds remaining before any previously scheduled alarm was set. If no alarm was previously set, it returns `0`.

**Example:**

```python
import signal
import time

# Define a handler for the alarm signal (SIGALRM)
def alarm_handler(signum, frame):
    print("Alarm triggered!")

# Set the custom alarm handler
signal.signal(signal.SIGALRM, alarm_handler)

# Schedule an alarm to trigger after 5 seconds
signal.alarm(5)

try:
    print("Waiting for the alarm to trigger...")
    time.sleep(10)
except KeyboardInterrupt:
    pass
```

In this example:
- We define a custom handler function `alarm_handler` to handle the alarm signal (`SIGALRM`).
- We set the custom alarm handler using `signal.signal(signal.SIGALRM, alarm_handler)`.
- We schedule an alarm to trigger after 5 seconds using `signal.alarm(5)`.
- We use a `try` block and `time.sleep(10)` to wait for the alarm to trigger.
- The alarm signal is delivered to the process after 5 seconds, and the custom handler prints "Alarm triggered!".

The `signal.alarm()` function is useful for creating timers and implementing timeouts in Python programs. It allows you to schedule actions to occur after a specified duration, which can be helpful for various timing-related tasks.