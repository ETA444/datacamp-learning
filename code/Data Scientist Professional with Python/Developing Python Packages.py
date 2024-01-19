# Code Exercises from Developing Python Packages #

## Chapter 1

### --- Exercise 1 --- ###




### --- Exercise 2 --- ###




### --- Exercise 3 --- ###




### --- Exercise 4 --- ###

INCHES_PER_FOOT = 12.0  # 12 inches in a foot
INCHES_PER_YARD = INCHES_PER_FOOT * 3.0  # 3 feet in a yard

UNITS = ("in", "ft", "yd")


def inches_to_feet(x, reverse=False):
    """Convert lengths between inches and feet.

    Parameters
    ----------
    x : numpy.ndarray
        Lengths in feet.
    reverse : bool, optional
        If true this function converts from feet to inches 
        instead of the default behavior of inches to feet. 
        (Default value = False)

    Returns
    -------
    numpy.ndarray
    """
    if reverse:
        return x * INCHES_PER_FOOT
    else:
        return x / INCHES_PER_FOOT



### --- Exercise 5 --- ###

"""User-facing functions."""
from impyrial.length.core import (
    inches_to_feet,
    inches_to_yards,
    UNITS
)


def convert_unit(x, from_unit, to_unit):
    """Convert from one length unit to another.

    Parameters
    ----------
    x : array_like
        Lengths to convert.
    from_unit : {'in', 'ft', 'yd'}
        Unit of the input lengths `x`
    to_unit : {'in', 'ft', 'yd'}
        Unit of the returned lengths

    Returns
    -------
    ndarray
        An array of converted lengths with the same shape as `x`. If `x` is a
        0-d array, then a scalar is returned.
    """
    # Convert length to inches
    if from_unit == "in":
        inches = x
    elif from_unit == "ft":
        inches = inches_to_feet(x, reverse=True)
    elif from_unit == "yd":
        inches = inches_to_yards(x, reverse=True)

    # Convert inches to desired units
    if to_unit == "in":
        value = inches
    elif to_unit == "ft":
        value = inches_to_feet(inches)
    elif to_unit == "yd":
        value = inches_to_yards(inches)

    return value


from impyrial.length.api import convert_unit

result = convert_unit(10, 'in', 'yd')
print(result)



### --- Exercise 6 --- ###

# This is the __init__.py file for the impyrial/length subpackage

# Import the convert_unit function from the api.py module
from .api import convert_unit

# This is the top level __init__.py file

# Import the length subpackage
from . import length

import impyrial

result = impyrial.length.convert_unit(10, 'in', 'yd')
print(result)




## Chapter 2

### --- Exercise 1 --- ###

# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="George Dreemer",
    description="A package for converting imperial lengths and weights.",
    name="impyrial",
    packages=find_packages(include=['impyrial', 'impyrial.*']),
    version="0.1.0",
)



### --- Exercise 2 --- ###

import impyrial

result = impyrial.length.convert_unit(10, 'in', 'yd')
print(result)



### --- Exercise 3 --- ###

"""Conversions between ounces and larger imperial weight units"""
OUNCES_PER_POUND = 16  # 16 ounces in a pound
OUNCES_PER_STONE = OUNCES_PER_POUND * 14.0  # 14 pounds in a stone

UNITS = ("oz", "lb", "st")


def ounces_to_pounds(x, reverse=False):
    """Convert weights between ounces and pounds.

    Parameters
    ----------
    x : array_like
        Weights in pounds.
    reverse : bool, optional
        If this is set to true this function converts from pounds to ounces
        instead of the default behaviour of ounces to pounds.

    Returns
    -------
    ndarray
        An array of converted weights with the same shape as `x`. If `x` is a
        0-d array, then a scalar is returned.
    """
    if reverse:
        return x * OUNCES_PER_POUND
    else:
        return x / OUNCES_PER_POUND


def ounces_to_stone(x, reverse=False):
    """Convert weights between ounces and stone.

    Parameters
    ----------
    x : array_like
        Weights in stone.
    reverse : bool, optional
        If this is set to true this function converts from stone to ounces
        instead of the default behaviour of ounces to stone.

    Returns
    -------
    ndarray
        An array of converted weights with the same shape as `x`. If `x` is a
        0-d array, then a scalar is returned.
    """
    if reverse:
        return x * OUNCES_PER_STONE
    else:
        return x / OUNCES_PER_STONE



### --- Exercise 4 --- ###

from setuptools import setup, find_packages

# Add install requirements
setup(
    author="<your-name>",
    description="A package for converting imperial lengths and weights.",
    name="impyrial",
    packages=find_packages(include=["impyrial", "impyrial.*"]),
    version="0.1.0",
    install_requires=['numpy>=1.10', 'pandas'],
)


### --- Exercise 5 --- ###

"""
pip freeze > requirements.txt

...
absl-py==0.15.0
agate==1.7.0
aiohttp==3.8.4
aiohttp-retry==2.8.3
aiosignal==1.3.1
alabaster==0.7.13
amqp==5.1.1
antlr4-python3-runtime==4.9.3
appdirs==1.4.4
arrow==1.2.3
asn1crypto==0.24.0
astroid==2.15.5
asttokens==2.0.5
astunparse==1.6.3
async-timeout==4.0.2
asyncssh==2.13.1
atpublic==4.0
attrs==23.1.0
...
"""


### --- Exercise 6 --- ###

""" README.md
# impyrial

A package for converting between imperial unit lengths and weights.

This package was created for the [DataCamp](https://www.datacamp.com) course "Developing Python Packages".

### Features

- Convert lengths between miles, yards, feet and inches.
- Convert weights between hundredweight, stone, pounds and ounces.

### Usage

```python
import impyrial

# Convert 500 miles to feet
impyrial.length.convert_unit(500, from_unit='yd', to_unit='ft')  # returns 1500.0

# Convert 100 ounces to pounds
impyrial.weight.convert_unit(100, from_unit='oz', to_unit='lb')  # returns 6.25
```

"""

### --- Exercise 7 --- ###

""" MANIFEST.in

include LICENSE
include README.md

"""



## Chapter 3

### --- Exercise 1 --- ###




### --- Exercise 2 --- ###




### --- Exercise 3 --- ###




### --- Exercise 4 --- ###




### --- Exercise 5 --- ###




### --- Exercise 6 --- ###




### --- Exercise 7 --- ###




### --- Exercise 8 --- ###




## Chapter 4

### --- Exercise 1 --- ###




### --- Exercise 2 --- ###




### --- Exercise 3 --- ###




### --- Exercise 4 --- ###




### --- Exercise 5 --- ###




### --- Exercise 6 --- ###




### --- Exercise 7 --- ###




### --- Exercise 8 --- ###




