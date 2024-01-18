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




### --- Exercise 2 --- ###




### --- Exercise 3 --- ###




### --- Exercise 4 --- ###




### --- Exercise 5 --- ###




### --- Exercise 6 --- ###




### --- Exercise 7 --- ###




### --- Exercise 8 --- ###




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




