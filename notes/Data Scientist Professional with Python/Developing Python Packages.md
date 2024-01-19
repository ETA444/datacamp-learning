
# Function and Class Documentation
## 'pyment -w -o numpydoc *file*'

**Pyment** is a tool used to generate template docstrings for *functions and classes*. 
- **Ran from the terminal:** It is ran from the terminal as shown below.
- **Many documentation styles:** It can generate any documentation style from:
	- Google
	- Numpydoc
	- reST (reStructured-text)
	- Javadoc (epytext)
- **Modify existing docs:** It can modify documentation from one style to another.

```terminal
pyment -w -o numpydoc textanalysis.py
```
Using **pyment** in the terminal and specifying:
- -w : overwrite specified file
- -o *numpydoc* : output in NumPy style
- Finally we specify the file we work with

```terminal
pyment -w -o google textanalysis.py
```
Using **pyment** to this time convert the existing numpy style documentation into google style documentation

# Package, subpackage and module documentation

## _ _ init _ _ .py

Documentation for these parts of a package are written in the respective `_ _ init _ _ .py` file.

For example **package** documentation:
- `mysklearn/__init__.py`
	```
	Linear regression for Python
	==============================
	
	mysklearn is a complete package for
	implementing linear regression in python.
	...
	```
For example **subpackage** documentation:
- `mysklearn/preprocessing/__init__.py`
	```
	A subpackage for standard preprocessing...
	...
	```
For example **module** documentation:
- `mysklearn/preprocessing/normalize.py`
	```
	A module for ...
	...
	```

# `setup.py` script

This is a script which makes the package you are creating installable. Also:
- Contains metadata for the package
- The setup script is NOT part of the package source code, therefore it is located outside of the main source code folder, e.g.:
	```markdown
	mypackage/ <-- outer directory
		|-- mypackage/ <-- source code dir
			|-- ...
	setup.py
```

## Inside `setup.py`

```python
# import required functions
from setuptools import setup, find_packages

# call setup function
setup(
	  author = "George Dreemer",
	  description = "My package",
	  name = "mypackage",
	  version = "0.1.0",
	  packages = find_packages(
		  include=[
		  "mypackage",
		  "mypackage.*"
		  ]
	  ),
	   
)
```
- Note: version number is comprised of three parts - (major number).(minor number).(patch number)

Thanks to the `setup.py` script the package can now be installed with:
```python
pip install -e .
# . = package in current directory
# -e = editable (so when the package is updated the installation is also updated)
```

# Dealing with dependencies

To deal with dependencies we look back to `setup.py`, where we need to add something new.

## Inside `setup.py`
```python
# import required functions
from setuptools import setup, find_packages

# call setup function
setup(
	# ...
	install_requires = [
		'pandas', 'scipy', 'matplotlib'
	],	   
)
```
Now when someone is pip installing your package, pip will additionally install these dependencies as well. 

### Controlling dependency version
In the above example any version of the package would have been accepted and not installed if it existed.

Sometimes our package needs a specific version of a dependency to work. This is handled in the following way:
```python
# import required functions
from setuptools import setup, find_packages

# call setup function
setup(
	# ...
	install_requires = [
		'pandas>=1.0', # ver 1.0 or later
		'scipy==1.1', # ver 1.1 only
		'matplotlib>=2.2.1,<3'
		# 2.2.1 or later but earlier than 3
	],	   
)
```
In the above example we control for the versions in three ways. The first and third ways are acceptable, while limiting to only one specific version is not recommended. Generally, it is **best to allow for as many versions as possible.** Furthermore, always doublecheck and remove unnecessary dependencies.

### Python versions
You can also specify which version of Python is necessary.
```python
# import required functions
from setuptools import setup, find_packages

# call setup function
setup(
	# ...
	python_requires = '>=2.7, !=3.0.*, !=3.1.*',   
)
```

## Making an environment for developers

When developing a package it is crucial to know your developing environment.

### Package versions
It is important to know exactly what versions the packages in your environment are. \

This information is extracted with:
```terminal
pip freeze
```
*Example output:*
```md
package == 1.0.0
package21 == 2.2.1
...
otherpackage == 1.7.2
otherpackage11 == 2.2.2
```

The information needs to be saved in the file `requirements.txt`.
```terminal
pip freeze > requirements.txt
```

Anyone can then install the requirements:
```terminal
pip install -r requirements.txt
```

# MANIFEST.in

Lists all the extra files to include in your package distribution.

```
include LICENSE
include README.md
```

 We specifically include these files in the `MANIFEST.in` file as they do not get installed with the package automatically.

# Publishing your package

## PyPI

The Python Package index is a public repository of Python packages where anyone could upload to.

- `pip` installs packages from here
- Anyone can upload packages

### Distributions
- **Distribution package:** a bundled version of your package which is ready to install.
	- **Source distribution:** a distribution package which is mostly your source code.
	- **Wheel distribution:** a distribution package which has been processed to make it faster to install.

#### How to build distributions
You build these distributions once your package is read from the terminal, like so:
```terminal
python setup.py sdist bdist_wheel
```
- `sdist` = source distribution
- `bdist_wheel` = wheel distribution

#### Getting your package out there

To upload your package to PyPI, you can do so from the terminal using `twine`:
```terminal
twine upload dist/*
```

You can also upload to the TestPyPI repository:
```terminal
twine upload -r testpypi dist/*
```

---
