
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
# Testing your package 

## Using `assert`
Writing test functions utilizing `assert` statements is a good way to keep your code in check.

```python
def get_ends(element):
	"""Get the first and last element in a list."""
	return element[0], element[-1]

def test_get_ends():
	assert get_ends([1,5,39,0]) == (1,0)
```

## Organizing tests inside your package

```markdown
mypackage/
|-- mypackage 
|-- tests     <- tests directory
|-- setup.py
|-- LICENSE
|-- MANIFEST.in
```

The **best way to structure** the `../tests/` directory is to mirror the structure of the source code directory (`../mypackage/`).

- **Test dir:**
```markdown
mypackage/tests/
|-- _ _ init _ _ .py
|-- mysubpackage
|   |-- _ _ init _ _ .py
|   |-- test_module1.py
|   |-- test_module2.py
...
```
- **Code dir:**
```markdown
mypackage/mypackage/
|-- _ _ init _ _ .py
|-- mysubpackage
|   |-- _ _ init _ _ .py
|   |-- module1.py
|   |-- module2.py
...
```

## Inside a Test module file
The test module script mirrors the source code module script similarly to the directory structure.

- **Inside `test_module1.py`**
	- Below we use the example with 3 simple functions to find the maximum & minimum.
```python
from mypackage.mysubpackage.module1 import (
	find_max, 
	find_min
)

def test_find_max(x):
	assert find_max([1,2,3,4]) == 4

def test_find_min(x):
	assert find_min([1,2,3,4]) == 1
```
- **Inside `module1.py`**
```python
def find_max(x):
	# ...
	return(x_max)

def find_min(x):
	# ...
	return(x_min)
```

## Running tests with `pytest`
To run tests we use `pytest` command in the terminal while our current directory is the top level directory of our package.
```terminal
pytest
```
- `pytest` looks inside the `test` directory
- It looks for modules and functions starting with `test_` like `test_module1.py` & `test_find_max()`.

## Testing multiple versions of Python

This `setup.py` allows any version of Python from version 2.7 upwards.
```python
from setuptools import setup, find_packages

setup(
	  # ...
	  python_requires = '>=2.7',
	  # ...
)
```

**To test these Python versions you must:**
- Install all these Python versions
- Run `tox`

### What is `tox`?
- Designed to run tests with multiple versions of Python

**To configure tox:**
- Create a **configuration file** - `tox.ini`
```markdown
mypackage/
|-- mypackage
|   |-- ...
|-- tests
|   |-- ...
|-- setup.py
|-- LICENSE
|-- MANIFEST.in
|-- tox.ini <--- configuration file
```

**Inside the file:**
```tox
[tox] <- these are called headings
envlist = py27, py35, py36, py37

[testenv]
deps = pytest
commands = 
	pytest
	... <- shell commands you need tox to run
```
- Headings are surrounded by square brackets, like so: `[...]`
- To test Python version X.Y add `pyXY` to `envlist`.
	- Python versions you want to test need to already be installed on your machine.
- The `commands` parameter lists the terminal commands `tox` will run.
	- The `commands` list can be any commands which will run from the terminal, like `ls`, `cd`, `echo` etc.

**To run `tox`**
Use the terminal navigating to the top-level directory and run:
```terminal
tox
```

# Keeping your package stylish

## Introducing `flake8`
- Standard Python style is described in *PEP8*
- A style guide dictates how code should be laid out
While:
- `pytest` is used to find bugs
- `flake8` is used to find styling mistakes

### Running `flake8`
`flake8` is a static code checker - meaning that it reads the code but doesn't run it.
- To run `flake8` in terminal:
```terminal
flake8 features.py
```
- It would print a series of style improvement suggestions, for example:
```terminal
features.py:2:1: F401 'math' imported but unused
..
```
- The format of the output above is as follows:
```terminal
<filename>:<line number>:<character number>: <error code> <description>
```

#### Ignoring violations

##### Using comments in the code
- For exceptions of `flake8` use `# noqa`, this will make the style checker ignore the line, e.g.:
```python
quadratic_1 = 6 * x**2 + 2*x + 4 # noqa
```
- In order to ignore a particular type of styling violation we still use `# noqa` but we specify the violation code we want to ignore, like so:
```python
quadratic_1 = 6 * x**2 + 2*x + 4 # noqa: E222
```

##### Using the terminal
If you want to avoid putting comments in your code, you can also instruct `flake8` through the terminal, like so:
- **Tell it what to ignore with `--ignore`**:
```terminal
flake8 --ignore E222 quadratic.py
```
- **Ask it what to look for with `--select`:
```terminal
flake8 --select F401,F841 features.py
```

##### Using `setup.cfg`
Create a `setup.cfg` file in the top level of the repository.
```markdown
mypackage/
|-- mypackage
|   |-- _ _ init _ _ .py
|   |-- mysubpackage
|   |   |-- _ _ init _ _ .py
|   |   |-- module1.py
|   |   |-- module2.py
|-- tests
|   |-- _ _ init _ _ .py
|   |-- mysubpackage
|   |   |-- _ _ init _ _ .py
|   |   |-- test_module1.py
|   |   |-- test_module2.py
|-- setup.py
|-- setup.cfg    <---------
|-- README.md
|-- LICENSE
|-- MANIFEST.in
|-- tox.ini
```
- **Inside the config file:**
	- You can ignore an error code altogether.
	- You can exclude files from QA.
	- You can ignore an error in a specific file.
```setup
[flake8]

ignore = E302
exclude = setup.py

per-file-ignores =
	mypackage/mysubpackage/module1.py: E222
```

###### Always use the least amount of filtering
In conclusion you should always use the **least filtering possible.**
1. Most preferred is to **only filter out a specific code on a specific line**: `# noqa : <code>`.
2. If that doesn't cut it, **completely exclude this line** by using `# noqa`.
3. If you need more global filtering, start with **specific code file ignores** `per-file-ignores` in the `setup.cfg` file.
4. Least preferred is to use **global commands** in the `setup.cfg` file: `ignore` and `exclude`.


# Faster package development with templates
The package file tree gets quite complicated due to all the extra files you need beside your source code:
```markdown
mypackage/
|-- mypackage
|   |-- _ _ init _ _ .py
|   |-- mysubpackage
|   |   |-- _ _ init _ _ .py
|   |   |-- module1.py
|   |   |-- module2.py
|-- tests
|   |-- _ _ init _ _ .py
|   |-- mysubpackage
|   |   |-- _ _ init _ _ .py
|   |   |-- test_module1.py
|   |   |-- test_module2.py
|-- setup.py
|-- setup.cfg
|-- README.md
|-- LICENSE
|-- MANIFEST.in
|-- tox.ini
```

## Templates /w `cookiecutter`
To avoid having to manually create all of these files we can use `cookiecutter` to make these files for us.
- Can be used to create empty Python packages
- Create all the additional files your package needs

**In order to use `cookiecutter` you run it from terminal, like so:**
```terminal
cookiecutter <template-url>
```
- The template URL points towards the template you want to use, the **standard Python package template** is this:
```terminal
cookiecutter https://github.com/audreyr/cookiecutter-pypackage
```
- You can find more templates [here](https://cookiecutter.readthedocs.io/en/1.7.2/README.html#a-pantry-full-of-cookiecutters)

### Running `cookiecutter`
When you run `cookiecutter` you'll be prompted with a few questions, such as:
- Full name of package creator:
```terminal
full_name [Audrey Roy Greenfeld]:
```
- As well as other information, such as email, github_username, project_name, project_slug
```terminal
...
use_pytest [n]: y
use_pypi_deployment_with_travis [y]: n
add_pyup_badge [n]: n
...
Select command_line_interface:
1 - Click
2 - Argparse
3 - No command-line interface
Choose from 1, 2, 3 [1]: 3
...
create_author_file [y]: y
```


# Version number and history

## CONTRIBUTING.md
- Either a markdown or reStructured-Text file
- Invites other developers to work on your package
- Tells them how to get started

## HISTORY.md
- Known as history, changelog or release notes
- Tells users what has changed between versions
#### Example Format for History file
```markdown
# History

## 0.3.0
### Changed
- Regression fitting sped up using NumPy...
### Deprecated
- Support ofr Python 3.5 has ended.
- `regression.regression` module removed.

## 0.2.1
### Fixed
- Fixed bug causing intercepts of zero.

## 0.2.0
### Added
- Multiple linear regression now available in new `regression.multiple_regression` module.
```
- Section for each released version
- Bullet points of the important changes
- Subsections for
	- Improvements
	- New additions
	- Bugs that have been fixed
	- Deprecations

## Version number
- Increase version number when ready for new release
- Cannot upload to PyPI a version number that has previously been used

### Changes to make each version
There are two places in the package where you need to update the version number:
- The `setup.py` file:
```python
# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
	  # ...
	  version = '0.1.1', # <------ here
	  # ...
)
```
- The top-level source `_ _ init _ _ .py` file:
```python
"""
Example title
====================================

mypackage is a ...
"""
__version__ = '0.1.1' # <------ here
```

### Using `bumpversion` to make the changes
`bumpversion` is a convenient tool to update all package version numbers

**To run it, go to the top-level of the repository and run `bumpversion`:**
- Major number (X.0.0)
```terminal
bumpversion major
```
- Minor number (0.X.0)
```terminal
bumpversion minor
```
- Patch number (0.0.X)
```terminal
bumpversion patch
```

# Makefiles and classifiers

## Classifiers
Classifiers are metadata for your package found in the `setup.py` file under `classifiers = `, which is a list of various metadata placeholders, such as:
```python
setup(
	  # ...
	  classifiers = [
		  'Development Status :: 2 - Pre-Alpha',
		  'Intended Audience :: Developers',
		  'License :: OSI Approved :: ...',
		  'Natural Language :: English',
		  'Programming Language :: Python :: 3',
		  'Programming Language :: Python :: 3.6'
	  ],
	  # ...
)
```
- Users can search for package on PyPI and when they filter for various classifiers your package will be listed at the appropriate ones.
- You should include:
	- Package status
	- Your intended audience
	- License type
	- Language
	- Versions of Python supported
- Lots more classifiers exist: [https://pypi.org/classifiers](https://pypi.org/classifiers)
## What are `Makefile`'s for?
Used to automate parts of building your package.
```markdown
mypackage/
...
|-- Makefile
```

### Inside the `Makefile`
Inside the `Makefile` you add various functions that would be used from the terminal. 
- These are similar to modules which are placeholders for a chain of terminal functions.
```Makefile
...
dist: ## builds source and wheel package
	python3 setup.py sdist bdist_wheel

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/

test: ## run tests quickly using pytest
	pytest

release: dist ## package and upload a release
	twine upload dist/*
```

### How do I use the `Makefile`?
To use the `Makefile` navigate to the top-level of the repository and in terminal type:
```terminal
make <function-name>
```
- The function name is any of the functions you created inside the `Makefile`.
- You can also get a summary of all the functions in the `Makefile`, using:
```terminal
make help
...
dist: builds source and wheel package
...
```

# Recap
- Modules vs subpackages vs packages
- Package structure and `_ _ init _ _ .py`
- Absolute and relative imports
- Documentation with `pyment`
- Code style with `flake8`
- Making your package installable with `setup.py`
- Dependencies with `install_requires` and `requirements.txt`
- Supporting files like `LICENSE`, `README.md`, `CONTRIBUTING.md` and `HISTORY.md`
- Building and uploading distributions to PyPI with `twine`
- Testing with `pytest` and `tox`
- Using package templates with `cookiecutter`
- Efficient package care with `Makefile`'s

