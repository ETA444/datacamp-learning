
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
