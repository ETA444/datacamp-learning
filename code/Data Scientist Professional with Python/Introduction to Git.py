# Code Exercises from Introduction to Git #

## Chapter 1

### --- Exercise 1 --- ###

""" Find how many files are in data directory:
$ ls
data report.md
$ cd data
$ ls
$ mental_health_survey.csv

Answer: 1
"""


### --- Exercise 2 --- ###

""" Find git version:
$ git --version
git version 2.17.1
"""



### --- Exercise 3 --- ###

""" Where is information about the history 
of the files in /home/repl/mh_survey/data stored?

$ ls -a
.. .git data report.md

Answer .git
"""


### --- Exercise 4 --- ###

"""Add a new row of data 
at the end of mental_health_survey.csv 
containing: "49,M,No,Yes,Never,Yes,Yes,No"
$ echo "49,M,No,Yes,Never,Yes,Yes,No" >> mental_health_survey.csv

Place the updated file in the staging area.
$ git add mental_health_survey.csv

Commit the modified file with the log message 
"Adding one new participant's data"
$ git commit -m "Adding one new participant's data"

"""


### --- Exercise 5 --- ###

"""Check which files are in the staging 
area but not yet committed.
$ git status

Add all files in your current directory and 
all subdirectories into the staging area.
$ git add .

Commit all files in the staging area 
with the log message "Added 3 participants
and a new section in report"
$ git commit -m "Added 3 participants and a new section in report"

"""


### --- Exercise 6 --- ###

"""How many lines have been added to the current 
version of mental_health_survey.csv compared 
to the version in the latest commit?
$ git diff mental_health_survey.csv
"""

### --- Exercise 7 --- ###

"""
Use a command to see how all files 
differ from the last saved revision.
$ git diff -m HEAD

Use a Git command to add report.md to the staging area.
$ git add report.md

Commit all files with the log message
"New participant data and reminder for analysis"
$ git commit -m "New participant data and reminder for analysis"
"""



## Chapter 2

### --- Exercise 1 --- ###

"""
Using the console, run a command 
to find the hash of the commit that 
added a summary report file.

$ git log
(space)
(q)

Answer: e39ecc89
"""


### --- Exercise 2 --- ###

"""
Use a command to display the repo's history.
$ git log 

$ git show 36b761
"""


### --- Exercise 3 --- ###

"""
Use an appropriate command to find out how
 current versions of files compare to
  the second most recent commit.

$ git show HEAD~1
"""


### --- Exercise 4 --- ###

"""Which files were modified between 
 the fourth most recent and
  second most recent commits?

$ git diff HEAD~3 HEAD~1
"""


### --- Exercise 5 --- ###

"""Display line-by-line changes
 and associated metadata for report.md.

$ git annotate report.md
"""


### --- Exercise 6 --- ###

"""
Unstage mental_health_survey.csv.
$ git reset HEAD mental_health_survey.csv

Add 41,M,Yes,No,No,No,Often,Yes to the end of mental_health_survey.csv.
$ echo "41,M,Yes,No,No,No,Often,Yes" >> mental_health_survey.csv

Restage this file.
git add mental_health_survey.csv

Make a commit with the log message "Extra participant"
git commit -m "Extra participant"
"""


### --- Exercise 7 --- ###


"""
Undo the changes made to report.md.
$ git checkout -- report.md
"""

### --- Exercise 8 --- ###

"""
Remove all files from the staging area.
$ git reset HEAD

Undo changes to all unstaged files since the last commit.
$ git checkout .
"""

### --- Exercise 9 --- ###
"""
Use a command to restore all files to 
their version located in the commit with 
a hash starting 7f71eade.

$ git checkout 7f71eade

"""

### --- Exercise 10 --- ###
"""
Display the last two commits for the report file
$ git log -2 report.md

Use the commit hash to restore the version of report.md 
from the second most recent commit.
$ git checkout e39ecc89 report.md

Put the restored version of report.md into the staging area.
$ git add report.md

Commit the restored file with a log message of "Restoring version from commit e39ecc8"
$ git commit -m "Restoring version from commit e39ecc8"

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


