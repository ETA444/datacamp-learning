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

""" 

Display all settings
$ git config --list

Change the email address to I_love_Git@datacamp.com.
$ git config --global user.email I_love_Git@datacamp.com

Check the global settings to see if the update has been made.
$ git config --global --list
"""


### --- Exercise 2 --- ###

"""
Create an alias for the global command used to check the state of files, calling it st.
$ git config --global alias.st "status"

Run the new alias command to confirm it works
$ git st

"""


### --- Exercise 3 --- ###

"""
Edit summary_statistics.txt and add the following text:
 "Mean age is 32 years old, with a standard deviation of 6.72".
$ echo "Mean age is 32 years old, with a standard deviation of 6.72" >> summary_statistics.txt

Add summary_statistics.txt to the staging area.
$ git add summary_statistics.txt

Make a commit with the log message "Adding age summary statistics".
$ git commit -m "Adding age summary statistics"

Create a new branch called report.
$ git checkout -b report
"""



### --- Exercise 4 --- ###

"""
How many branches are there in the repo
$ git branch
"""


### --- Exercise 5 --- ###

"""
Execute a command to compare the 
alter-report-title and summary-statistics branches.
$ git diff alter-report-title summary-statistics
"""


### --- Exercise 6 --- ###

"""
Switch to the report branch
$ git checkout report

Add "80% of participants were 
male, compared to the industry average of 
67%." to the end of report.md.
$ echo "80% of participants were male, compared to the industry average of 67%." >> report.md

Place report.md into the staging area.
$ git add report.md

Make a commit with the log message "Add gender summary in report".
$ git commit -m "Add gender summary in report"
"""


### --- Exercise 7 --- ###

"""
Merge the report branch into the main branch.
$ git merge report main

"""


### --- Exercise 8 --- ###

"""
Edit report.md, removing content from
the summary_statistics branch along
with any Git conflict syntax.
$ nano report.md
"""


## Chapter 4

### --- Exercise 1 --- ###

"""
Create a Git repo called anxiety_workplace 
in your current directory.
$ git init anxiety_workplace

Move into the new directory
$ git anxiety_workplace

Create a file called todo.txt containing 
the following text: "TODO: Recap on existing research.".
$ echo "TODO: Recap on existing research." > todo.txt

"""


### --- Exercise 2 --- ###

"""
Turn your project into a git repo
$ git init

"""



### --- Exercise 3 --- ###

"""
Clone /home/john/repo, naming the cloned repo as john_anxiety_project.
$ git clone /home/john/repo john_anxiety_project

"""



### --- Exercise 4 --- ###

"""
Add the name john for the /home/john/repo repo
$ git remote add john /home/john/repo

List all remotes including their URL(s).
$ git remote -v
"""



### --- Exercise 5 --- ###

"""
Run a command to find out the name(s) of remote 
repos linked to your project.
$ git remote -v

Gather contents from the remote origin repo
into your local main branch.
$ git fetch origin main

Compare the remote repo with your local main branch.
$ git diff origin main
"""



### --- Exercise 6 --- ###

"""
Use a single command to fetch and merge 
the origin repo into your local main branch.
$ git pull origin main

Append "No existing mental health diagnosis."
to the end of protocol.md.
$ echo "No existing mental health diagnosis." >> protocol.md

Add protocol.md to the staging area
$ git add protocol.md

Make a commit with the log message
"Updating eligibility criteria".
$ git commit -m "Updating eligibility criteria"

"""



### --- Exercise 7 --- ###

"""
Add the two updated files to the staging area.
$ git add .

Save changes with the log message
"Budget inaccuracy added to the issue log and report".
$ git commit -m "Budget inaccuracy added to the issue log and report"

Update the origin remote with changes made in your local main branch
$ git push origin main
"""



### --- Exercise 8 --- ###

"""
Send changes in the local repo's main
branch to the remote repo called origin.
$ git push origin main

Pull origin into the local main branch,
entering a message of your choice inside the text editor.
$ git pull origin main

Try again to update the remote with changes made in the local main.
$ git push origin main
"""