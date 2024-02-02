
# Introduction to version control with Git

## What is a version?
1. Contents of a file at a given point in time
2. Metadata: information associated with the file
	- Author
	- Location
	- File Type
	- Last Saved

## What is version control?
Version control is a **group of systems and processes to manage changes made** to documents, programs, and directories.
- Useful for anything that changes over time
- Good for anything that needs to be shared

You can achieve various goals with version control:
- Track files in different states
- Simultaneous file development (continuous development)
- Combine different version of files
- Identify a particular version
- Revert changes

## Git
**Git** is a popular version control system for computer programming and data projects.
- Open source
- Scalable
- Note: Git is not GitHub, but Git can be used with GitHub

### Benefits of Git
- Git stores everything so nothing is lost
- Git notifies us when there is a conflicting content in files
- Git synchronizes across different people and computers

### Using Git
- Git commands are run on the shell, also known as terminal
- The **shell:** 
	- is a program for executing commands
	- Can be used to easily preview, modify, or inspect files and directories

#### Useful shell commands
- `pwd`: prints path of current wd
```shell
pwd
```
- `ls`: prints list of contents in current directory
```shell
ls
```
- `cd`: change directory
```shell
cd
```
- `git --version`: check version of git
```shell
git --version
```
- `echo "..." >> filename.txt`: new line to existing file
```shell
echo "49,M,No,Yes,Never,Yes,Yes,No" >> mental_health_survey.csv
```
- `echo "..." > filename.txt`: make a new file with line
- `nano`: open file in lightweight text editor in console

# Saving files

## Staging and committing
- Saving a draft
	- **Staging area**
	- *'Like putting a letter in an envelope'*
- Save files/update the repo
	- **Through commits**
	- *'Like putting an envelope in a mailbox'*

### Accessing the .git dir
Directories like `.git` are hidden directories, to access them you need to:
```shell
ls -a
```

### Saving a file (staging): `git add`
- `git add file.txt`: Adding a single file
```shell
git add report.md
```
- `git add .`: Adding all modified files
```shell
git add .
```

### Committing a file: `git commit`
- `git commit -m`: with the -m flag we indicate the message of the commit, e.g. "Updated report.md"
```shell
git commit -m "Updated Introduction to Git.md"
```

### Check the status of files
It is useful to know which files are in the staging area and which files have changes that are not in the staging area yet. 
```shell
git status
```

# Comparing files

## Comparing a single file
- `git diff filename.txt`: compare an unstaged file with the last committed version
```shell
git diff filename.txt
```
- `git diff -r HEAD filename.txt`: compare a staged file with the last committed version
```shell
git diff -r HEAD filename.txt
```
