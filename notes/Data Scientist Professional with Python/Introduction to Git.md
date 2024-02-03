
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


# Storing data with Git

## Commit Structure in Git
Git commits consist of three main parts:

1. **Commit:**
   - Contains metadata about the commit.

2. **Tree:**
   - Tracks the names and locations of files in the repository.

3. **Blob:**
   - Stands for binary large object.
   - May contain data of any kind.
   - Represents a compressed snapshot of a file's contents.

### Git  log
We can use `git log` to view the history of commits.
- Press `space` to show more recent commits
- Press `Q` to quit the log and return to the terminal

#### Finding a particular commit
We need the first 6-8 characters of the `hash` to identify a commit with `git show`:
```shell
git show c27fa856
```

# Viewing changes
## The `HEAD` shortcut
- Compares staged files to the version in the last commit
```shell
git diff -r HEAD~1
```
- Use a tilde `~` to pick a specific commit to compare versions to:
	- `HEAD~1` = the second most recent commit
	- `HEAD~2` = the commit before that

### Using `HEAD~` with `git show`
```shell
git show HEAD~3
```

## Using `git diff` to compare commits
For example to compare the fourth and third most recent commits:
```shell
git diff HEAD~3 HEAD~2
```

## Changes per document by line
```shell
git annotate report.md
```
We get output in one line in the following format:
- Hash
- Author
- Time
- Line #
- Line Content


## Summary
- `git show HEAD~1`: Show what changed in the second most recent commit.
- `git diff 35f4b4dd 186398f`: Show changes between two commits using hashes
- `git diff HEAD~1 HEAD~2`: Show changes between two commits using `HEAD`
- `git annotate file`: Show line-by-line changes and associated metadata

# Undoing changes before committing 

## Unstaging all files
- To unstage all files:
```shell
git reset HEAD
```

## Undo changes to an unstaged file
- Suppose we need to undo changes to a file in the repository:
```shell
git checkout -- summary_statistics.csv
```
- `checkout` means switching to a different version
	- Default is the last commit
- This means losing all changes made to the unstaged file forever

To **undo changes to all unstaged files:**
```shell
git checkout .
```
- This command will undo changes to all unstaged files in the current directory (`.`)

# Reverting the whole repo and saving to previous state

```shell
git reset HEAD

git checkout .

git add .

git commit -m "Restored repo to previous commit"
```


# Restoring and reverting

## Customizing the `log` output
- We can restrict the number of commits displayed using `-`:
```shell
git log -3
```
- To only look at the commit history of one file:
```shell
git log -3 report.md
```
- Restrict `git log` by date:
```shell
git log --since='Month Day Year'

git log --since='Apr 2 2022'

git log --since='Apr 2 2022' --until='Apr 11 2022'
```

### Restoring to an old version of a file
```shell
# first get the commit hash
git log --since='Jul 6 2022'
# we see that it is commit dc9d8f

# restore to this version
git checkout dc9d8f mental_health_survey.csv

# if wanted to revert the whole repo to that ver
git checkout dc9d8f
```

## Cleaning a repo
- See what files are not being tracked:
```shell
git clean -n
```
- To delete these files (cannot be undone):
```shell
git clean -f
```

