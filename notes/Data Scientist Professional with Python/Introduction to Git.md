
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


# Configuring Git
Git has customizable settings to speed up or improve how we work.

## Levels of settings
To access a list of customizable settings we use:
- `git config --list`

Git has three levels of settings:
1. `--local`: settings for one specific project
2. `--global`: settings for all of our projects
3. `--system`: settings for every user on this computer

### What can we configure
```shell
git config --list
```
```output
user.email=repl@datacamp.com
user.name=Rep Loop
core.editor=nano
...
```
- `user.email` and `user.name` are needed by some commands, so setting these saves time.
	- Both of these are global settings

### Changing our settings
We can use the following syntax to modify settings:
```shell
git config --global user.name User1

git config --global user.name 'User 1' # if spaces

git config --global user.email user1@email.com
```

### Using an alias
- Set up an alias through global settings
- Typically used to shorten a command

For example to create an alias for committing files by executing `ci`:
```shell
git config --global alias.ci `commit -m`
```
We can now commit files using:
```shell
git ci 'Commit msg'
```

#### Creating a custom alias
- We can create an alias for any command

For example, if we often unstage files we can create:
```shell
git config --global alias.unstage 'reset HEAD'
```

### Tracking aliases
Git helps us track our aliases in the `.gitconfig` file.

We can access it like this:
```shell
git config --global --list
```
```output
alias.ci=commit -m
alias.unstage=reset HEAD
```

## Ignoring certain files
We can instruct git to ignore specific files using the `.gitignore` file.

For example, we could ignore any files with ext `.log`:
```gitignore
*.log
```
- Commonly ignored files: APIs, credentials, system files, software dependencies.


# Branches
Git uses **branches** to systematically track multiple versions of files.
- In different branches some files might be the same while others different, others don't exist

## Source and destination
When merging two branches:
- the commits are called parent commits
	- `source` - the branch we want to merge **from**
	- `destination` - the branch we want to merge **into**

## Identifying branches
We use `git branch` to see the branches in our repo.
```shell
git branch
```
```output
  alter-report-title
  main
* summary-statistics
```
- * = current branch

- **To create a new branch:** `git checkout -b branchname`
```shell
git checkout -b report
```

- **To see the difference between branches:**
```shell
git diff main summary-statistics
```


# Working with branches

## Why do we need to switch branches?

A branch in a Git repository is like a separate line of development. It allows you to work on new features, bug fixes, or experiments without affecting the main or default branch (often called `main` or `master`).
- You can make changes in a branch, experiment, and make sure everything works as expected. If you're satisfied with your work, you can then merge the changes from your branch into the main branch, making those changes part of the official project.

So, branches provide a way to isolate different streams of work, making it easier to manage changes and collaborate with others in a controlled manner. They also allow you to experiment without immediately affecting the main project, and once you're confident in your changes, you can integrate them back into the main branch.

- Common to work on different components of a project simultaneously
- Allows us to keep making progress concurrently

### Switching branches
- `git checkout -b new_branch` to create a new branch

To switch branches, we use `checkout` but without a flag:
```shell
git checkout other_branch
```

### Why do we merge branches?
- `main` = ground truth
- Each branch should be for a specific task
- Once the task is complete we should merge our changes into `main`

#### Merging branches
```shell
git merge source destination
# example test into main
git merge test main
```

# Handling conflict

# Branch Conflicts and Best Practices

## Branch Conflicts

In Git, a conflict between branches occurs when changes made in one branch conflict with changes made in another branch. This often happens when both branches modify the same part of a file, leading Git to be unsure of which changes to incorporate.

## Best Practices to Avoid Conflicts

1. **Frequent Pulls:**
   - Regularly pull the latest changes from the main branch to your working branch to stay updated with the latest modifications.

2. **Small, Focused Commits:**
   - Break your work into small, focused commits. This reduces the chances of conflicting changes in a single commit.

3. **Communication:**
   - Communicate with your team to understand ongoing changes and coordinate efforts to minimize conflicts.

4. **Use Feature Branches:**
   - Create feature branches for specific tasks. This isolates your changes from the main branch until you're ready to merge.

5. **Testing:**
   - Test your changes thoroughly before merging. This helps catch issues early and reduces the chances of conflicts.

6. **Git Pull with Rebase:**
   - Instead of a standard `git pull`, consider using `git pull --rebase`. This integrates changes from the main branch more smoothly.

7. **Git Merge Strategies:**
   - Explore and understand different Git merge strategies, such as fast-forward, recursive, or octopus, and choose the one that suits your workflow.

8. **Merge Locally First:**
   - Merge the main branch into your feature branch locally before pushing changes. Resolve any conflicts locally to avoid pushing conflicted changes.

By following these best practices, you can minimize the likelihood of conflicts between branches and streamline your collaborative development process.
