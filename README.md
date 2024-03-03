# CIS 583 Deep Learning Project

## IMPORTANT: Please make your own branch and switch to it!

Before making changes to any existing files:
1) Open a terminal 
2) Ensure terminal is at this repository's root level. `ls` should reveal files like environment.yml, utils.py, etc.
3) Double-check that you have the latest code using `git pull`
4) Double-check the branch you are currently on with `git branch` - if you are already on your own custom branch, **NOT main**, then you are all set.
5) If you are not yet on your own custom branch, you can:

    * Create a new branch: `git checkout -b name_of_your_new_branch`
    * Move to an existing branch: `git checkout name_of_your_existing_branch`

6) After adding and committing your changes, test where possible and reasonable to check for breaking changes. If there seem to be none, submit a pull request to merge in your edits.

## Conda Virtual Environment Setup

0) Ensure conda is installed (see [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))
1) Open a terminal
2) Ensure terminal is at this repository's root level. `ls` should reveal environment.yml
3) Enter the following into the terminal: 
    `conda env create -f environment.yml`


## GitHub Token Setup

This may or may not be needed - do not complete these steps at first, and only come back if necessary.

1) Click on your profile image at top right in GitHub site
2) Select "Settings"
3) Scroll down to "Developer settings" on the left bar
4) On left bar select "Personal access tokens", then "Tokens (classic)"
5) At top, select "Generate new token (classic)"
6) The Note may be left blank. Ensure the Expiration will last until the semester's end (unless you wish to repeat this process before the course wraps up). 
7) Select the scope "repo"
8) Scroll down and select "Generate token"
9) Copy the token text and keep it safe. Use it if prompted for a password from the terminal.