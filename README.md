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

### Method 1: Conda Env from Scratch
2) Enter the following into the terminal:
    `conda create -n cis583_env python=3.10 ipykernel ipython joblib matplotlib nltk pandas scikit-learn scipy spacy seaborn tensorflow tensorboard`
3) When prompted, type in 'y' or 'yes' to confirm the packages to be installed.

### Method 2: Conda Env from Yaml
2) Ensure terminal is at this repository's root level. `ls` should reveal environment.yml
3) Enter the following into the terminal, and type in 'y' to confirm installation as needed: 
    `conda env create -f environment.yml`

Once conda installs all packages, regardless of the method you went with, check that it worked:
4) Open a Jupyter Notebook file (.ipynb). In Visual Studio Code, at the upper right-hand corner, click "Select Kernel."
5) Select "From Python environment" and cis583_env (or however you chose to name the environment) should now be available to select.
6) Once the kernel is chosen and now shows at the Notebook's upper right corner, run Notebook cells to confirm kernel is working properly and has all necessary packages. 
7) If any packages are missing, or further packages are needed, return to the terminal and type: `conda activate cis583_env` (or the name you chose) then  `conda install <package1> <package2>`

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