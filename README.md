# CIS 583 Deep Learning Project

## Conda Virtual Environment Setup

0) Ensure conda is installed (see [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))
1) Open a terminal
2) Ensure terminal is at this repository folder's root level. `ls` should reveal environment.yml
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