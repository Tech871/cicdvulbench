
Building a Dataset from GitHub

## What Happens

* `repos.py`: Downloads repositories by year, language, and license
* `pulls.py`: Extracts pull requests from them
* `targets.py`: Determines vulnerability based on pull request descriptions
* `commits.py`: Retrieves commits for pull requests that fix vulnerabilities
* `files.py`: Gets the file with the vulnerability before and after the change
* `rows.py`: For each language and vulnerability, generates a training dataset

## How to Run

Get a GitHub API token at [https://github.com/settings/tokens](https://github.com/settings/tokens) with read permissions.

Create a `.env` file in the root of the project listing tokens separated by commas with no spaces:

`GITHUB_TOKENS=x,y,z`

## Where to Look

Intermediate results can be found in the folder `tmp/language/year/license`:

* `progress.txt` — execution progress
* `repos.json` — found repositories
* `pulls.json` — collected closed pull requests
* `targets.json` — vulnerabilities identified from pull request descriptions
* `commits.json` — pull request commits with vulnerability fixes
* `files.json` — files before and after with the proposed vulnerability fix

