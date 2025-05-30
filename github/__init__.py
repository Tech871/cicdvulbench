import requests
import sys
import os

from base64 import b64decode
from time import sleep
from dotenv import load_dotenv
import re

load_dotenv()

GITHUB_API_URL = 'https://api.github.com'


def get_tokens():
    return os.getenv('GITHUB_TOKENS').split(',')


def get_token():
    return get_tokens()[0]


def get_headers(token):
    return {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }


def get_res(url, token, params=None):
    headers = get_headers(token)
    try:
        sleep(1)
        res = requests.get(url, params, headers=headers)
        if res.status_code == 502:
            sleep(2)
            res = requests.get(url, params, headers=headers)
        if res.status_code == 200:
            return res
        res.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f'request error: {e}', file=sys.stderr)


def get_query(language, license, start_date, finish_date):
    return f'language:{language} license:{license} created:{start_date}..{finish_date}'


def get_total_count(query, token):
    res = get_res(f'{GITHUB_API_URL}/search/repositories', token, {'q': query})
    return res.json()['total_count'] if res else 0


def get_page_repos(page, query, token):
    params = {
        'per_page': 100,
        'page': page,
        'q': query
    }
    res = get_res(f'{GITHUB_API_URL}/search/repositories', token, params)
    if not res:
        return []

    repos = res.json().get('items', None)
    if not repos:
        print(f'no repos for {query} page={page}', file=sys.stderr)
        return []

    return [
        {
            'full_name': repo['full_name'],
            'description': repo['description'],
            'forks': repo['forks'],
            'watchers': repo['watchers'],
            'open_issues': repo['open_issues'],
            'size': repo['size']
        }
        for repo in repos
    ]


def get_repos(query, token, per_page=100):
    repos = []

    page = 1
    while True:
        page_repos = get_page_repos(page, query, token)
        repos += page_repos

        if len(page_repos) < per_page:
            break

        page += 1

    return repos


def get_pulls(repo, token):
    repo = repo['full_name']

    res = get_res(f'{GITHUB_API_URL}/repos/{repo}/pulls?state=closed', token)
    if not res:
        return []

    pulls = res.json()
    if not pulls:
        print(f'no pulls in {repo}', file=sys.stderr)
        return []

    return [
        {
            'repo': repo,
            'pull': pull['number'],
            'title': pull['title'],
            'body': pull['body'],
            'author': pull['user']['login'],
            'merged': pull['merged_at'],
            'created': pull['created_at']
        }
        for pull in pulls
    ]


def get_commits(vul, token):
    repo = vul['repo']
    pull = vul['pull']

    res = get_res(f'{GITHUB_API_URL}/repos/{repo}/pulls/{pull}/commits', token)

    if not res:
        return []

    commits = res.json()
    if not commits:
        print(f'no commits in {repo}/pulls/{pull}', file=sys.stderr)
        return []

    return [
        {
            'sha': commit['sha'],
            'message': commit['commit']['message'],
        }
        for commit in commits
    ]


def get_previous_sha(repo, sha, token):
    res = get_res(f'{GITHUB_API_URL}/repos/{repo}/commits/{sha}', token)
    if res:
        parents = res.json().get('parents', None)
        if parents:
            return parents[0]['sha']


def get_file(repo, filename, sha, token):
    res = get_res(f'https://api.github.com/repos/{repo}/contents/{filename}?ref={sha}', token)
    if res:
        return b64decode(res.json()['content']).decode('utf-8')

    print(f'{res.status_code}: can\'t get content for {repo}/{filename}')


def get_before_after_sha(repo, commits, token):
    before_sha = get_previous_sha(repo, commits[0]['sha'], token)
    after_sha = commits[-1]['sha']

    return before_sha, after_sha


def get_compared_files(repo, before_sha, after_sha, token):
    res = get_res(f'{GITHUB_API_URL}/repos/{repo}/compare/{before_sha}...{after_sha}', token)
    return res.json()['files'] if res else []


def get_sha(url):
    pattern = re.compile(r'https://github\.com/[\w\-_]+/[\w\-_]+/commit/([0-9a-f]{40})')
    match = pattern.match(url)

    if match:
        return match.group(1)

    pattern = re.compile(r'https://github\.com/[\w\-_]+/[\w\-_]+/pull/\d+/commits/([0-9a-f]{40})')
    match = pattern.match(url)

    if match:
        return match.group(1)


def get_repo(url):
    pattern = re.compile(r'https://github\.com/([\w\-_]+/[\w\-_]+)/.*')
    match = pattern.match(url)

    if match:
        return match.group(1)


def get_pull(url):
    pattern = re.compile(r'https://github\.com/[\w\-_]+/[\w\-_]+/pulls?/(\d+).*')
    match = pattern.match(url)

    if match:
        return int(match.group(1))


def get_ext(filename):
    return filename.rsplit('.')[-1]


EXT_LANGUAGE = {
    'java': 'java',
    'cpp': 'c++',
    'h': 'c++',
    'py': 'python',
    'go': 'go'
}


def is_code(language, filename):
    return EXT_LANGUAGE.get(get_ext(filename)) == language


def is_modified(language):
    def wrapper(file):
        return file['status'] == 'modified' and is_code(language, file['filename'])

    return wrapper


def get_change_score(file):
    return max(file['additions'], file['deletions']) / 10 + min(file['additions'], file['deletions'])
