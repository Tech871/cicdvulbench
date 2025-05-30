from progress import save_progress
from storage import save_json, has_json

from github import get_total_count, get_repos, get_query, get_token

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def split_date_range(start_date, finish_date, interval_days=1):
    date_ranges = []

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    finish_date = datetime.strptime(finish_date, "%Y-%m-%d")

    current_date = start_date
    while current_date < finish_date:
        next_date = min(current_date + relativedelta(days=interval_days) - timedelta(days=1), finish_date)
        date_ranges.append((current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')))
        current_date = next_date + timedelta(days=1)

    return date_ranges


def get_start_date(year):
    return f'{year}-01-01'


def get_finish_date(year):
    return min(f'{year}-12-31', datetime.now().strftime("%Y-%m-%d"))


def main(language, token, year, license):
    if has_json('repos', language, year, license):
        return

    start_date = get_start_date(year)
    finish_date = get_finish_date(year)

    query = get_query(language, license, start_date, finish_date)
    total = get_total_count(query, token)

    progress = save_progress('repos', total, language, year, license)

    repos = []
    for start_date, finish_date in split_date_range(start_date, finish_date):
        query = get_query(language, license, start_date, finish_date)
        repos += get_repos(query, token)
        progress(len(repos))

    save_json('repos', repos, language, year, license)


if __name__ == '__main__':
    main('java', get_token(), 2025, 'mit')
