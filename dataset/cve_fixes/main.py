import pandas as pd

import sqlite3
from collections import defaultdict
from storage import save_csv, get_path


def load(language):
    sqlite_connection = None

    try:
        path = get_path('src')
        sqlite_connection = sqlite3.connect(f'{path}/cve_fixes.db')
        cursor = sqlite_connection.cursor()

        sqlite_select_query = f"""SELECT f.code_after, f.code_before, cc.cwe_id
FROM file_change f, commits c, fixes fx, cve cv, cwe_classification cc
WHERE f.hash = c.hash
AND c.hash = fx.hash
AND fx.cve_id = cv.cve_id
AND cv.cve_id = cc.cve_id
AND f.programming_language='{language.capitalize()}';"""

        cursor.execute(sqlite_select_query)
        records = cursor.fetchall()
        print('rows:', len(records))

        cursor.close()
        return records

    except sqlite3.Error as err:
        exit(f'sqlite error: {err}')
    finally:
        if sqlite_connection:
            sqlite_connection.close()


def read(language):
    records = load(language)

    rows = defaultdict(list)
    for before, after, cwe in records:
        if cwe.startswith('CWE-'):
            cwe = int(cwe[4:])

            if before != 'None':
                rows[cwe].append((1, before))
            if after != 'None':
                rows[cwe].append((0, after))
    return rows


def save(language, rows):
    for target, rows in rows.items():
        dataset = pd.DataFrame(rows, columns=['target', 'code'])

        save_csv('code', dataset['code'], language, target, source='cve_fixes')
        save_csv('target', dataset['target'], language, target, source='cve_fixes')


def main(languages):
    for language in languages:
        print('-', language)

        rows = read(language)
        save(language, rows)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=str, default='java')

    args = parser.parse_args()

    main(args.languages.split(','))
