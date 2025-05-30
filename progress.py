from datetime import timedelta, datetime
from time import time
from storage import save_txt


def to_str(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_progress(what, start, count, total):
    if count == 0:
        return f'{what}: {total} {to_str(start)}'

    if count == total:
        return f'{what}: {total} {to_str(int(time()))}'

    past = int(time()) - start
    speed = round(past / count, 2)
    left = int(total * speed) - past

    past = str(timedelta(seconds=past))
    left = str(timedelta(seconds=left))

    return f'{what}: {count}/{total} [{past}<{left} {speed}s/it]'


def save_progress(what, total, language, year=None, license=None):
    start = int(time())

    def wrapper(count):
        progress = get_progress(what, start, count, total)
        save_txt('progress', progress, language, year, license)

    return wrapper
