import json
import os
import io
import shutil
import tempfile
from io import TextIOWrapper

from botocore.response import StreamingBody
from botocore.exceptions import ClientError
from transformers import TrainerCallback

from typing import BinaryIO

import pandas as pd
import numpy as np

import boto3
import torch


def remove_root(path):
    root = os.getenv('ROOT_DIR', os.path.dirname(os.path.abspath(__file__)))
    return path.removeprefix(root).removeprefix('/')


class S3:
    endpoint_url = 'https://storage.yandexcloud.net'

    def __init__(self, bucket):
        session = boto3.Session()
        self.client = session.client(service_name='s3', endpoint_url=self.endpoint_url)
        self.resource = session.resource(service_name='s3', endpoint_url=self.endpoint_url)
        self.bucket = bucket
        self.cached_dirs = {}

    def open(self, path, mode=None) -> StreamingBody | TextIOWrapper:
        key = remove_root(path)
        body = self.resource.Object(bucket_name=self.bucket, key=key).get()['Body']
        return body if mode == 'b' else io.TextIOWrapper(body, encoding='utf-8')

    def write(self, path, buf: BinaryIO):
        key = remove_root(path)
        self.client.upload_fileobj(Bucket=self.bucket, Key=key, Fileobj=buf)

    def write_str(self, path, s):
        key = remove_root(path)
        body = io.BytesIO(s.encode('utf-8'))
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)

    def write_bytes(self, path, b: bytearray):
        key = remove_root(path)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=io.BytesIO(b))

    def exists(self, path):
        key = remove_root(path)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise

    def get_cache_dir(self, path):
        prefix = remove_root(path)

        if prefix not in self.cached_dirs:
            tmp = tempfile.mkdtemp(dir='/tmp')

            bucket = self.resource.Bucket(self.bucket)
            for obj in bucket.objects.filter(Prefix=prefix):
                filename = os.path.join(tmp, obj.key.removeprefix(prefix + '/'))
                dirname = os.path.dirname(filename)
                os.makedirs(dirname, exist_ok=True)
                bucket.download_file(obj.key, filename)

            self.cached_dirs[prefix] = tmp

        return self.cached_dirs[prefix]

    def clean_cache(self):
        for v in self.cached_dirs.values():
            shutil.rmtree(v, ignore_errors=True)
        self.cached_dirs.clear()

    def remove_dir(self, path):
        prefix = remove_root(path)
        bucket = self.resource.Bucket(self.bucket)

        for obj in bucket.objects.filter(Prefix=prefix):
            obj.delete()

        bucket.Object(prefix).delete()

    def upload_dir(self, path):
        prefix = remove_root(path)
        for root, _, files in os.walk(path):
            for file in files:
                filename = os.path.join(root, file)
                key = filename.removeprefix(path + '/')
                key = os.path.join(prefix, key)
                try:
                    self.client.upload_file(filename, self.bucket, key)
                except FileNotFoundError:
                    print('No such file or directory:', filename)

    def get_last_checkpoint(self, path):
        key = remove_root(path)
        objects = self.resource.Bucket(self.bucket).objects.filter(Prefix=key)
        paths = [o.key.removeprefix(key + '/').split('/', 1)[0] for o in objects]

        checkpoints = [p for p in paths if p.startswith('checkpoint-')]
        assert len(checkpoints)

        checkpoint = max(int(c.removeprefix('checkpoint-')) for c in checkpoints)
        return f'{key}/checkpoint-{checkpoint}'


def get_s3(args):
    return args.s3_bucket and S3(args.s3_bucket)


def to_str(arg):
    return ','.join(map(str, arg)) if isinstance(arg, list) else str(arg)


def get_path(*args):
    root = os.getenv('ROOT_DIR', os.path.dirname(os.path.abspath(__file__)))
    paths = map(to_str, filter(bool, args))
    path = os.path.join(root, *paths)
    os.makedirs(path, exist_ok=True)
    return path


LANGUAGE_EXT = {
    'java': 'java',
    'c++': 'cpp',
    'python': 'py',
    'go': 'go'
}


def save_code(code, language, s3=None):
    ext = LANGUAGE_EXT[language]
    path = get_path('tmp')
    file_path = f'{path}/code.{ext}'

    if s3:
        s3.write_str(file_path, code)
    else:
        with open(file_path, 'w', encoding='utf-8') as dst:
            dst.write(code)

    return file_path


def save_txt(what, values, language, year, license, s3=None):
    path = get_path('tmp', language, year, license)
    path = f'{path}/{what}.txt'
    print('save_txt:', path)
    if s3:
        s3.write_str(path, values)
    else:
        with open(f'{path}/{what}.txt', 'w', encoding='utf-8') as dst:
            dst.write(values)


def save_json(what, values, language, year=None, license=None, s3=None):
    print(what, len(values))
    path = get_path('tmp', language, year, license)
    path = f'{path}/{what}.json'
    print('save_json:', path)
    if s3:
        buf = io.StringIO()
        json.dump(values, buf, indent=4, ensure_ascii=False)
        s3.write(path, io.BytesIO(buf.getvalue().encode('utf-8')))
    else:
        with open(path, 'w', encoding='utf-8') as dst:
            json.dump(values, dst, indent=4, ensure_ascii=False)


def has_json(what, language, year=None, license=None, s3=None):
    path = get_path('tmp', language, year, license)
    path = f'{path}/{what}.json'
    return s3.exists(path) if s3 else os.path.exists(path)


def load_json(what, language, year=None, license=None, s3=None):
    path = get_path('tmp', language, year, license)
    path = f'{path}/{what}.json'
    with get_src(path, s3) as src:
        return json.load(src)


def get_src(path, s3=None):
    return s3.open(path) if s3 else open(path, encoding='utf-8')


def load_jsonl(language, source, s3=None):
    path = get_path('src', language)
    path = f'{path}/{source}.jsonl'
    with get_src(path, s3) as src:
        return [json.loads(record) for record in src]


def get_dir(
        language, target, where='dst',
        source=None, train=None, test=None, classifier=None
):
    return get_path(where, language, target, source, train, test, classifier)


def save_csv(
        what, values, language, target, where='dst',
        source=None, train=None, test=None, classifier=None,
        s3=None
):
    path = get_path(where, language, target, source, train, test, classifier)
    path = f'{path}/{what}.csv'
    print('save_csv:', path)

    if s3:
        buf = io.StringIO()
        values.to_csv(buf, index=False, header=False)
        s3.write(path, io.BytesIO(buf.getvalue().encode('utf-8')))
    else:
        values.to_csv(path, index=False, header=False)


def load_csv(
        what, language, target, where='dst',
        source=None, train=None, test=None, classifier=None,
        s3=None
):
    path = get_path(where, language, target, source, train, test, classifier)
    path = f'{path}/{what}.csv'
    try:
        with get_src(path, s3) as src:
            data = pd.read_csv(src, header=None).values
    except Exception as e:
        print(e)
        return np.array([])

    return data if data.shape[1] > 1 else data.ravel()


def get_last_checkpoint(output_dir):
    checkpoints = [c for c in os.listdir(
        output_dir) if c.startswith('checkpoint-')]
    assert len(checkpoints)

    checkpoint = max(int(c.removeprefix('checkpoint-')) for c in checkpoints)
    return f'{output_dir}/checkpoint-{checkpoint}'


def get_best_checkpoint(checkpoint):
    with open(f'{checkpoint}/trainer_state.json') as src:
        return json.load(src).get('best_model_checkpoint') or checkpoint


class SaveToS3Callback(TrainerCallback):
    def __init__(self, s3, args):
        self.s3 = s3
        self.output_dir = args.output_dir
        self.logging_dir = args.logging_dir

    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if 'out' in self.output_dir and args.local_rank == 0:
            print('on_save:', self.output_dir)

            last_checkpoint = get_last_checkpoint(self.output_dir)
            best_checkpoint = get_best_checkpoint(last_checkpoint)

            if last_checkpoint == best_checkpoint:
                print('best_checkpoint:', best_checkpoint)

                try:
                    self.s3.remove_dir(self.output_dir)
                except Exception as e:
                    print('remove_dir:', e)
                try:
                    self.s3.upload_dir(last_checkpoint)
                except Exception as e:
                    print('upload_dir:', e)

            shutil.rmtree(last_checkpoint, ignore_errors=True)

    def on_log(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if 'log' in self.logging_dir and args.local_rank == 0:
            print('on_log:', self.logging_dir)
            try:
                self.s3.remove_dir(self.logging_dir)
            except Exception as e:
                print('remove_dir:', e)
            try:
                self.s3.upload_dir(self.logging_dir)
            except Exception as e:
                print('upload_dir:', e)
