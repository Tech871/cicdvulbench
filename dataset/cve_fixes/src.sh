#!/bin/bash

mkdir ../../src
cd ../../src || exit

wget https://zenodo.org/records/13118970/files/CVEfixes_v1.0.8.zip?download=1

mv CVEfixes_v1.0.8.zip\?download=1 CVEfixes_v1.0.8.zip

unzip CVEfixes_v1.0.8.zip

gzcat CVEfixes_v1.0.8/Data/CVEfixes_v1.0.8.sql.gz | sqlite3 cve_fixes.db

rm -fr CVEfixes_v1.0.8 CVEfixes_v1.0.8.zip wget-log
