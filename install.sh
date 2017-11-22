#!/usr/bin/env bash

# hack for installing snorkel
tar -xvf deps.tar.gz

# unzip our database files
cd data/db/
bunzip2 cdr.db.bz2
bunzip2 spouse.db.bz2