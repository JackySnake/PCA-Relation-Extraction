#!/usr/bin/env bash

# hack for installing snorkel
unzip deps.zip

# set pathing
source set_env.sh

# unzip our database files
cd data/db/
bunzip2 cdr.db.bz2
bunzip2 spouse.db.bz2