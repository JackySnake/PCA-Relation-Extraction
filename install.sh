#!/usr/bin/env bash

# manually install Snorkel and dependencies
git clone https://github.com/HazyResearch/snorkel.git
mv snorkel snorkel-core
cd snorkel-core

# Make sure the submodules are installed
git submodule update --init --recursive
pip install --requirement python-package-requirement.txt

cd ../
ln -s snorkel-core/snorkel .
ln -s snorkel-core/treedlib .
ln -s snorkel-core/tree_structs.py .

source set_env.sh

# unzip our database files
cd data/db/
bunzip2 cdr.db.bz2
bunzip2 spouse.db.bz2