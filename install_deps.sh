#
# Use conda to install Snorkel dependencies
#
conda install beautifulsoup4 -y
conda install ipywidgets -y
conda install jupyter -y
conda install nltk -y
conda install requests -y
conda install pandas -y
conda install matplotlib -y
conda install "numpy>=1.11" -y
conda install "scipy>=0.18" -y
conda install "sqlalchemy>=1.0.14" -y
conda install "tensorflow>=1.0" -y
conda install six -y
conda install numba -y
conda install tika -y
conda install psycopg2 -y
conda install spacy -y
conda install future -y

# lxml has to be installed via pip 
pip install lxml==3.6.4
pip install git+https://github.com/HazyResearch/numbskull@master