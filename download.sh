#!/usr/bin/env bash

#
# Download embeddings
#
echo "Downloading pre-trained embedding data..."
url=http://i.stanford.edu/hazy/share/embs/blog.embs.tar.gz
data_tar="blog.embs"

if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi

echo "Unpacking pre-trained embedding data..."
tar -zxvf $data_tar.tar.gz -C data
bunzip2

echo "Deleting tar file..."
rm $data_tar.tar.gz

echo "Done!"
