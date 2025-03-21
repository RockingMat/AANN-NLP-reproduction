
URL=https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip
DIR=data/babylm

mkdir -p $DIR
wget $URL -O $DIR/babylm_data.zip
unzip $DIR/babylm_data.zip -d $DIR
rm $DIR/babylm_data.zip

mv $DIR/babylm_data/* $DIR
rm -r $DIR/babylm_data

# concatenate all files within each dir babylm_100M, babylm_dev, babylm_test

cat $DIR/babylm_100M/* > $DIR/train_100M.txt
cat $DIR/babylm_dev/* > $DIR/dev.txt
cat $DIR/babylm_test/* > $DIR/test.txt

# sentence-tokenize all files using src/sentence_tokenize.py
python ./sentence_tokenize.py --source $DIR/train_100M.txt --target $DIR/train.sents&
python ./sentence_tokenize.py --source $DIR/dev.txt --target $DIR/dev.sents&
python ./sentence_tokenize.py --source $DIR/test.txt --target $DIR/test.sents
wait

# remove the original dirs
rm -r $DIR/babylm_100M
rm -r $DIR/babylm_10M
rm -r $DIR/babylm_dev
rm -r $DIR/babylm_test

rm $DIR/train_100M.txt
rm $DIR/dev.txt
rm $DIR/test.txt

# Uncomment this if you want to also run postags

# Warning: this requires a different spacy environment that is compatible with and old version of transformers

# python src/postag.py \
#     --source $DIR/train.sents \
#     --target $DIR/train.postags