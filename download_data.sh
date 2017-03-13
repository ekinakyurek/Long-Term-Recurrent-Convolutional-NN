echo "Downloading Flickr30k images and tokenized captions"
mkdir Flickr30k
cd  Flickr30k
wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar
tar -xf flickr30k-images.tar
rm flickr30k-images.tar
wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz
tar -xzf flickr30k.tar.gz
rm flickr30k.tar.gz
echo "Succesfully downloaded in to Flickr30k folder"
cd ../
mkdir MsCoCo
cd MsCoCo
cd ../
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip -qq train2014.zip -d MsCoCo/
rm train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip -qq val2014.zip -d MsCoCo/
rm val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
unzip -qq captions_train-val2014.zip -d MsCoCo/
rm captions_train-val2014.zip
