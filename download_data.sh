echo "Downloading Flickr30k images and tokenized captions"
mkdir Flickr30k
cd  Flickr30k
wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar
tar -xzvf flickr30k-images.tar
rm flickr30k-images.tar
wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz
tar -xzvf flickr30k.tar.gz
rm flickr30k.tar.gz
echo "Succesfully downloaded in to Flickr30k folder"
cd ../
mkdir MsCoCo
cd MsCoCo
mkdir train_images
cd ../
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip -d MsCoCo/train_images


