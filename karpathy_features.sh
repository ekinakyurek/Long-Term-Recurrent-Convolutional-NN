echo "Downloading Flickr30k features"
wget http://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip
mkdir ./Flickr30k/karpathy/
unzip -qq flickr30k.zip -d ./Flickr30k/karpathy/
mv ./Flickr30k/karpathy/flickr30k/* ./Flickr30k/karpathy/
rm -rf ./Flickr30k/karpathy/flickr30k/
mkdir ./Flickr30k/karpathy/features/
echo "Succesfully downloaded in to Flickr30k folder"
