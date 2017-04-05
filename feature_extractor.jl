using JLD
using MAT
using JSON
using Knet

dataset = open("./data/Flickr30k/karpathy/dataset.json");
data = JSON.parse(dataset);
images = data["images"];
file = matopen("./data/Flickr30k/karpathy/vgg_feats.mat");
features = read(file,"feats");

for image in images
    feature = features[:,image["imgid"]+1];
    filename = split(image["filename"],'.')[1];
    save("./data/Flickr30k/karpathy/features/$(filename).jld", "feature", feature);
end
