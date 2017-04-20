using JLD
using MAT
using JSON
using Knet

include("tokenizer.jl")

global vocab = Dict{String, Int}()
caption_dicts = Array{Array{Tuple{Tuple{Int64,Array{String,1}},Int64},1},1}()
Tokenizer.tokenize(vocab, caption_dicts)


dataset = open("./data/Flickr30k/karpathy/dataset.json");
data = JSON.parse(dataset);
images = data["images"];
file = matopen("./data/Flickr30k/karpathy/vgg_feats.mat");
features = read(file,"feats");

feats = Dict{Int,Array{Float32}}()

# for image in images
#     feature = features[:,image["imgid"]+1];
#     filename = split(image["filename"],'.')[1];
#     save("./data/Flickr30k/karpathy/features/$(filename).jld", "feature", feature);
# end

for image in images
     feature = features[:,image["imgid"]+1];
     filename = split(image["filename"],'.')[1];
     get!(feats,parse(Int,filename),feature);
end

println(length(feats))
count = 1;
for (el,v) in caption_dicts[1]
     id = el[1];
     filename = "./data/Flickr30k/karpathy/features/$(id).jld";
     if isfile(filename)
         exist = get(feats,id,nothing);
         if exist == nothing
           println(id)
           get!(feats,id,load(filename,"feature"))
         end
    end
    count += 1;
end

println(length(feats))

save("./data/Flickr30k/karpathy/features/feats.jld", "features", feats);
