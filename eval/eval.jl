#MSCOCO 2014
using JSON
#file = open("ids_coco_bm4");
file = open("candidate_ids.txt");
candidate_ids = readlines(file);
candidate_ids = map(p->parse(Int,p), candidate_ids)

file = open("../data/MsCoCo/captions_val2014.json");
captions = JSON.parse(file);
annotations = captions["annotations"];

caps = Dict();
for item in annotations
         arr = get!(caps, item["image_id"], [])
         if length(arr) == 5
           continue;
         end
         cap = strip(item["caption"]);
         cap = strip(cap,['.']);
         cap = cap * " ."
         push!(arr, lowercase(cap))
end

refs = Any[]
for i=0:4
  push!(refs, open("./coco_refs/ref$(i)","w"))
end

for id in candidate_ids
           arr = get(caps,id,nothing)
           for i=1:length(refs)
             println(refs[i],strip(arr[i]))
           end
end

map(p->close(p),refs);
println("MSCOCO Scores")
run(pipeline(`perl multi-bleu.perl ./coco_refs/ref`, stdin="candidates.txt"))
#Flickr30k
file = open("candidate_ids_flickr");
candidate_ids = readlines(file);
candidate_ids = map(p->parse(Int,p), candidate_ids)

file = open("../data/Flickr30k/results_20130124.token");
captions = readlines(file);

caps = Dict();
for line in captions
          info = split(line,"#")
          id = info[1]
          id = split(id,".")[1]
          id = parse(Int,id)
          cap = info[2]
          cap = split(cap,"\t")[2]
          cap = lowercase(strip(cap))
          arr = get!(caps,id, [])
          push!(arr,cap)
end

refs = Any[]
for i=0:4
  push!(refs, open("./flickr_refs/f_ref$(i)","w"))
end

for id in candidate_ids
           arr = get(caps,id,nothing)
           if arr==nothing
             println("id is missing in reference: ", id)
             break;
           end
           for i=1:length(refs)
             println(refs[i],strip(arr[i]))
           end
end
map(p->close(p),refs);

println("Flickr30k Scores")
run(pipeline(`perl multi-bleu.perl ./flickr_refs/f_ref`, stdin="candidates_flickr"))

#run(`perl multi-bleu.perl f_ref < candidates_f.txt`)
#run(`perl multi-bleu.perl ref < candidates.txt`)
