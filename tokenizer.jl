module Tokenizer
using JSON
default_caption_files = ["data/Flickr30k/results_20130124.token", "data/MsCoco/captions_val2014.json", "data/MsCoco/captions_train2014.json"]
default_caption_files = ["data/Flickr30k/results_20130124.token"];
function tokenize(;data_files=default_caption_files)
  global vocab = Dict{String, Int}()
  global captions_dicts = Any[];
  for i=1:length(data_files)
      file = open(data_files[i])
      tokenization_type = split(data_files[i],'.')[2]
      if tokenization_type == "token"
        captions_dict = tokenize_flicker_captions(readlines(file));
        push!(captions_dicts, captions_dict)
        vocab = get_vocab(captions_dict, vocab)
      elseif tokenization_type == "json"
        captions_dict = tokenize_mscoco_captions(readlines(file));
        push!(captions_dicts, captions_dict)
        vocab = get_vocab(captions_dict, vocab)
      else
        println("invalid caption file: ", data_files[i])
      end
      close(file)
  end
  return vocab, captions_dicts ;
end

function tokenize_flicker_captions(captions)
  captions_dict = Dict{Int,Array{Array{AbstractString,1},1}}()
  for i=1:length(captions)
      words = map(lowercase, split(captions[i],' '))
      for i=2:length(words)-1
        words[i] = lowercase(strip(words[i],['\"', '.', ',', '#', '&', '\'', ')', '(', '!', ' ', '/', '?', '\t']))
      end
      id = split(split(words[1],'#')[1],'.')[1];
      id = parse(Int,id)
      a = get!(captions_dict, id , [])
      push!(a, words[2:end-1]);
  end
  return captions_dict
end

function tokenize_mscoco_captions(captions)
    captions_dict = Dict{Int,Array{Array{AbstractString,1},1}}()
    data = JSON.parse(captions[1])["annotations"]
    for obj in data
      words = split(obj["caption"],' ');
      for i=1:length(words)
        words[i] = lowercase(strip(words[i],['\"', '.', ',', '#', '&', '\'', ')', '(', '!', ' ', '/', '?', '\t']))
      end
      id = obj["image_id"]
      a = get!(captions_dict, id ,[])
      push!(a, words)
    end
    return captions_dict
end

function get_vocab(dict, vocab)
  for (k,v) in dict
    for word_arrays in v
      for word in word_arrays
          #word=strip(word,['\"', '.', ',', '#', '&', '\'', ')', '(', '!', ' ', '/', '?', '\t'])
          if length(word) > 0
            get!(vocab, word, 1+length(vocab));
          end
      end
    end
  end
  return vocab
end

#vocab = get_vocab(dict1, vocab)


#println(length(vocab))
#vocab = get_vocab(dict2, vocab)
#println(length(vocab))
#vocab = get_vocab(dict3, vocab)
#println(length(vocab))
end
