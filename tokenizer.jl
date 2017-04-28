module Tokenizer
using JSON

#default_caption_files = ["data/Flickr30k/results_20130124.token", "data/MsCoco/captions_val2014.json", "data/MsCoco/captions_train2014.json"]
default_caption_files = ["data/Flickr30k/results_20130124.token"];
function tokenize(vocab, caption_dicts; data_files=default_caption_files)
  word_counts = Dict{String,Int}();
  for i=1:length(data_files)
      file = open(data_files[i])
      tokenization_type = split(data_files[i],'.')[2]
      if tokenization_type == "token"
        dict = tokenize_flicker_captions(readlines(file));
        push!(caption_dicts, dict)
        vocab = get_vocab(dict, vocab, word_counts)
      elseif tokenization_type == "json"
        dict = tokenize_mscoco_captions(readlines(file));
        push!(caption_dicts, dict)
        vocab = get_vocab(dict, vocab, word_counts)
      else
        println("invalid caption file: ", data_files[i])
      end
      close(file)
  end

  filtervocab(5,vocab,word_counts)
  return vocab, caption_dicts ;
end

function tokenize_flicker_captions(captions)
  captions_dict = Array{Tuple{Tuple{Int64,Array{String,1}},Int64},1}()
  for i=1:length(captions)
      words = map(lowercase, split(captions[i],[' ','\t','#','.', '\n']))
      id = words[1]
      id = parse(Int64,id)
      count = 4;
      for i=4:length(words)
        words[count] = lowercase(strip(words[count],[' ', '.', ',','#', '\'', ')', '(', '!', '/', '?', '\t', '`']))
        if length(words[count]) < 1
          deleteat!(words,count)
          count-=1;
        end
        count += 1;
      end
      push!(captions_dict, ((id,words[4:end]),count-4));
  end
  captions_dict = sort(captions_dict, by = tuple -> last(tuple), rev=false)
  return captions_dict
end

function tokenize_mscoco_captions(captions)
    captions_dict = Array{Tuple{Tuple{Int64,Array{String,1}},Int64},1}()
    data = JSON.parse(captions[1])["annotations"]
    for obj in data
      words = split(obj["caption"],' ');
      count = 1;
      for i=1:length(words)
        words[count] = lowercase(strip(words[count],[' ', '.', ',', '#', '\'', ')', '(', '!', '/', '?', '\t', '`']))
        if length(words[count]) < 1
          deleteat!(words,count)
          count-=1;
        end
        count += 1;
      end
      id = obj["image_id"]
      push!(captions_dict, ((id,words),count-1));
    end
    captions_dict = sort(captions_dict, by = tuple -> last(tuple), rev=false)
    return captions_dict
end

function get_vocab(dict, vocab, word_counts)

  for data in dict
    for word in data[1][2]
          if length(word) > 0
            get!(vocab, word, 1+length(vocab));
            count = get!(word_counts,word,0)
            word_counts[word] = count+1;
          end
    end
  end
  return vocab

end

function filtervocab(threshold,vocab,word_counts)
  for (word,count) in word_counts
    if count < threshold
      delete!(vocab, word)
    end
  end

  temp = copy(vocab)
  empty!(vocab)

  get!(vocab,"~~",1+length(vocab)) #eos
  get!(vocab,"``",1+length(vocab)) #bos
  get!(vocab,"##",1+length(vocab))  #unk
  
  for (key,value) in temp
    get!(vocab, key, 1+length(vocab));
  end

  temp = 0
end

#vocab = get_vocab(dict1, vocab)


#println(length(vocab))
#vocab = get_vocab(dict2, vocab)
#println(length(vocab))
#vocab = get_vocab(dict3, vocab)
#println(length(vocab))
end
