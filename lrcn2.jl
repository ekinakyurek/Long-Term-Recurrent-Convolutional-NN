for p in ("Knet","ArgParse","ImageMagick","MAT","Images", "Compat", "QuartzImageIO")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet
!isdefined(:VGG) && include(Knet.dir("examples","vgg.jl"))
!isdefined(:Tokenizer) && include("./tokenizer.jl")
"""
julia lrcn.jl image-file-or-url

This example implements the Long-term recurrent convolutional network model from

Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

* Paper url: https://arxiv.org/pdf/1411.4389.pdf
* Project page: https://github.com/ekinakyurek/lrcn
"""
module LRCN
using Knet, AutoGrad, ArgParse, Compat, MAT, Images;
using Tokenizer;

const vggurl = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"
const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const Flickr30k_captions = "data/Flickr30k/results_20130124.token"
const MsCoCo_captions = "data/MsCoCo/captions_train2014.json"
const MsCoCo_validation = "data/MsCoCo/captions_val2014.json"
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "LRCN.jl (c) Ekin Akyürek, 2017. Long-term Recurrent Convolutional Networks for Visual Recognition and Description"
    @add_arg_table s begin
        ("image"; default=imgurl; help="Image file or URL.")
        ("--model"; default=Knet.dir("data","imagenet-vgg-verydeep-16.mat"); help="Location of the model file")
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--bestfile"; help="Save best model to file")
        ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of characters.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[1000]; help="Sizes of one or more LSTM layers.")
        ("--embed"; arg_type=Int; default=1000; help="Size of the embedding vector.")
        ("--cnnout"; arg_type=Int; default=4096; help="Size of the cnn visual output vector.");
        ("--epochs"; arg_type=Int; default=5; help="Number of epochs for training.")
        ("--recall"; arg_type=Int; default=10; help="Number of tries for caption generation.")
        ("--batchsize"; arg_type=Int; default=80; help="Number of senteces to train on in parallel.")
        ("--lr"; arg_type=Float64; default=0.1; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=5.0; help="Value to clip the gradient norm at.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--vggon"; action=:store_true; help="load vgg weights")
        ("--feature";action=:store_true; help="extract features")
        ("--beam_width";arg_type=Int;default=2;help="width of beam search")
        #TODO ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
    end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  println("opts=",[(k,v) for (k,v) in o]...)
  o[:seed] > 0 && srand(o[:seed])
  o[:atype] = eval(parse(o[:atype]))

  if any(f->(o[f]!=nothing), (:loadfile, :savefile, :bestfile))
      Pkg.installed("JLD")==nothing && Pkg.add("JLD") # error("Please Pkg.add(\"JLD\") to load or save files.")
      eval(Expr(:using,:JLD))
  end

  #isempty(o[:datafiles]) && push!(o[:datafiles],Flickr30k_captions) # Flickr30k is default
  isempty(o[:datafiles]) && push!(o[:datafiles],MsCoCo_captions, MsCoCo_validation)
  gpu()>=0 && KnetArray(Float32,2,2) * KnetArray(Float32,2,2); #To initialize cudablas

  println("Tokenization starts")
  #vocab = {"words1"=>1,"words2"=>2, ....}
  #captions_dicts = [((id1, ["word1","word2",...]),length1),((id2, ["word1","word2",...]),length2),...]
  global vocab = Dict{String, Int}();
  caption_dicts = Array{Array{Tuple{Tuple{Int64,Array{String,1}},Int64},1},1}()
  Tokenizer.tokenize(vocab, caption_dicts; data_files=o[:datafiles])
  println("Tokenization finished")

  global vocab_size = length(vocab);
  global batchsize = o[:batchsize];

  println("Intializing LSTM weigths")
  if o[:loadfile]==nothing
   model = initweights(o[:atype], o[:hidden], vocab_size, o[:embed], o[:cnnout])
  else
   info("Loading model from $(o[:loadfile])")
   model = map(p->convert(o[:atype],p), load(o[:loadfile], "model"))
   global vocab = load(o[:loadfile], "vocab")
   global vocab_size = length(vocab);
  end
  optim = initparams(model);
  println("LSTM is initialized")
  println("$vocab_size unique words")

  global convnet = nothing; global averageImage = nothing;
  if o[:vggon]
    println("Intializing cnn_weights")
    info("Reading $(o[:model])");
    if !isfile(o[:model])
     println("Should I download the VGG model (492MB)? Enter 'y' to download, anything else to quit.")
     readline() == "y\n" || return
     download(vggurl,o[:model])
    end
    vgg = matread(o[:model]);
    params = get_params_cnn(vgg)
    global convnet = get_convnet(params...);
    global averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
    info("Cnn is initialized")
  end

  println("Loading existing features to train")
  #global feats = load("./data/Flickr30k/featsn.jld", "features")
  global feats = load("./data/MsCoCo/train2014/train_featsn.jld","features");
  global featsvl = load("./data/MsCoCo/val2014/val_featsn.jld", "features");
  println("Features loaded")

  if o[:generate] > 0
    if o[:vggon]
      image = read_image_data(o[:image], averageImage)
      for i=1:o[:recall]
       generate(model,initstate(model,1), image, vocab, o[:generate], o[:beam_width])
      end
    else
      #DBG: Add random selection from dataset
      for i=1:o[:recall]
       id =caption_dicts[1][rand(1:end)][1][1];
       generate(model,initstate(model, 1), id, vocab,o[:generate], o[:beam_width]);
      end
    end
  end

  if !isempty(caption_dicts)
    println("Batching starts")
    if o[:feature]
      println("extracting image features for all dataset for once")
      #extract_features(caption_dicts[1],  "./data/Flickr30k/","feats2", "");
      #extract_features(caption_dicts[1], "./data/MsCoCo/train2014/","train_feats", "COCO_train2014_");
      #extract_features(caption_dicts[1], "./data/MsCoCo/val2014/","val_feats","COCO_val2014_");
      return;
      println("image features extracted")
    end
    sequence = map(t->minibatch(t, vocab, o[:batchsize]), caption_dicts)
    caption_dicts = 0; gc();
    println("Batching finished")
    println("Training starts:....")
    train!(model, optim, sequence, vocab, o)
  end

  if o[:savefile] != nothing
      info("Saving last model to $(o[:savefile])")
      save(o[:savefile], "model", model, "vocab", vocab)
  end

end

function extract_features(seq, dataset,filename, header)
  feats = Dict{Int,Array{Float32,2}}();
  #feats =   load("$(dataset)$(filename).jld", "features", feats); gc();
  ids = Dict{Int,Int}();
  for t=1:length(seq)
      get!(ids,seq[t][1][1],length(ids)+1)
  end
  ids = collect(keys(ids))
  l = length(ids);
  println(l);
  prefix = dataset*header
  k=1;
  @inbounds for id in ids
      image = read_image_data(prefix*dec(id,12)*".jpg", averageImage)
      image = convnet(image);
      feats[id] = convert(Array{Float32}, image)

      if k%1000 == 1
            println(length(feats));
      end
      if k%10000 == 1
            save(dataset*filename*".jld", "features", feats);
            println(length(feats), ". image saved");
     end
     k+=1;
  end
  println("It is finished. I do last saving")
  save(dataset*filename*".jld", "features", feats);
end

function train!(model, optim, sequence, vocab, o)
    s0 = initstate(model, o[:batchsize])
    if o[:fast]
         for epoch=1:o[:epochs]
               train1(model, optim, s0, sequence[1]; batch_size=o[:batchsize], lr=o[:lr], gclip=o[:gclip], cnnout=o[:cnnout], pdrop=0.0)
               if o[:savefile] != nothing
                   info("Saving last model to $(o[:savefile])")
                   save(o[:savefile], "model", model, "vocab", vocab)
               end
               losses = zeros(Float32, 2);
               losses[1] = average_loss(model, sequence[1],feats; batch_size=o[:batchsize], cnnout=o[:cnnout], pdrop=0.0)
               losses[2] = average_loss(model, sequence[2],featsvl; batch_size=o[:batchsize], cnnout=o[:cnnout], pdrop=0.0)
               #losses = map(d->average_loss(model, d ; batch_size=o[:batchsize], cnnout=o[:cnnout], pdrop=0.0), sequence)
               println((:epoch,epoch,:loss,losses...))
               datasheet = open("e512_h1000_p_0.0.out","a+");
               println(datasheet,(:epoch,epoch,:loss,losses...));
               close(datasheet);
          end
           gpu()>=0 && Knet.cudaDeviceSynchronize()
        return
    end
end

function initeosbos(batch_size;atype=KnetArray{Float32})
  global eos = zeros(Int,batch_size)
  eos[:] = 1
  global bos = zeros(Int,batch_size)
  bos[:] = 2
  global unk = zeros(Int,batch_size)
  unk[:] = 3
end

function minibatch(caption_dict, word_to_index, batch_size)
    println("Minibatching starts")
    #to minibatch the train data
    if length(caption_dict) > 30000
      println("Deleting unbatchable captions: $(length(caption_dict))")
      delete_unbatchable_captions!(caption_dict, batch_size)
      println("Unbatchable captions is deleted: $(length(caption_dict))")
    else
      batch_size = 10
      println("Deleting unbatchable captions: $(length(caption_dict))")
      delete_unbatchable_captions!(caption_dict, batch_size)
      println("Unbatchable captions is deleted: $(length(caption_dict))")
    end

    initeosbos(batch_size)
    #length for one caption  caption[i][2]
    lengths = map(t->t[2], caption_dict)
    #total word count
    nbatch = div(sum(lengths), batch_size)

    #initialize output sequence, input sequence, and range of the LSTM network for each batch
    sequence = [zeros(Int,batch_size) for i=1:nbatch];
    input_ids = [zeros(Int,batch_size) for i=1:batch_size:length(lengths)]
    index = 1; l = 1; input_index = 1;

    for i=1:batch_size:length(lengths)
        l = lengths[i]
        for j=i:i+batch_size-1
          (id,words), _ = caption_dict[j]
          input_ids[input_index][j-i+1] = id
          for k=index:index+l-1
             sequence[k][j-i+1] = get(word_to_index,words[k-index+1],unk[1])
          end
        end
        index = index+l;
        input_index += 1;
    end

    println("Minibatching finished")
    return sequence,input_ids,lengths
end

function delete_unbatchable_captions!(caption_dict,batch_size)
    lengths = map(t->t[2], caption_dict)
    limit = length(lengths)-batch_size+1;
    max_length = maximum(lengths)
    current_length = lengths[1]; current_index = 1;
    ranges = Int64[];
    while current_index < limit
        if lengths[current_index+batch_size-1]==current_length
            current_index += batch_size;
        else
            old_index = current_index;
            current_index = 0;
            while current_index==0
                current_length+=1;
                if current_length > max_length
                  break
                end
                current_index = findfirst(lengths,current_length)
            end
            append!(ranges,collect(old_index:current_index-1))
        end
        if current_index >= limit
            append!(ranges,collect(current_index:length(lengths)))
            break;
        end
    end
    deleteat!(caption_dict,ranges);
    return caption_dict;
end


function train1(param, optim, state, seq; batch_size=20, lr=2.0, gclip=0.0, cnnout=4096, pdrop=0.0, atype=KnetArray{Float32})
  Knet.gc(); gc();
  sequence = seq[1];input_ids = seq[2];lengths = seq[3]

  initeosbos(batchsize);

  start_indices = ones(Int,length(sequence));
  index = 1; count = 1;
  for t = 1:batch_size:length(lengths)
        l=lengths[t];
        start_indices[count] = index
        index = index+l; count+=1;
  end
  total_sequence_size = sum(lengths);

  # index_to_char = Array(String, length(vocab))
  # for (k,v) in vocab; index_to_char[v] = k; end

  index = 1; l=1; input_index = 1; trained = 0;
  input = KnetArray(Float32, batch_size, cnnout);

 for t in shuffle(1:batch_size:length(lengths))
    l = lengths[t]
    if l>28
       continue;
    end
    input_index = Int((t-1)/batch_size + 1);
    index = start_indices[input_index];

    #   print(input_ids[input_index][1],":")
    #   print(index_to_char[bos[1]],' ')
    #   for i=index:index+l-1
    #      expected = sequence[i][1]
    #      print(index_to_char[expected],' ');
    #   end
    #  print(index_to_char[eos[1]],' ')
    #  println();

    #println("length of the senteces in the batch: ", l);
    for i=1:batch_size
           id = input_ids[input_index][i]
           feature_index = get(feats,id,nothing)
           input[i,:] = convert(atype,feature_index);
    end
    #input = convert(KnetArray{Float32},input);
    gloss = lossgradient(param, copy(state), input, sequence, index:index+l-1; pdrop = pdrop)

     trained = trained + l
     if t%2000 == 1
       println((trained*batch_size)/total_sequence_size);
       println("loss in sentence: ", loss(param, copy(state), input, sequence, index:index+l-1; pdrop = pdrop))
       Knet.knetgc();gc();
     end
    #  if gclip > 0
    #     gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
    #      if gnorm > gclip
    #        for k=1:length(gloss)
    #          gloss[k] = (gloss[k] * gclip) / gnorm
    #        end
    #      end
    #  end
     update!(param, gloss, optim);
     isa(state,Vector{Any}) || error("State should not be Boxed.")
    end
end

function initparams(model)
  prms = Array(Any,length(model))
  for k=1:length(prms)
    prms[k] = Adam()
  end
  return prms
end

function average_loss(param, seq, featsms; batch_size=20, cnnout=4096, atype=KnetArray{Float32}, pdrop=0.0)

  sequence = seq[1]
  input_ids = seq[2]
  lengths = seq[3]

  batch_size = size(sequence[1],1)
  state = initstate(param, batch_size)
  initeosbos(batch_size)

  start_indices = ones(Int,length(sequence));
  index = 1; i = 1;
  for t = 1:batch_size:length(lengths)
        l=lengths[t];
        start_indices[i] = index
        index = index+l; i+=1;
  end

  total_sequence_size = sum(lengths);

  #index_to_char = Array(String, length(vocab))
  #for (k,v) in vocab; index_to_char[v] = k; end

  index = 1; l=1; input_index = 1; calculated = 0;
  total = 0.0; count = 0

  #for t = 1:batch_size:length(lengths)
  input = KnetArray(Float32, batch_size, cnnout);
  Knet.gc(); gc();
  for t in shuffle(1:batch_size:length(lengths))

     l = lengths[t]
     if l>28
       continue;
     end

     input_index = Int((t-1)/batch_size + 1);
     index = start_indices[input_index];

     #println("length of the senteces in the batch: ", l);
     for i=1:batch_size
           id = input_ids[input_index][i]
           feature_index = get(featsms,id,nothing)
           input[i,:] = convert(atype,feature_index);
     end

    cnn_input = input*param[end-3];
    lstm_input = param[end-2][bos,:];

    new_state = copy(state);
    for t in  index:index+l-1
        ypred = lrcn(param,new_state,cnn_input,lstm_input;pdrop=pdrop)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        index = similar(sequence[t])
        @inbounds for i=1:batch_size
            index[i] = i + (sequence[t][i]-1)*batch_size;
        end
        total += sum(ynorm[index])
        count += batch_size;
        lstm_input = param[end-2][sequence[t],:];
    end

    ypred = lrcn(param,new_state,cnn_input,lstm_input; pdrop=pdrop)
    ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
    index = similar(eos)
    @inbounds for i=1:batch_size
        index[i] = i + (eos[i]-1)*batch_size
    end
    total += sum(ynorm[index])
    count += batch_size;

    calculated = calculated + l;
    if t%100 == 1
      println(100*(calculated*batch_size/total_sequence_size), " % of training completed: ");
      println("current loss: ", -total/count);
      Knet.knetgc();gc();
    end
  end
  println(-total / count)
  return -total / count
end


function initweights(atype, hidden, vocab, embed, cnnout)
  init(d...)=atype(xavier(d...))
  bias(d...)=atype(zeros(d...))
  model = Array(Any, 2*length(hidden)+4)
  X = embed
  for k = 1:length(hidden)
    H = hidden[k]
    model[2k-1] = init(X+H, 4H)
    model[2k] = bias(1, 4H)
    model[2k][1:H] = 1 # forget gate bias = 1
    X = H
  end
  model[end-3] = init(cnnout,Int(embed/2))
  model[end-2] = init(vocab,Int(embed/2))
  model[end-1] = init(hidden[end],vocab)
  model[end] = bias(1,vocab)
  return model;
end

let blank = nothing; global initstate
function initstate(model, batch)
    nlayers = div(length(model)-3,2)
    state = Array(Any, 2*nlayers)
    for k = 1:nlayers
        bias = model[2k]
        hidden = div(length(bias),4)
        if typeof(blank)!=typeof(bias) || size(blank)!=(batch,hidden)
            blank = fill!(similar(bias, batch, hidden),0)
        end
        state[2k-1] = state[2k] = blank
    end
    return state
end
end

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function lrcn(w, s, x_cnn, x_lstm; pdrop=0.0)
  x = hcat(x_cnn,x_lstm);
  for i = 1:2:length(s)
      x = dropout(x,pdrop);
      (s[i],s[i+1]) = lstm(w[i],w[i+1],s[i],s[i+1],x)
      x = s[i]
  end
  return x * w[end-1] .+ w[end]
end

function loss(param,state,input,sequence,range; pdrop=0.0)
    total = 0.0; count = 0;
    #Beginning of sentence is multiplied by embedding matrix
    lstm_input = param[end-2][bos,:]
    #CNN output for the batch of sentences is multiplied with one parameter
    input = input * param[end-3];
    #Training starts for all words
   for t in range
        ypred = lrcn(param,state,input,lstm_input;pdrop=pdrop)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        index = similar(sequence[t])
        @inbounds for i=1:batchsize
             index[i] = i + (sequence[t][i]-1)*batchsize
        end
        total += sum(ynorm[index])
        count += batchsize;
        lstm_input = param[end-2][sequence[t],:]
    end
    #We expect from model to predict end of sentence
    ypred = lrcn(param,state,input,lstm_input; pdrop=pdrop)
    ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
    index = similar(eos)
    @inbounds for i=1:batchsize
        index[i] = i + (eos[i]-1)*batchsize
    end
    total += sum(ynorm[index])
    count += batchsize
    return -total / count
end

lossgradient = grad(loss);

function generate(param, state, input, vocab, nword, beam_width)
    println("Generating Starts:");
    #Create index to word array
    index_to_char = Array(String, length(vocab))
    for (k,v) in vocab; index_to_char[v] = k; end

    #Initialize eos, bos and unk tokens for batchsize 1.
    initeosbos(1)

    if typeof(input) != Int
      # Get CNN features with convnet
      input = convnet(input);
      # Normalizing features
      input = input/sum(input);
    else
      #Loading from existing features
      println(input,":")
      input = get(feats,input,nothing);
      if input == nothing
            return;
      end
      input = convert(KnetArray,reshape(input,1,4096));
    end

    #Beginning of sentence is multiplied by embedding matrix
    lstm_input = param[end-2][bos,:];
    #CNN output for the batch of sentences is multiplied with one parameter
    input = input * param[end-3];
    #
    xs = Array{Tuple{Array{Int,1},Float32},1}();
    states = Array{typeof(state),1}();
    for i=1:beam_width
        push!(xs,(bos,1.0))
        push!(states,copy(state))
    end
    xp = beam_search(xs,states, input, param, nword,1)
    for i=1:length(xp[1][1])
        print(index_to_char[xp[1][1][i]], " ");
    end
    println()
    println("Generating Done")
end

function beam_search(x,states,input,param,nword,current)
  new_x = Array{Tuple{Array{Int,1},Float32},1}();

  for i=1:length(x)
       current_index = x[i][1][end]
       current_probability = x[i][2]
       lstm_input = param[end-2][current_index:current_index,:];
       ypred = lrcn(param,states[i],input,lstm_input)
       ynorm = exp(logp(ypred,2));
       ynorm = convert(Array{Float32},ynorm);
       ynorm = reshape(ynorm,vocab_size);
       xmaxes = sortperm(ynorm, rev=true);
       xmaxes = xmaxes[1:length(x)]
       pmaxes = ynorm[xmaxes]*current_probability
       for j=1:length(x)
         xm = [x[i][1];xmaxes[j]]
         push!(new_x,(xm,pmaxes[j]));
       end
       if current==1
         break;
       end
  end

  sorted = sortperm(new_x, by = tuple -> last(tuple), rev=true)
  xs = new_x[sorted[1:length(x)]];
  #println(xs)
  if current>nword ##eos
    return xs;
  end
  new_states = similar(states);
  for i=1:length(xs)
      new_states[i] = copy(states[ceil(Int,sorted[i]/length(x))]);
  end
  beam_search(xs, new_states, input, param, nword, current+1)
end

function sample(p)
    p = convert(Array,p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

#CNN Parameters from VLFEAT VGG.NET
function get_params_cnn(CNN; last_layer="fc7")
    layers = CNN["layers"]
    weights, operations, derivatives = [], [], []

    for l in layers
        get_layer_type(x) = startswith(l["name"], x)
        operation = filter(x -> get_layer_type(x), LAYER_TYPES)[1]
        push!(operations, operation)
        push!(derivatives, haskey(l, "weights") && length(l["weights"]) != 0)

        if derivatives[end]
            w = l["weights"]
            if operation == "conv"
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif operation == "fc"
                w[1] = transpose(mat(w[1]))
            end
            push!(weights, w)
        end

        last_layer != nothing && get_layer_type(last_layer) && break
    end

    map(w -> map(KnetArray, w), weights), operations, derivatives
end

# convolutional network operations
convx(x,w)=conv4(w[1], x; padding=1, mode=1) .+ w[2]
relux = relu
poolx = pool
probx(x) = x
fcx(x,w) = w[1]*mat(x) .+ w[2];
tofunc(op) = eval(parse(string(op, "x")))
forw(x,op) = tofunc(op)(x)
forw(x,op,w) = tofunc(op)(x,w)

function get_convnet(weights, operations, derivatives)
    function convnet(xs)
        i, j = 1, 1
        num_weights, num_operations = length(weights), length(operations)
        while i <= num_operations && j <= num_weights
            if derivatives[i]
                xs = forw(xs, operations[i], weights[j])
                j += 1
            else
                xs = forw(xs, operations[i])
            end
            i += 1
        end
        return transpose(xs);
    end
end

function read_image_data(img, averageImage;atype=KnetArray{Float32})
     if contains(img,"://")
         info("Downloading $img")
         img = download(img)
     end
     a0 = load(img)
     new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
     a1 = Images.imresize(a0, new_size)
     i1 = div(size(a1,1)-224,2)
     j1 = div(size(a1,2)-224,2)
     b1 = a1[i1+1:i1+224,j1+1:j1+224]
     cdim = length(b1[1])
     if cdim != 3
      c1 = cat(3,channelview(b1),channelview(b1),channelview(b1))
      c1 = permutedims(c1,[2,1,3])
     else
        c1 = permutedims(channelview(b1), (3,2,1))
     end
     d1 = convert(Array{Float32}, c1)
     e1 = reshape(d1[:,:,1:3], (224,224,3,1))
     f1 = (255 * e1 .- averageImage)
     g1 = permutedims(f1, [2,1,3,4])
     x1 = KnetArray(g1)
end

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
    import JLD: writeas, readas
    type KnetJLD; a::Array; end
    writeas(c::KnetArray) = KnetJLD(Array(c))
    readas(d::KnetJLD) = KnetArray(d.a)
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end
