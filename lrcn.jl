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
const LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]
const Flickr30k_captions = "data/Flickr30k/results_20130124.token"
global averageImage = nothing;

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
        ("--epochs"; arg_type=Int; default=20; help="Number of epochs for training.")
        ("--recall"; arg_type=Int; default=5; help="Number of tries for caption generation.")
        ("--batchsize"; arg_type=Int; default=20; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=100; help="Number of steps to unroll the network for.")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=2.0; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=5.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.08; help="Initial weights set to winit*randn().")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        #TODO ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
        ("--top"; default=5; arg_type=Int; help="Display the top N classes")
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

    a = KnetArray(Float32,2,2) * KnetArray(Float32,2,2); #To initialize cudablas

    isempty(o[:datafiles]) && o[:loadfile]==nothing && push!(o[:datafiles],Flickr30k_captions) # Flickr30k

    if o[:loadfile]==nothing
      println("Tokenization starts")
      #vocab = {"words1"=>1,"words2"=>2, ....}
      global vocab = Dict{String, Int}()
      #captions_dicts = [((id1, ["word1","word2",...]),length1),((id2, ["word1","word2",...]),length2),...]
      caption_dicts = Array{Array{Tuple{Tuple{Int64,Array{String,1}},Int64},1},1}()
      Tokenizer.tokenize(vocab, caption_dicts; data_files=o[:datafiles])
      get!(vocab,"~",1+length(vocab)) #eos
      get!(vocab,"``",1+length(vocab)) #bos
      info("Tokenization finished with captions")

      println("Intializing cnn_weights")
      if !isfile(o[:model])
       println("Should I download the VGG model (492MB)? Enter 'y' to download, anything else to quit.")
       readline() == "y\n" || return
       download(vggurl,o[:model])
      end
      info("Reading $(o[:model])");
      vgg = matread(o[:model]);
      params = get_params_cnn(vgg)
      convnet = get_convnet(params...);
      global averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
      info("Cnn is initialized")


      println("Intializing LSTM weigths")
      model = initweights(o[:atype], o[:hidden], length(vocab), o[:embed], o[:cnnout], o[:winit])
      optim = initparams(model);
      info("LSTM is initialized")
    else
      info("Loading model from $(o[:loadfile])")
      vocab = load(o[:loadfile], "vocab")
      model = map(p->convert(o[:atype],p), load(o[:loadfile], "model"))
    end

    info("$(length(vocab)) unique words")

    if o[:generate] > 0
        println("Generation starts")
        state = initstate(model,1)
        img = read_image_data(o[:image], averageImage)
        #initialize eos and bos with batchsize 1;
        initeosbos(1;atype=o[:atype]);
        #Generate captions for given image
        generate(model, convnet, state, img, vocab, o[:generate])
        println("Generation finished")

    end


    if !isempty(caption_dicts)
      #init eos and bos with given batchsize
      initeosbos(o[:batchsize]);
      #get word sequences for caption datasets;
      sequence = map(t->minibatch(t, vocab, o[:batchsize]), caption_dicts)
      #no need to caption dicts any more delete it from memory. Call garbage collectors
      caption_dicts = 0; gc(); Knet.knetgc();
      println("Data is created")
      println("Training starts:....")
      train!(model, optim ,convnet, sequence, vocab, o)
      println("Generating after training......")
      for i=1:o[:recall]
        state = initstate(model, 1);
        #Flickr dataset image id for first bath first element.
        id = sequence[1][2][1][1]
        println(id,": ",i, ".","try ");
        image = read_image_data("./data/Flickr30k/flickr30k-images/$id.jpg", averageImage)
        #Initialize eos and bos with batchsize 1
        initeosbos(1);
        #Generate from id of image in Flickr30k database
        generate2(model,id,state,vocab,o[:generate]);
        #generate(model, convnet, state, image, vocab, o[:generate])
      end
      println("Generation finished")
    end

    if o[:savefile] != nothing
        info("Saving last model to $(o[:savefile])")
        save(o[:savefile], "model", model, "vocab", vocab)
    end

end


function initeosbos(batch_size;atype=KnetArray{Float32})
  global eos = falses(batch_size, length(vocab))
  eos[:,end-1] = 1;
  global eos = convert(atype,eos);
  global bos = falses(batch_size, length(vocab))
  bos[:,end] = 1;
  global bos = convert(atype,bos);
end


function train!(model, optim, convnet, sequence, vocab, o)
    s0 = initstate(model, o[:batchsize])
    lr = o[:lr]
    if o[:fast]
        @time (for epoch=1:o[:epochs]
               train1(model,optim, convnet, copy(s0), sequence[1]; batch_size=o[:batchsize], lr=lr, gclip=o[:gclip], cnnout=o[:cnnout])
               end; gpu()>=0 && Knet.cudaDeviceSynchronize())
        return
    end
    losses = map(d->report_loss(model, convnet, copy(s0), d ; batch_size=o[:batchsize], cnnout=o[:cnnout]), sequence)
    println((:epoch,0,:loss,losses...))
end

function delete_unbatchable_captions!(caption_dict,batch_size)
    lengths = map(t->t[2], caption_dict)

    limit = length(lengths)-batch_size+1;
    max_length =  maximum(lengths)

    current_length = lengths[1]

    current_index = 1;

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

        if current_index > limit
            append!(ranges,collect(current_index:length(lengths)))
            break;
        end

    end
    deleteat!(caption_dict,ranges);
    return caption_dict;
end


function minibatch(caption_dict, word_to_index, batch_size)
    println("Minibatching starts")

    #to minibatch the data
    println("Deleting unbatchable captions: $(length(caption_dict))")
    delete_unbatchable_captions!(caption_dict, batch_size)
    println("Unbatchable captions is deleted: $(length(caption_dict))")

    #length for one caption  1+caption[i][2] because of eos
    lengths = map(t->t[2], caption_dict)
    #total word count
    nbatch = div(sum(lengths), batch_size)

    #initialize output sequence, input sequence, and range of the LSTM network for each batch
    sequence = [falses(batch_size, length(word_to_index)) for i=1:nbatch];

    input_ids = [ones(Int,batch_size) for i=1:batch_size:length(lengths)]

    index = 1; l = 1; input_index = 1;

    # global eos = falses(batch_size, length(word_to_index))
    # eos[:,end-1] = 1;
    # global eos = convert(KnetArray{Float32},eos);
    #
    # global bos = falses(batch_size, length(word_to_index))
    # bos[:,end] = 1;
    # global bos = convert(KnetArray{Float32},bos);

    for i=1:batch_size:length(lengths)
        l = lengths[i]
        for j=i:i+batch_size-1
          (id,words), _ = caption_dict[j]
          input_ids[input_index][j-i+1] = id
          for k=index:index+l-1
             sequence[k][j-i+1, word_to_index[words[k-index+1]]] = 1
          end
        end
        index = index+l;
        input_index += 1;
    end

    println("Minibatching finished")
    return sequence,input_ids,lengths
end

function read_image_data(img, averageImage)
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
      c1 = permutedims(channelview(b1), (3,2,1))
      d1 = convert(Array{Float32}, c1)
      e1 = reshape(d1[:,:,1:3], (224,224,3,1))
      f1 = (255 * e1 .- averageImage)
      g1 = permutedims(f1, [2,1,3,4])
      x1 = KnetArray(g1)
end

# sequence[t]: input token at time t
# state is modified in place

function complete_karpathy_features(param, convnet, state, seq; batch_size=20, lr=1.0, gclip=0.0)
  sequence = seq[1]
  input_ids = seq[2]
  lengths = seq[3]
  index = 1; l=1; input_index = 1;
  count = 0;
  for t = 1:batch_size:length(lengths)
         l = lengths[t]
         for i=1:batch_size
               id = input_ids[input_index][i]
               filename = "./data/Flickr30k/karpathy/features/$id.jld";
               if isfile(filename)


               else
                println(id)
                image = read_image_data("./data/Flickr30k/flickr30k-images/$id.jpg", averageImage)
                image = convnet(image);
                image = convert(Array{Float32},image);
                save(filename, "feature", image)
                count += 1;
               end
         end
         index = index +l;
         input_index += 1
  end
  return count;
end

function train1(param,optim,convnet, state, seq; batch_size=20, lr=1.0, gclip=0.0, cnnout=4096)
  #complete_karpathy_features(convnet, seq, batch_size)
  sequence = seq[1]
  input_ids = seq[2]
  lengths = seq[3]

  #index_to_char = Array(String, length(vocab))
  #for (k,v) in vocab; index_to_char[v] = k; end
  atype = typeof(param[1])
  index = 1; l=1; input_index = 1;
  #for t = 1:batch_size:length(lengths)
 for t = 1:batch_size:10*batch_size

     l = lengths[t]
     input = Array(Float32, batch_size, cnnout);

     for i=1:batch_size
           id = input_ids[input_index][i]
           filename = "./data/Flickr30k/karpathy/features/$id.jld";
           if isfile(filename)
             input[i,:] = load(filename, "feature");
           else
             println(id,"not found in karpathy features")
             image = read_image_data("./data/Flickr30k/flickr30k-images/$id.jpg", averageImage)
             image = convnet(image);
             image = convert(Array{Float32},image);
             save(filename, "feature", image)
             input[i,:] = image;
           end
     end


     input = convert(KnetArray{Float32},input);

     # print(input_ids[input_index][1],":")
     # for i=index:index+l-1
     #   expected = find(sequence[i][1,:])
     #   print(index_to_char[expected[1]],' ');
     # end
     #println();

     gloss = lossgradient(param, state, input, sequence, index:index+l-1)

     if t%100 == 1
       println(t,".sentence trained")
       println("loss in sentence: ", loss(param, state, input, sequence, index:index+l-1))
     end

     if gclip > 0
        gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
         if gnorm > gclip
           for k=1:length(gloss)
             gloss[k] = (gloss[k] * gclip) / gnorm
           end
         end
     end

     for k in 1:length(param)
         update!(param, gloss, optim);
     end

     Knet.knetgc(); gc;

     isa(state,Vector{Any}) || error("State should not be Boxed.")
     # The following is needed in case AutoGrad boxes state values during gradient calculation
     for i = 1:length(state)
         state[i] = AutoGrad.getval(state[i])
     end

     index = index + l;
     input_index += 1

    end
end

function initparams(model)
  prms = Array(Any,length(model))
  for k=1:length(prms)
    prms[k] = Adam()
  end
  return prms
end

function report_loss(param, convnet, state, seq; batch_size=20, cnnout=4096)

  total = 0.0; count = 0

  atype = typeof(AutoGrad.getval(param[1]))

  sequence = seq[1]
  input_ids = seq[2]
  lengths = seq[3]

  index = 1; l=1; input_index = 1;

  for t = 1:batch_size:length(lengths)
    l = lengths[t]

    input = Array(Float32, batch_size, cnnout);

    for i=1:batch_size
          id = input_ids[input_index][i]
          filename = "./data/Flickr30k/karpathy/features/$id.jld";
          if isfile(filename)
            input[i,:] = load(filename, "feature");
          else
            println(id,"not found in karpathy features")
            image = read_image_data("./data/Flickr30k/flickr30k-images/$id.jpg", averageImage)
            image = convnet(image);
            image= convert(Array{Float32},image);
            save(filename, "feature", image)
            input[i,:] = image;
          end
    end


    input = convert(atype,input);

    lstm_input = bos;
    #input = ones(Float32, 1, 1000);

    for c in index:index+l-1
        ypred = lrcn(param,state,input,lstm_input)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        ygold = convert(atype, sequence[c])
        total += sum(ygold .* ynorm)
        count += size(ygold,1)
        lstm_input = ygold
    end

    ypred = lrcn(param,state,input,lstm_input)
    ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
    ygold = eos
    total += sum(ygold .* ynorm)
    count += size(ygold,1)

    index = index + l;
    input_index += 1;
  end
  return -total / count
end


function initweights(atype, hidden, vocab, embed, cnnout, winit)
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
  model[end-3] = winit*init(cnnout,Int(embed/2))
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

function lrcn(w,s, x_cnn, x_lstm)
  y = x_cnn*w[end-3];
  z = x_lstm*w[end-2];
  x = hcat(y,z);
  for i = 1:2:length(s)
      (s[i],s[i+1]) = lstm(w[i],w[i+1],s[i],s[i+1],x)
      x = s[i]
  end
  return x * w[end-1] .+ w[end]
end

function loss(param,state,input,sequence,range)
    total = 0.0; count = 0
    atype = typeof(AutoGrad.getval(param[1]))
    lstm_input = bos;

    for t in range
        ypred = lrcn(param,state,input,lstm_input)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        ygold = convert(atype, sequence[t])
        total += sum(ygold .* ynorm)
        count += size(ygold,1)
        lstm_input = ygold
    end

    ypred = lrcn(param,state,input,lstm_input)
    ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
    ygold = eos;
    total += sum(ygold .* ynorm)
    count += size(ygold,1)

    return -total / count
end

lossgradient = grad(loss);

function generate(param, convnet,  state, input, vocab, nword)
    index_to_char = Array(String, length(vocab))
    for (k,v) in vocab; index_to_char[v] = k; end

    input = convnet(input);
    index = 1;

    println("Generating Starts");
    lstm_input = bos;
    for i=1:nword
        ypred = lrcn(param,state,input,lstm_input)
        ynorm = logp(ypred,2);
        index = sample(exp(ynorm));
        lstm_input = convert(KnetArray{Float32}, ynorm .== maximum(ynorm))
        print(index_to_char[index], " ");
    end
    println();
    println("Generating Done")
end


function generate2(param, id,  state, vocab, nword)
    index_to_char = Array(String, length(vocab))
    for (k,v) in vocab; index_to_char[v] = k; end

    filename = "./data/Flickr30k/karpathy/features/$id.jld";
    input = zeros(Float32,1,4096);
    if isfile(filename)
      input[1,:] = load(filename, "feature");
    end
    input = convert(typeof(param[1]),input);

    index = 1
    println("Generating Starts");
    lstm_input = bos;
    for i=1:nword
        ypred = lrcn(param,state,input,lstm_input)
        ynorm = logp(ypred,2);
        index = sample(exp(ynorm));
        lstm_input = convert(KnetArray, ynorm .== maximum(ynorm))
        print(index_to_char[index], " ");
    end
    println();
    println("Generating Done")
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

function cnn_predict(convnet,x)
   return convnet(x);
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
