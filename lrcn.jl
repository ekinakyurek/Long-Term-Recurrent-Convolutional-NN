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


    # we initialize a model from loadfile, train using datafiles (both optional).
    # if the user specifies neither, train a model using the charlm.jl source code.
    isempty(o[:datafiles]) && o[:loadfile]==nothing && push!(o[:datafiles],Flickr30k_captions) # Flickr30k

    # read text and create global vocab variable

    #text = map((@compat readstring), o[:datafiles])
    #!isempty(text) && info("Chars read: $(map((f,c)->(basename(f),length(c)),o[:datafiles],text))")

    #gpu() >= 0 || error("LRCN only works on GPU machines.")

    #println(o[:hidden])
    # vocab (char_to_index) comes from the initial model if there is one, otherwise from the datafiles.
    # if there is an initial model make sure the data has no new vocab
    if o[:loadfile]==nothing

        println("Tokenization starts")
        #vocab = {"words1"=>1,"words2"=>2, ....}
        global vocab = Dict{String, Int}()
        #captions_dicts = [((id1, ["word1","word2",...]),length1),((id2, ["word1","word2",...]),length2),...]
        caption_dicts = Array{Array{Tuple{Tuple{Int64,Array{String,1}},Int64},1},1}();
        Tokenizer.tokenize(vocab, caption_dicts;data_files=o[:datafiles])
        get!(vocab,"~",1+length(vocab)); #eos
        get!(vocab,"`",1+length(vocab)); #bos


        info("Tokenization finished with captions size: $(sizeof(caption_dicts))")

        println("Intializing cnn_weights")

          if !isfile(o[:model])
              println("Should I download the VGG model (492MB)? Enter 'y' to download, anything else to quit.")
              readline() == "y\n" || return
              download(vggurl,o[:model])
          end

         info("Reading $(o[:model])")
         vgg = matread(o[:model])
         params = get_params_cnn(vgg)
         convnet = get_convnet(params...)
         global averageImage = convert(Array{Float32},vgg["meta"]["normalization"]["averageImage"])
         info("Cnn is initialized")

         println("Intializing LSTM weigths")
         model = initweights(o[:atype], o[:hidden], length(vocab), o[:embed], o[:cnnout],o[:winit])
         info("LSTM is initialized with size : $(size(model))")
    else
        # info("Loading model from $(o[:loadfile])")
        # vocab = load(o[:loadfile], "vocab")
        # for t in text, c in t; haskey(vocab, c) || error("Unknown char $c"); end
        # model = map(p->convert(o[:atype],p), load(o[:loadfile], "model"))
    end

    info("$(length(vocab)) unique words with size: $(sizeof(vocab))")


    #  info("Reading $(o[:image])")
    #img = data(o[:image], zeros(Float32,224,224,3,1))



    if o[:generate] > 0
        println("Generation starts")
        state = initstate(o[:atype],o[:hidden],1)
        img = read_image_data(o[:image], averageImage)
        global eos = falses(1, length(vocab))
        eos[:,end-1] = 1;
        global eos = convert(KnetArray{Float32},eos);

        global bos = falses(1, length(vocab))
        bos[:,end] = 1;
        global bos = convert(KnetArray{Float32},bos);
        #image = ones(224,224,3,1);
        generate(model, convnet, state, img, vocab, o[:generate])
        println("Generation finished")
    end

    if !isempty(caption_dicts)
      #y1 = cnn_predict(w,img)
      # z1 = vec(Array(y1))
      # s1 = sortperm(z1,rev=true)
      # p1 = exp(logp(z1))
      # display(hcat(p1[s1[1:o[:top]]], description[s1[1:o[:top]]]))
      # println()
      sequence = map(t->minibatch(t, vocab, o[:batchsize]), caption_dicts)
      #no need to caption dicts any more delete it from memory
      caption_dicts = 0; gc(); Knet.knetgc();
      println("Data is created with size: $(sizeof(sequence))")
      train!(model,convnet, sequence, vocab, o)
    end

    if o[:savefile] != nothing
        info("Saving last model to $(o[:savefile])")
        save(o[:savefile], "model", model, "vocab", vocab)
    end

end

function train!(model, convnet, sequence, vocab, o)
    s0 = initstate(o[:atype], o[:hidden], o[:batchsize])
    lr = o[:lr]
    if o[:fast]
        @time (for epoch=1:o[:epochs]
               train1(model, convnet, copy(s0), sequence[1]; batch_size=o[:batchsize], lr=lr, gclip=o[:gclip], cnnout=o[:cnnout])
               end; gpu()>=0 && Knet.cudaDeviceSynchronize())
        return
    end
    losses = map(d->report_loss(model, convnet, copy(s0), d ; batch_size=o[:batchsize], cnnout=o[:cnnout]), sequence)
    println((:epoch,0,:loss,losses...))
    devset = ifelse(length(data) > 1, 2, 1)
    devlast = devbest = losses[devset]
    for epoch=1:o[:epochs]
        @time train1(model, copy(s0), sequence[1]; slen=o[:seqlength], lr=lr, gclip=o[:gclip])
        @time losses = map(d->report_loss(model, convnet, copy(s0),d ; batch_size=o[:batchsize], cnnout=o[:cnnout]), sequence)
        println((:epoch,epoch,:loss,losses...))
        if o[:gcheck] > 0
            gradcheck(loss, model, copy(s0), sequence[1], 1:o[:seqlength]; gcheck=o[:gcheck])
        end
        devloss = losses[devset]
        if devloss < devbest
            devbest = devloss
            if o[:bestfile] != nothing
                info("Saving best model to $(o[:bestfile])")
                save(o[:bestfile], "model", model, "vocab", vocab)
            end
        end
        if devloss > devlast
            lr *= o[:decay]
            info("New learning rate: $lr")
        end
        devlast = devloss
    end
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

    global eos = falses(batch_size, length(word_to_index))
    eos[:,end-1] = 1;
    global eos = convert(KnetArray{Float32},eos);

    global bos = falses(batch_size, length(word_to_index))
    bos[:,end] = 1;
    global bos = convert(KnetArray{Float32},bos);

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

function train1(param, convnet, state, seq; batch_size=20, lr=1.0, gclip=0.0, cnnout=4096)
   #complete_karpathy_features(param, convnet, state, seq; batch_size=batch_size, lr=lr, gclip=gclip)
   sequence = seq[1]
   input_ids = seq[2]
   lengths = seq[3]


    index = 1; l=1; input_index = 1;

    for t = 1:batch_size:length(lengths)
        l = lengths[t]
        #input_ids = inputs_ids[index:index+l-1]
        #println(input_id)
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

        input = convert(KnetArray{Float32},input);
        #input = cnn_predict(convnet, input)
        #input = ones(Float32, batch_size, 1000);
        gloss = lossgradient(param, state, input, sequence, index:index+l-1)

        if t%1000 == 1
          println(t,". sentence trained")
          println("loss in sentence: ", loss(param, state, input, sequence, index:index+l-1))
          Knet.knetgc();gc();
        end

        gscale = lr
        if gclip > 0
            gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
            if gnorm > gclip
                gscale *= gclip / gnorm
            end
        end

        for k in 1:length(param)
            # param[k] -= gscale * gloss[k]
            axpy!(-gscale, gloss[k], param[k])
        end

        isa(state,Vector{Any}) || error("State should not be Boxed.")
        # The following is needed in case AutoGrad boxes state values during gradient calculation
        for i = 1:length(state)
            state[i] = AutoGrad.getval(state[i])
        end

        index = index +l;
        input_index += 1
    end
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

    input = Array(Float32,batch_size,cnnout);

    for i=1:batch_size
          id = input_ids[input_index][i]
          filename = "./data/Flickr30k/karpathy/features/$id.jld";
          if isfile(filename)
            input[i,:] = load(filename, "feature");
          else
            println(id,"not found in karpathy features")
            image = read_image_data("./data/Flickr30k/flickr30k-images/$id.jpg", averageImage)
            image = convnet(image);
            input[i,:] = convert(Array{Float32},image);
          end
    end

    #input = ones(Float32,224,224,3,1);
    #input = cnn_predict(convnet, input)
    input = convert(atype,input)
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
    param = Array(Any, 2*length(hidden)+3)
    input = embed
    for k = 1:length(hidden)
        param[2k-1] = winit*randn(input+hidden[k], 4*hidden[k])
        param[2k]   = zeros(1, 4*hidden[k])
        param[2k][1:hidden[k]] = 1 # forget gate bias
        input = hidden[k]
    end
    param[end-2] = winit*randn(cnnout+vocab,embed)
    param[end-1] = winit*randn(hidden[end],vocab)
    param[end] = zeros(1,vocab)
    return map(p->convert(atype,p), param)
end

# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2*length(hidden))
    for k = 1:length(hidden)
        state[2k-1] = zeros(batchsize,hidden[k])
        state[2k] = zeros(batchsize,hidden[k])
    end
    return map(s->convert(atype,s), state)
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
  x = hcat(x_cnn,x_lstm) * w[end-2]
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
    index = 1
    println("Generating Starts");
    lstm_input = bos;
    for i=1:nword
        ypred = lrcn(param,state,input,lstm_input)
        ynorm = logp(ypred);
        lstm_input = (ynorm .== maximum(ynorm)) .* ynorm;
        index = sample(exp(ynorm));
        if index == length(vocab)
          #eos
          break;
        end
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
