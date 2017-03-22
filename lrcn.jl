for p in ("Knet","ArgParse","ImageMagick","MAT","Images", "Compat", "QuartzImageIO")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet

!isdefined(:VGG) && include(Knet.dir("examples","vgg.jl"))

"""
julia lrcn.jl image-file-or-url

This example implements the Long-term recurrent convolutional network model from

Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

* Paper url: https://arxiv.org/pdf/1411.4389.pdf
* Project page: https://github.com/ekinakyurek/lrcn
"""
module LRCN

using Knet, AutoGrad, ArgParse,Compat, MAT, Images

using Main.VGG: data, imgurl

const imgurl = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
const default_file = "data/Flickr30k/results_20130124.token"
const eos = "~~"
function main(args=ARGS)
    s = ArgParseSettings()

    s.description = "LRCN.jl (c) Ekin Akyürek, 2017. Long-term Recurrent Convolutional Networks for Visual Recognition and Description"

    @add_arg_table s begin
        ("image"; default=imgurl; help="Image file or URL.")
        ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--bestfile"; help="Save best model to file")
        ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of characters.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[1000]; help="Sizes of one or more LSTM layers.")
        ("--embed"; arg_type=Int; default=1000; help="Size of the embedding vector.")
        ("--epochs"; arg_type=Int; default=20; help="Number of epochs for training.")
        ("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
        ("--seqlength"; arg_type=Int; default=100; help="Number of steps to unroll the network for.")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=4.0; help="Initial learning rate.")
        ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
        ("--winit"; arg_type=Float64; default=0.3; help="Initial weights set to winit*randn().")
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

    # we initialize a model from loadfile, train using datafiles (both optional).
    # if the user specifies neither, train a model using the charlm.jl source code.
    isempty(o[:datafiles]) && o[:loadfile]==nothing && push!(o[:datafiles],default_file) # Flickr30k
    include("tokenizer.jl")
    # read text and create global vocab variable

    #text = map((@compat readstring), o[:datafiles])
    #!isempty(text) && info("Chars read: $(map((f,c)->(basename(f),length(c)),o[:datafiles],text))")

    #gpu() >= 0 || error("LRCN only works on GPU machines.")

    #println(o[:hidden])
    # vocab (char_to_index) comes from the initial model if there is one, otherwise from the datafiles.
    # if there is an initial model make sure the data has no new vocab
    if o[:loadfile]==nothing
        vocab, caption_dicts = Tokenizer.tokenize(data_files=o[:datafiles])
        get!(vocab,eos,1+length(vocab)); #eos
        w = cnn_weights(o[:embed]);
        model = initweights(o[:atype], o[:hidden], length(vocab), o[:embed], o[:winit])
    else
        # info("Loading model from $(o[:loadfile])")
        # vocab = load(o[:loadfile], "vocab")
        # for t in text, c in t; haskey(vocab, c) || error("Unknown char $c"); end
        # model = map(p->convert(o[:atype],p), load(o[:loadfile], "model"))
    end

    info("$(length(vocab)) unique words.")


  #  info("Reading $(o[:image])")
    #img = data(o[:image], zeros(Float32,224,224,3,1))


    if o[:generate] > 0
        state = initstate(o[:atype],o[:hidden],1)
        img = data(o[:image], zeros(Float32,224,224,3,1))
        #image = ones(224,224,3,1);
        generate(model, w, state, img, vocab, o[:generate])
    end

    if !isempty(caption_dicts)
      #y1 = cnn_predict(w,img)
      # z1 = vec(Array(y1))
      # s1 = sortperm(z1,rev=true)
      # p1 = exp(logp(z1))
      # display(hcat(p1[s1[1:o[:top]]], description[s1[1:o[:top]]]))
      # println()
      train!(model,w, caption_dicts, vocab, o)
    end

    if o[:savefile] != nothing
        info("Saving last model to $(o[:savefile])")
        save(o[:savefile], "model", model, "vocab", vocab)
    end

end

function train!(model, w, caption_dicts, vocab, o)
    s0 = initstate(o[:atype], o[:hidden], o[:batchsize])
    data = map(t->minibatch(t, vocab, o[:batchsize]), caption_dicts)
    lr = o[:lr]
    if o[:fast]
        @time (for epoch=1:o[:epochs]
               train1(model,w, copy(s0), data[1]; slen=o[:seqlength], lr=lr, gclip=o[:gclip])
               end; gpu()>=0 && Knet.cudaDeviceSynchronize())
        return
    end
    losses = map(d->report_loss(model, w, copy(s0),d), data)
    println((:epoch,0,:loss,losses...))
    devset = ifelse(length(data) > 1, 2, 1)
    devlast = devbest = losses[devset]
    for epoch=1:o[:epochs]
        @time train1(model, copy(s0), data[1]; slen=o[:seqlength], lr=lr, gclip=o[:gclip])
        @time losses = map(d->report_loss(model, w,copy(s0),d), data)
        println((:epoch,epoch,:loss,losses...))
        if o[:gcheck] > 0
            gradcheck(loss, model, copy(s0), data[1], 1:o[:seqlength]; gcheck=o[:gcheck])
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

function get_word_count(caption_dict)
  count = 0;
  for (k,v) in caption_dict
    for i=1:5
      for word in v[i]
          if length(word) > 0
            count += 1;
          end
      end
      #eos
      count += 1;
    end
  end
  return count;
end



function minibatch(caption_dict, word_to_index, batch_size)
    println("MinitBatching Starts")
    nbatch = div(get_word_count(caption_dict), batch_size)
    data = [falses(1, length(word_to_index)) for i=1:nbatch ]
    ranges = Array{UnitRange}(length(caption_dict)*5);
    inputs = Array{Int}(length(caption_dict)*5);

    id = 1
    caption = 1;
    for (k,v) in caption_dict
      for i=1:5
        #data size for one caption  1+length(v[i]) (+1 for eos)
        ranges[caption] = caption > 1 ? (last(ranges[caption-1])+1:1+length(v[i])+last(ranges[caption-1])):(1:1+length(v[i]))
        inputs[caption] = k;
        caption += 1;
        for word in v[i]
          if length(word) > 0
            col = word_to_index[word]
            data[id][col] = 1
            id += 1;
          end
        end
        #eos
        data[id][end] = 1
        id += 1;
      end
    end
    println("MinitBatching Finished")
    return data,inputs,ranges
end

# sequence[t]: input token at time t
# state is modified in place
function train1(param, w, state, data; slen=100, lr=1.0, gclip=0.0)
   sequence = data[1]
   inputs = data[2]
   ranges = data[3]
    for t = 1:length(ranges)
        input_id = inputs[t]
        #println(input_id)
        input = Main.VGG.data("./data/Flickr30k/flickr30k-images/$input_id.jpg", zeros(Float32,224,224,3,1))
        #input = ones(Float32,224,224,3,1);
        input = cnn_predict(w, input)
        #input = ones(Float32, 1, 1000);
        @time gloss = lossgradient(param, state, input, sequence, ranges[t])
        println(t,". sentence trained")
        println("loss in sentence: ", loss(param, state, input, sequence, ranges[t]))
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
    end
end


function initweights(atype, hidden, vocab, embed, winit)
    param = Array(Any, 2*length(hidden)+3)
    input = embed
    for k = 1:length(hidden)
        param[2k-1] = winit*randn(input+hidden[k], 4*hidden[k])
        param[2k]   = zeros(1, 4*hidden[k])
        param[2k][1:hidden[k]] = 1 # forget gate bias
        input = hidden[k]
    end
    param[end-2] = winit*randn(embed,embed)
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

function lrcn(w,s,x)
  x = x * w[end-2]
  for i = 1:2:length(s)
      (s[i],s[i+1]) = lstm(w[i],w[i+1],s[i],s[i+1],x)
      x = s[i]
  end
  return x * w[end-1] .+ w[end]
end



function loss(param,state,input, sequence,range)
    total = 0.0; count = 0
    atype = typeof(AutoGrad.getval(param[1]))
    input = convert(atype,input)
    for t in range
        ypred = lrcn(param,state,input)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        ygold = convert(atype, sequence[t])
        total += sum(ygold .* ynorm)
        count += size(ygold,1)
        #input = ygold
    end
    return -total / count
end


function generate(param, w,  state, input, vocab, nword)
    index_to_char = Array(String, length(vocab))
    for (k,v) in vocab; index_to_char[v] = k; end
    input = cnn_predict(w, input)
    #input = ones(1,1000);
    index = 1
    println("Generating Starts");
    for i=1:nword
        ypred = lrcn(param,state,input)
        index = sample(exp(logp(ypred)))
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

function report_loss(param, w, state, data)
  total = 0.0; count = 0
  atype = typeof(AutoGrad.getval(param[1]))
  sequence = data[1]
  inputs = data[2]
  ranges = data[3]

  for i=1:length(ranges)
    input_id = inputs[i]
    input = Main.VGG.data("./data/Flickr30k/flickr30k-images/$input_id.jpg", zeros(Float32,224,224,3,1))
    #input = ones(Float32,224,224,3,1);
    input = cnn_predict(w, input)
    #input = ones(Float32, 1, 1000);

    for t in ranges[i]
        ypred = lrcn(param,state,input)
        ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
        ygold = convert(atype, sequence[t])
        total += sum(ygold .* ynorm)
        count += size(ygold,1)
        #input = ygold
    end

  end
  return -total / count
end

lossgradient = grad(loss);

function train(w, data; lr=.1, epochs=20, nxy=0)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(w, x, y)
            for i in 1:length(w)
                # w[i] -= lr * g[i]
                axpy!(-lr, g[i], w[i])
            end
        end
    end
    return w
end

function batchnorm(w, x, ms; mode=1, epsilon=1e-5)
    mu, sigma = nothing, nothing
    if mode == 0
        d = ndims(x) == 4 ? (1,2,4) : (2,)
        s = reduce((a,b)->a*size(x,b), d)
        mu = sum(x,d) / s
        sigma = sqrt(epsilon + (sum(x.-mu,d).^2) / s)
    elseif mode == 1
        mu = shift!(ms)
        sigma = shift!(ms)
    end

    # we need getval in backpropagation
    push!(ms, AutoGrad.getval(mu), AutoGrad.getval(sigma))
    xhat = (x.-mu) ./ sigma
    return w[1] .* xhat .+ w[2]
end

function get_params(params)
    len = length(params["value"])
    ws, ms = [], []
    for k = 1:len
        name = params["name"][k]
        value = convert(Array{Float32}, params["value"][k])
        if endswith(name, "moments")
            push!(ms, reshape(value[:,1], (1,1,size(value,1),1)))
            push!(ms, reshape(value[:,2], (1,1,size(value,1),1)))
        elseif startswith(name, "bn")
            push!(ws, reshape(value, (1,1,length(value),1)))
        elseif startswith(name, "fc") && endswith(name, "filter")
            push!(ws, transpose(reshape(value,size(value,3,4))))
        elseif startswith(name, "conv") && endswith(name, "bias")
            push!(ws, reshape(value, (1,1,length(value),1)))
        else
            push!(ws, value)
        end
    end
    map(KnetArray, ws), map(KnetArray, ms)
end

function cnn_loss(w,x,ygold)
    ypred = lrcn(w,x)
    ynorm = logp(ypred,1)  # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

function cnn_predict(w,x)
  x = pool(relu(conv4(w[1],x;padding=0) .+ w[2]); window=2, stride=2);
  x = pool(tanh(conv4(w[3],x;padding=0) .+ w[4]); window=2, stride=2);
  #println(size(x))
  #DBG Change reshape to make it row vector not column since it goes to LSTM eventually
  x = reshape(x,(53*53*12,size(x,4)));
  x = tanh(w[5]*x .+ w[6]);
  x = reshape(x, (1,1000));
  return x;
end

function cnn_weights(embed; atype=KnetArray{Float32}, winit=0.1)
  w = Array(Any, 6)
  w[1] = randn(Float32,5,5,3,12)*winit; w[2] = zeros(Float32,1,1,12,1);
  w[3] = randn(Float32,5,5,12,12)*winit; w[4] = zeros(Float32,1,1,12,1);
  w[5] = randn(Float32,embed,53*53*12)*winit; w[6] = zeros(Float32,embed,1);
  return map(a->convert(atype,a), w);
end

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
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
