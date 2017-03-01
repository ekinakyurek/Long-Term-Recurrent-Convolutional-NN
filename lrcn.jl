for p in ("Knet","ArgParse","ImageMagick","MAT","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet
!isdefined(:VGG) && (local lo=isdefined(:load_only); load_only=true;include(Knet.dir("examples","vgg.jl")); load_only=lo)

"""
julia lrcn.jl image-file-or-url

This example implements the Long-term recurrent convolutional network model from

Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

* Paper url: https://arxiv.org/pdf/1411.4389.pdf
* Project page: https://github.com/ekinakyurek/lrcn
"""
module LRCN
using Knet, AutoGrad, ArgParse, MAT, Images
using Main.VGG: data, imgurl
const modelurl = "http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat";

function main(args)
    s = ArgParseSettings()
    s.description = "LRCN.jl (c) Ekin Akyürek, 2017. Long-term Recurrent Convolutional Networks for Visual Recognition and Description"

    @add_arg_table s begin
        ("image"; default="./pictures/cat.jpeg"; help="Image file or URL.")
        ("--model"; default=Knet.dir("data", "imagenet-resnet-101-dag.mat");
         help="resnet MAT file path")
        ("--top"; default=5; arg_type=Int; help="Display the top N classes")
    end

    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)

    gpu() >= 0 || error("LRCN only works on GPU machines.")

    if !isfile(o[:model])
        println("Should I download the ResNet-101 model (160MB)?",
                " Enter 'y' to download, anything else to quit.")
        readline() == "y\n" || return
        download(modelurl,o[:model])
    end

    info("Reading $(o[:model])")
    model = matread(abspath(o[:model]))
    avgimg = model["meta"]["normalization"]["averageImage"]
    avgimg = convert(Array{Float32}, avgimg)
    description = model["meta"]["classes"]["description"]
    w, ms = get_params(model["params"])

    info("Reading $(o[:image])")
    img = data(o[:image], avgimg)

    get model by length of parameters
    modeldict = Dict(
        162 => (resnet50, "resnet50"),
        314 => (resnet101, "resnet101"),
        467 => (resnet152, "resnet152"))
    !haskey(modeldict, length(w)) && error("wrong resnet MAT file")
    CNN, name = modeldict[length(w)]

    info("Classifying with ", name)
    @time y1 = CNN(w,img,ms)
    z1 = vec(Array(y1))
    s1 = sortperm(z1,rev=true)
    p1 = exp(logp(z1))
    display(hcat(p1[s1[1:o[:top]]], description[s1[1:o[:top]]]))
    println()
end


function lrcn(w,x)
  return x;
end

function loss(w,x,ygold)
    ypred = lrcn(w,x)
    ynorm = logp(ypred,1)  # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
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

function weights(;ftype=Float32,atype=KnetArray)
    w = Array(Any, 116)
    w[1] = xavier(Float32,7,7,3,64);    w[2] = zeros(Float32,1,1,64,1);
    w[3] = xavier(Float32,1,1,64,64);   w[4] = zeros(Float32,1,1,64,1);
    w[5] = xavier(Float32,3,3,64,192);  w[6] = zeros(Float32,1,1,192,1);
    #inception_3a
    w[7] = xavier(Float32,1,1,192,64);  w[8] = zeros(Float32,1,1,64,1);
    w[9] = xavier(Float32,1,1,192,96);  w[10] = zeros(Float32,1,1,96,1);
    w[11] = xavier(Float32,1,1,192,16); w[12] = zeros(Float32,1,1,16,1);
    w[13] = xavier(Float32,3,3,96,128); w[14] = zeros(Float32,1,1,128,1);
    w[15] = xavier(Float32,5,5,16,32);  w[16] = zeros(Float32,1,1,32,1);
    w[17] = xavier(Float32,1,1,192,32); w[18] = zeros(Float32,1,1,32,1);
    #inception_3b
    w[19] = xavier(Float32,1,1,256,128); w[20] = zeros(Float32,1,1,128,1);
    w[21] = xavier(Float32,1,1,256,128); w[22] = zeros(Float32,1,1,128,1);
    w[23] = xavier(Float32,1,1,256,32); w[24] = zeros(Float32,1,1,32,1);
    w[25] = xavier(Float32,3,3,128,192); w[26] = zeros(Float32,1,1,192,1);
    w[27] = xavier(Float32,5,5,32,96);  w[28] = zeros(Float32,1,1,96,1);
    w[29] = xavier(Float32,1,1,256,64); w[30] = zeros(Float32,1,1,64,1);
    #inception_4a
    w[31] = xavier(Float32,1,1,480,192); w[32] = zeros(Float32,1,1,192,1);
    w[33] = xavier(Float32,1,1,480,96); w[34] = zeros(Float32,1,1,96,1);
    w[35] = xavier(Float32,1,1,480,16); w[36] = zeros(Float32,1,1,16,1);
    w[37] = xavier(Float32,3,3,96,208); w[38] = zeros(Float32,1,1,208,1);
    w[39] = xavier(Float32,5,5,16,48);  w[40] = zeros(Float32,1,1,48,1);
    w[41] = xavier(Float32,1,1,480,64); w[42] = zeros(Float32,1,1,64,1);
    #inception_4b
    w[43] = xavier(Float32,1,1,512,160); w[44] = zeros(Float32,1,1,160,1);
    w[45] = xavier(Float32,1,1,512,112); w[46] = zeros(Float32,1,1,112,1);
    w[47] = xavier(Float32,1,1,512,24); w[48] = zeros(Float32,1,1,24,1);
    w[49] = xavier(Float32,3,3,112,224); w[50] = zeros(Float32,1,1,224,1);
    w[51] = xavier(Float32,5,5,24,64); w[52] = zeros(Float32,1,1,64,1);
    w[53] = xavier(Float32,1,1,512,64);  w[54] = zeros(Float32,1,1,64,1);
    #inception_4c
    w[55] = xavier(Float32,1,1,512,128); w[56] = zeros(Float32,1,1,128,1);
    w[57] = xavier(Float32,1,1,512,128); w[58] = zeros(Float32,1,1,128,1);
    w[59] = xavier(Float32,1,1,512,24); w[60] = zeros(Float32,1,1,24,1);
    w[61] = xavier(Float32,3,3,128,256); w[62] = zeros(Float32,1,1,256,1);
    w[63] = xavier(Float32,5,5,24,64); w[64] = zeros(Float32,1,1,64,1);
    w[65] = xavier(Float32,1,1,512,64);  w[66] = zeros(Float32,1,1,64,1);
    #inception_4d
    w[67] = xavier(Float32,1,1,512,112); w[68] = zeros(Float32,1,1,112,1);
    w[69] = xavier(Float32,1,1,512,144); w[70] = zeros(Float32,1,1,144,1);
    w[71] = xavier(Float32,1,1,512,32); w[72] = zeros(Float32,1,1,32,1);
    w[73] = xavier(Float32,3,3,144,288); w[74] = zeros(Float32,1,1,288,1);
    w[75] = xavier(Float32,5,5,32,64); w[76] = zeros(Float32,1,1,64,1);
    w[77] = xavier(Float32,1,1,512,64);  w[78] = zeros(Float32,1,1,64,1);
    #inception_4e
    w[79] = xavier(Float32,1,1,528,256); w[80] = zeros(Float32,1,1,256,1);
    w[81] = xavier(Float32,1,1,528,160); w[82] = zeros(Float32,1,1,160,1);
    w[83] = xavier(Float32,1,1,528,32); w[84] = zeros(Float32,1,1,32,1);
    w[85] = xavier(Float32,3,3,160,320); w[86] = zeros(Float32,1,1,320,1);
    w[87] = xavier(Float32,5,5,32,128); w[88] = zeros(Float32,1,1,128,1);
    w[89] = xavier(Float32,1,1,528,128);  w[90] = zeros(Float32,1,1,128,1);
    #inception_5a
    w[91] = xavier(Float32,1,1,832,256); w[92] = zeros(Float32,1,1,256,1);
    w[93] = xavier(Float32,1,1,832,160); w[94] = zeros(Float32,1,1,160,1);
    w[95] = xavier(Float32,1,1,832,32); w[96] = zeros(Float32,1,1,32,1);
    w[97] = xavier(Float32,3,3,160,320); w[98] = zeros(Float32,1,1,320,1);
    w[99] = xavier(Float32,5,5,32,128); w[100] = zeros(Float32,1,1,128,1);
    w[101]= xavier(Float32,1,1,832,128);  w[102] = zeros(Float32,1,1,128,1);
    #inception_5b
    w[103] = xavier(Float32,1,1,832,384); w[104] = zeros(Float32,1,1,384,1);
    w[105] = xavier(Float32,1,1,832,192); w[106] = zeros(Float32,1,1,192,1);
    w[107] = xavier(Float32,1,1,832,48); w[108] = zeros(Float32,1,1,48,1);
    w[109] = xavier(Float32,3,3,192,384); w[110] = zeros(Float32,1,1,384,1);
    w[111] = xavier(Float32,5,5,48,128); w[112] = zeros(Float32,1,1,128,1);
    w[113]= xavier(Float32,1,1,832,128);   w[114] = zeros(Float32,1,1,128,1);
    #FC layers
    w[115]= xavier(Float32,1000,1024);   w[116] = zeros(Float32,1000,1);
    return map(a->convert(atype,a), w)
  end


# mode, 0=>train, 1=>test
function resnet50(w,x,ms; mode=1)
    # layer 1
    conv1  = conv4(w[1],x; padding=3, stride=2) .+ w[2]
    bn1    = batchnorm(w[3:4],conv1,ms; mode=mode)
    pool1  = pool(bn1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[5:34], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[35:73], r2, ms; mode=mode)
    r4 = reslayerx5(w[74:130], r3, ms; mode=mode) # 5
    r5 = reslayerx5(w[131:160], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[161] * mat(pool5) .+ w[162]
end

# mode, 0=>train, 1=>test
function resnet101(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:72], r2, ms; mode=mode)
    r4 = reslayerx5(w[73:282], r3, ms; mode=mode)
    r5 = reslayerx5(w[283:312], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[313] * mat(pool5) .+ w[314]
end

# mode, 0=>train, 1=>test
function resnet152(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:108], r2, ms; mode=mode)
    r4 = reslayerx5(w[109:435], r3, ms; mode=mode)
    r5 = reslayerx5(w[436:465], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[466] * mat(pool5) .+ w[467]
end
# mode, 0=>train, 1=>test
function resnet50(w,x,ms; mode=1)
    # layer 1
    conv1  = conv4(w[1],x; padding=3, stride=2) .+ w[2]
    bn1    = batchnorm(w[3:4],conv1,ms; mode=mode)
    pool1  = pool(bn1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[5:34], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[35:73], r2, ms; mode=mode)
    r4 = reslayerx5(w[74:130], r3, ms; mode=mode) # 5
    r5 = reslayerx5(w[131:160], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[161] * mat(pool5) .+ w[162]
end

# mode, 0=>train, 1=>test
function resnet101(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:72], r2, ms; mode=mode)
    r4 = reslayerx5(w[73:282], r3, ms; mode=mode)
    r5 = reslayerx5(w[283:312], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[313] * mat(pool5) .+ w[314]
end

# mode, 0=>train, 1=>test
function resnet152(w,x,ms; mode=1)
    # layer 1
    conv1 = reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = pool(conv1; window=3, stride=2)

    # layer 2,3,4,5
    r2 = reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = reslayerx5(w[34:108], r2, ms; mode=mode)
    r4 = reslayerx5(w[109:435], r3, ms; mode=mode)
    r5 = reslayerx5(w[436:465], r4, ms; mode=mode)

    # fully connected layer
    pool5  = pool(r5; stride=1, window=7, mode=2)
    fc1000 = w[466] * mat(pool5) .+ w[467]
end

# Batch Normalization Layer
# works both for convolutional and fully connected layers
# mode, 0=>train, 1=>test
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

function reslayerx0(w,x,ms; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms; mode=mode)
end

function reslayerx1(w,x,ms; padding=0, stride=1, mode=1)
    relu(reslayerx0(w,x,ms; padding=padding, stride=stride, mode=mode))
end

function reslayerx2(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    ba = reslayerx1(w[1:3],x,ms; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba,ms; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb,ms; padding=pads[3], stride=strides[3], mode=mode)
end

function reslayerx3(w,x,ms; pads=[0,0,1,0], strides=[2,2,1,1], mode=1) # 12
    a = reslayerx0(w[1:3],x,ms; stride=strides[1], padding=pads[1], mode=mode)
    b = reslayerx2(w[4:12],x,ms; strides=strides[2:4], pads=pads[2:4], mode=mode)
    relu(a .+ b)
end

function reslayerx4(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    relu(x .+ reslayerx2(w,x,ms; pads=pads, strides=strides, mode=mode))
end

function reslayerx5(w,x,ms; strides=[2,2,1,1], mode=1)
    x = reslayerx3(w[1:12],x,ms; strides=strides, mode=mode)
    for k = 13:9:length(w)
        x = reslayerx4(w[k:k+8],x,ms; mode=mode)
    end
    return x
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
