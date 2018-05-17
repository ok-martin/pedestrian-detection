function net = proj6_part2_cnn_init()

    % load the pretrained model
    net = load(fullfile('..','..','input-mat','imagenet-vgg-f.mat')) ;

    net.meta.inputSize = [224 224 3] ;
    net.meta.trainOpts.learningRate = 0.001;
    net.meta.trainOpts.batchSize = 50 ;
    net.meta.trainOpts.numEpochs = 11;
        
    % random number generator
    rng('default');
    rng(0);
    f=1/100;
    
    fc8InputDim = 4096; %10
    fc8OutputDim = 2; %15

    net.layers{end-1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,1,fc8InputDim,fc8OutputDim, 'single'), zeros(1, fc8OutputDim, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0, ...
                               'name', 'fc8') ;

    % overwrite soft max loss layer (with a new one)
    net.layers{end} = struct('type', 'softmaxloss') ;


    % replace the layer
    net.layers{end-1} = struct('type', 'conv', ...
                               'weights', {{f*randn(1,1,fc8InputDim,fc8OutputDim, 'single'), zeros(1, fc8OutputDim, 'single')}}, ...
                               'stride', 1, ...
                               'pad', 0, ...
                               'name', 'fc8') ;


    % overwrite softmaxloss (a new one)
    net.layers{end} = struct('type', 'softmaxloss') ;

    drop1 = struct('name', 'dropout1', 'type', 'dropout', 'rate' , 0.5) ;
    drop2 = struct('name', 'dropout2', 'type', 'dropout', 'rate' , 0.5) ;

    % insert a new dropout layer
    net = InsertLayer(net, drop1, 18);
    net = InsertLayer(net, drop2, 21);

    % Create the model. Fill in rest with defaul values                   
    net = vl_simplenn_tidy(net);

    vl_simplenn_display(net, 'inputSize', [224 224 3 50])

    % [copied from the project webpage]
    % proj6_part2_cnn_init.m will start with net = load('imagenet-vgg-f.mat');
    % and then edit the network rather than specifying the structure from
    % scratch.

    % You need to make the following edits to the network: The final two
    % layers, fc8 and the softmax layer, should be removed and specified again
    % using the same syntax seen in Part 1. The original fc8 had an input data
    % depth of 4096 and an output data depth of 1000 (for 1000 ImageNet
    % categories). We need the output depth to be 15, instead. The weights can
    % be randomly initialized just like in Part 1.

    % The dropout layers used to train VGG-F are missing from the pretrained
    % model (probably because they're not used at test time). It's probably a
    % good idea to add one or both of them back in between fc6 and fc7 and
    % between fc7 and fc8.
end

function net = InsertLayer(net, layer, index)
    saveLayer = net.layers{index};
    net.layers{index} = layer;

    for i=(index+1):size(net.layers, 2)
        saveLayer2 = net.layers{i};
        net.layers{i} = saveLayer;
        saveLayer = saveLayer2;
    end
    net.layers{i+1} = saveLayer;
end
