function net = vp_cnn_init()
    % setup program variables
    run vp_vars.m;
    
    % numer of classes
    categories = 2;
    % index of the fc7 layer
    fc7 = 15; % 33
    % create the model variable
    net = [];

    % load the pretrained model
    old_net = load(prog.files.inMat);
 
    % add dropout layers in network (saved model has dropout removed)
    drop1 = struct('name', 'dropout6', 'type', 'dropout', 'rate' , 0.5);
    drop2 = struct('name', 'dropout7', 'type', 'dropout', 'rate' , 0.5);

    % insert after just before fc7, and just after fc7 (after relu 7)
    old_net.layers = [old_net.layers(1:fc7) drop1 ... 
        old_net.layers((fc7+1):(fc7+2)) drop2 old_net.layers(fc7+3:end)];

    % ignore the classification and last softmax layers
    net = old_net;
    net.layers = net.layers(1:end-2);
    
    % ---------------------------------------------------------------------
    % Classification
    % add own conv layer and loss layer
    net.layers{end+1} = struct('type', 'conv','name','fc8', 'weights', {{zeros(1,1,4096,categories, 'single'), ...
        zeros(categories,1,'single')}}, ...
        'stride',1,'pad',0,'learningRate',[1,2],'weightDecay',[1,0],'momentum', ... 
        {{zeros(1,1,4096,categories, 'single'), ... 
        zeros(categories,1,'single')}},'precious',false);

    % add loss layer
    net.layers{end+1} = struct('type', 'softmaxloss'); 

    % create the model, fill in rest with defaul values    
    net = vl_simplenn_tidy(net);
        
    % ---------------------------------------------------------------------
    % show the layers in the model
    vl_simplenn_display(net, 'inputSize', [224 224 3 50]);
end

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
    
        % random number generator
%     rng('default');
%     rng(0);
%     f=1/100;
%     
%     fc8InputDim = 4096; %10
%     fc8OutputDim = 2; %15
% 
%     % extract weights for the first 2 categories
%     vectors = net.layers{end-1}.weights{1,1}(1,1,:,1:2);
%     vectorw = net.layers{end-1}.weights{1,2}(1:2);

%     % replace the fc8 layer
%     net.layers{end-1} = struct('type', 'conv', ...
%                                'weights', {{f*randn(1,1,fc8InputDim,fc8OutputDim, 'single'), zeros(1, fc8OutputDim, 'single')}}, ...
%                                'stride', 1, ...
%                                'pad', 0, ...
%                                'name', 'fc8') ;

%     % insert new dropout layers
%     drop1 = struct('name', 'dropout1', 'type', 'dropout', 'rate' , 0.5) ;
%     drop2 = struct('name', 'dropout2', 'type', 'dropout', 'rate' , 0.5) ;
% 
%     net = InsertLayer(net, drop1, 18);
%     net = InsertLayer(net, drop2, 21);

% function net = InsertLayer(net, layer, index)
%     saveLayer = net.layers{index};
%     net.layers{index} = layer;
% 
%     for i=(index+1):size(net.layers, 2)
%         saveLayer2 = net.layers{i};
%         net.layers{i} = saveLayer;
%         saveLayer = saveLayer2;
%     end
%     net.layers{i+1} = saveLayer;
% end
