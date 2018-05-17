function [net, info] = proj6_part2()

    prog.imgSize = [224 224];
    prog.files.inMat = fullfile('..','..','input-mat','imagenet-vgg-f.mat');
    prog.files.outEpch = fullfile('..','..','output-mat', 'prj6');
    prog.files.outMat = fullfile('..','..','output-mat', 'genius.mat');
    prog.files.inImg = fullfile('..','..','input-images');

    run(fullfile('..','matconvnet', 'matlab', 'vl_setupnn.m')) ;
    % run(fullfile('vlfeat-0.9.20', 'toolbox', 'vl_setup.m'));

    % where trained networks and plots are saved.
    opts.expDir = prog.files.outEpch;

    % the number of training images in each batch
    opts.batchSize = 50 ;
    
    % affects the error
    opts.learningRate = 0.00001 ;

    % 
    opts.numEpochs = 10 ;

    % learning rate decay as an alternative to the fixed learning rate 
    % opts.learningRate = logspace(-4, -5.5, 300) ;
    % opts.numEpochs = numel(opts.learningRate) ;

    %opts.continue controls whether to resume training from the furthest
    %trained network found in opts.batchSize. If you want to modify something
    %mid training (e.g. learning rate) this can be useful. You might also want
    %to resume a network that hit the maximum number of epochs if you think
    %further training can improve accuracy.
    opts.continue = true ;

    %GPU support is off by default.
    % opts.gpus = [] ;

    % This option lets you control how many of the layers are fine-tuned.
    % opts.backPropDepth = 2; %just retrain the last real layer (1 is softmax)
    % opts.backPropDepth = 9; %just retrain the fully connected layers
    % opts.backPropDepth = +inf; %retrain all layers [default]

    % --------------------------------------------------------------------

    % prep data
    net = proj6_part2_cnn_init(prog);

    imdb = proj6_part2_setup_data(prog, net.meta.normalization.averageImage);

    % Train
    [net, info] = cnn_train(net, imdb, @getBatch, opts, 'val', find(imdb.images.set == 2));

    % Save the result for later use
    net.layers(end) = [] ;
    net.avgImg = imdb.avgImg;
    save(prog.files.outMat, '-struct', 'net') ;
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
    % getBatch is called by cnn_train.

    %'imdb' is the image database.
    %'batch' is the indices of the images chosen for this batch.

    %'im' is the height x width x channels x num_images stack of images. If
    % opts.batchSize is 50 and image size is 64x64 and grayscale, im will be 64x64x1x50.
    %'labels' indicates the ground truth category of each image.

    %This function is where you should 'jitter' data.
    % --------------------------------------------------------------------
    image_size = [224 224];
    im = imdb.images.data(:,:,:,batch) ;
    im = imresize(im, image_size);
    labels = imdb.images.labels(1,batch) ;
    % Add jittering here before returning im
end