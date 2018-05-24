function vp_cnn_learn(varargin)
    
    % setup program variables
    run vp_vars.m;
    
    % start matconvnet
    run(prog.files.matconvnet);

    % training parameters for the CNN
    trainOpts = prog.trainOpts;    
    trainOpts = vl_argparse(trainOpts, varargin);
    
    % prep the CNN model
    net = vp_cnn_init();
    
    % load the prepared images
    imdb = load(prog.files.inImgDb);
    

    %%
    % adjust the classes
    net.meta.classes = imdb.meta.classes;
    net.meta.inputSize = [224 224 3];
    net.meta.trainOpts = prog.trainOpts;
    
    % save the cnn mean
    imdb.meta.cnn_mean = net.meta.normalization.averageImage;
    imdb.meta.imgSize = net.meta.normalization.imageSize(1:2);
    
    %% train the model
    %[net, info] = cnn_train(net, imdb, @getBatch, trainOpts, 'val', find(imdb.images.set == 2));   
    [net, info] = cnn_train(net, imdb, @getBatch, trainOpts);

    
    % Save the result
    save(prog.files.outMat, '-struct', 'net');
end


function [im, labels] = getBatch(imdb, batch)
    % get the images
    im = imdb.images.data(:,:,:,batch);
    
    % make sure it is up to CNNs standard
    im = single(im); % 255  range
    im = imresize(im, imdb.meta.imgSize); 
    im = bsxfun(@minus, im, imdb.meta.cnn_mean); % cnn_mean vp_mean
    
    % get the label for the image
    labels = imdb.images.labels(batch);

    % randomly flip images
    if rand > 0.5
        fliplr(im);
    end
    
    
%     figure(2);
%     montage(im);
%     disp(labels);

%     im = 256 * im;
%     im = reshape(im, 50, 30, 3, []);
%     im = reshape(im, 224, 224, 3, []);
%     im = imresize(im, image_size);
%     im = im2single(im);
end
