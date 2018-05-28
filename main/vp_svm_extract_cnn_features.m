function feature = vp_ExtractFeatuersCNN(im, net)
    % make sure it is up to CNNs standard
    img_ = single(im); % 255  range
    img_ = imresize(img_, net.meta.normalization.imageSize(1:2)); 
    img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage); % cnn_mean vp_mean

    % extract CNN features
    res = vl_simplenn(net, img_);
    
    % save them
    feature = squeeze(gather(res(end).x));
end