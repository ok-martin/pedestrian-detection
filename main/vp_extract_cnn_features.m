% -------------------------------------------------------------------------
function feats = vp_extract_cnn_features(img, net)
    % make sure it is up to CNNs standard
    img_ = single(img); % 255  range
    img_ = imresize(img_, net.meta.normalization.imageSize(1:2)); 
    img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage); % cnn_mean vp_mean

    % extract the features
    res = vl_simplenn(net, img_);
    feats = res(19).x;
end
