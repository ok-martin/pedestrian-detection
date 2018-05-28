Im = imread('../output/test-images/obj2.png');

[featureVector, hogVisualization] = extractHOGFeatures(Im);

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
    % prep svm data structures
    data_size = size(imdb.images.data, 4);
    features = zeros(data_size*3844, 9);
    labels = zeros(data_size*3844, 1);
    
    % prep sliding window
    map_size = 64;
    win_x = 3;
    win_y = 3;
    stride_x = 1;
    stride_y = 1;
    max_ys = floor((map_size - win_y+stride_y)/ stride_y);
    max_xs = floor((map_size -win_x+stride_x) / stride_x);
    win_index = 1;
    window = -1;
    
    % save featuers
    for index = 1:data_size
        
        % extract the features
        im = reshape(vp_svm_extract_cnn_features(imdb.images.data(:,:,:,index), net), [map_size, map_size]);
        
        % get every sliding window feature
        while window > -2
            window = window + 1;
            row = floor(window / (max_xs));
            % if within the image
            if (row < max_ys)
                x = (mod(window, max_xs)*stride_x);
                y = (row * stride_y);
                % get the window of features
                features(win_index, :) = reshape(im(y+1:(y+win_y),x+1:(x+win_x)), [9, 1]);
                % save the label
                labels(win_index, 1) = imdb.images.labels(index);
                win_index = win_index + 1;
            else
                disp(window);
                window = -10;
            end
        end
    end
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
[det_bboxes, det_scores] = selectStrongestBbox(det_bboxes, det_scores, 'OverlapThreshold', 1-nonmax_treshold);


% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
run(prog.files.vlfelat);

% load the image database
im = imread('../output/test-images/obj3.jpg');

im = im2single(im);

imshow(im);

cellSize = 8 ;
hog = vl_hog(im, cellSize, 'verbose');

imhog = vl_hog('render', hog, 'verbose') ;
clf ; imagesc(imhog) ; colormap gray ;

size(hog)


% ------------------------------------------------------------------------
% ------------------------------------------------------------------------


% setup the program variables
run vp_vars.m;

% setup MatConvNet and VlFeatvm
run(prog.files.matconvnet);
run(prog.files.vlfeat);

% read the image
im = imread('../output/test-images/obj1.png');
im_height = size(im,1);
im_width = size(im,2);

% prepare the CNN model
net = vp_prep_detection_model(prog.files.outMat, 2, prog.net.drop6, prog.net.drop7);

% prep the SVM model
svm_model = loadCompactModel('svm_model.mat');

% extract the feature map
map_size = 64;
features = vp_svm_extract_cnn_features(im, net);
features = reshape(features, [map_size, map_size]);

% window size
win_x = 3;
win_y = 3;
% stride size
stride_x = 1;
stride_y = 1;
% keep track of the position within the image
window = -1;
% max number of rows
max_ys = floor((map_size - win_y+stride_y)/ stride_y);
% max number of columns
max_xs = floor((map_size -win_x+stride_x) / stride_x);

% while there is still more windows
while window > -2
    % which position in the image are we up to
    window = window + 1;

    % get current row
    row = floor(window / (max_xs));
    
    % if within the image
    if (row < max_ys)
        % get x, y coordinates in the image
        x = (mod(window, max_xs)*stride_x);
        y = (row * stride_y);
        z = reshape(features(y+1:(y+win_y),x+1:(x+win_x)), [9, 1]);
        [label,score] = predict(svm_model, z);
    else
        disp(window);
        window = -2;
    end
end

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

H = im_height;
W = im_width;
POOLY = 264;
POOLX = 102;
STRIDEY = 10;
STRIDEX = 10;
PADTOP = 0;
PADBOTTOM = 0;
PADLEFT = 0;
PADRIGHT = 0;

z = vl_nnpool(im2single(im), [POOLY, POOLX]);

YH = floor((H + (PADTOP+PADBOTTOM) - POOLY)/STRIDEY) + 1;
YW = floor((W + (PADLEFT+PADRIGHT) - POOLX)/STRIDEX) + 1;

z(YH, YW, :)

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------