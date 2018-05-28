% setup the program variables
run vp_vars.m;

% ------------------------------------------------------------------------
% get the image
inputDir = '../output/test-images/obj1.png';
im = imread(inputDir); %imdb.images.data(:,:,:,2)

im_height = size(im,1);
im_width = size(im,2);

% ------------------------------------------------------------------------
% prepare the CNN model
net = vp_prep_detection_model(prog.files.outMat, 2, prog.net.drop6, prog.net.drop7);
% prep the SVM model
svm_model = loadCompactModel('svm_model.mat');

% ------------------------------------------------------------------------
% maximum box overlap
nonmax_treshold = 0.48;

% pre-allocate space
detections = 0;
det_bboxes = zeros(500, 4);
det_scores = zeros(500, 1);
feats = zeros(1, 4096);

% ------------------------------------------------------------------------
% window size
win_x = 102; %102
win_y = 264; %264
% stride size
stride_x = win_x/2; 
stride_y = win_y/2;
% keep track of the position within the image
window = -1;
% max number of rows
max_ys = floor((im_height - win_y+stride_y)/ stride_y);
% max number of columns
max_xs = floor((im_width -win_x+stride_x) / stride_x);

% ------------------------------------------------------------------------
tic
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

        feats(1,:) = vp_svm_extract_cnn_features(im(y+1:(y+win_y),x+1:(x+win_x),:), net);
        [label, score] = predict(svm_model, feats);
        if label == 1
            % mark detection
            detections = detections + 1;
            det_bboxes(detections, :) = [x, y, win_x, win_y];
            det_scores(detections) = max(score);
        else
            % increase the stride, skip a window
            window = window + 1;
        end
    else
        % out of bounds, completed the window
        fprintf('windows: %d\n', window);
        window = -2;
    end
end


% remove zero rows
det_bboxes = det_bboxes(1:detections,:);
det_scores = det_scores(1:detections,:);

% perform non-max supression
[det_bboxes, det_scores] = vp_nonmax_suppression(win_x, win_y, detections, det_bboxes, det_scores, nonmax_treshold);

toc

% -------------------------------------------------------------------------
figure;
im = insertObjectAnnotation(im,'rectangle',det_bboxes,cellstr(num2str(det_scores*100)),'Color','green','TextBoxOpacity',0.99,'FontSize',12,'LineWidth',2);
imshow(im);
