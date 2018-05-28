% setup the program variables
run vp_vars.m;
addpath('regions');
addpath('regions/Dependencies');

% directories
inputDir = '../output/test-images/obj1.png';
model = '../output/mat/genius2.mat'; %'matconvnet/imagenet-vgg-f.mat';

im = imread(inputDir);
im = imresize(im, [480, 640]);

% setup MatConvNet and vlfeat
run(prog.files.matconvnet);
run(prog.files.vlfeat);

% Prepare the model
net = vp_prep_detection_model(prog.files.outMat, 1, prog.net.drop6, prog.net.drop7);

% maximum box overlap
nonmax_treshold = 0.48;

% pre-allocate space
detections = 0;
det_bboxes = zeros(500, 4);
det_scores = zeros(500, 1);


% colour types afect
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1};

% similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% controls size of segments of initial segmentation.
k = 1100; 
minSize = 900; % default = k;
sigma = 0.7; % default = 0.8.



tic
% Perform Selective Search
[boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
boxes = BoxRemoveDuplicates(boxes);
%boxes = selectStrongestBbox(boxes, zeros(size(boxes, 1), 1), 'OverlapThreshold', 0.5);

% x, y, w, h,
boxes2 = [boxes(:,1), boxes(:,2), boxes(:,3)-boxes(:,1), boxes(:,4)-boxes(:,2)];
boxes2(boxes2(:,3) > 306, :) = [];
boxes2(boxes2(:,4) > 528, :) = [];
boxes2(boxes2(:,3) < 51, :) = [];
boxes2(boxes2(:,4) < 132, :) = [];
boxes3 = boxes2(nms(boxes2, 0.45), :);


for index=1:size(boxes2, 1)
    [det_bboxes, det_scores, detections] = slide_window( ... 
        im, net, boxes2(index, :), det_bboxes, det_scores, detections);
end
% remove zero rows
det_bboxes = det_bboxes(1:detections,:);
det_scores = det_scores(1:detections,:);

% perform non-max supression
%[det_bboxes, det_scores] = vp_nonmax_suppression(win_x, win_y, detections, det_bboxes, det_scores, nonmax_treshold);
[det_bboxes, det_scores] = selectStrongestBbox(det_bboxes, det_scores, 'OverlapThreshold', 1-nonmax_treshold);
%det_bboxes = nms(det_bboxes, 1-nonmax_treshold);
toc

figure;
im = insertObjectAnnotation(im,'rectangle',det_bboxes,cellstr(num2str(det_scores*100)),'Color','green','TextBoxOpacity',0.99,'FontSize',12,'LineWidth',2);
imshow(im);






function [det_bboxes, det_scores, detections] = slide_window( ... 
im, net, propsed_box, det_bboxes, det_scores, detections)

x1 = propsed_box(1);
x2 = propsed_box(1)+propsed_box(3); 
y1 = propsed_box(2);
y2 = propsed_box(2)+propsed_box(4);
im = im(x1:x2,y1:y2,:);
scores = cnn_detect(im, net);
if scores(1) > 0.7
    %figure;imshow(im);
    detections = detections + 1;
    det_bboxes(detections, :) = [x1, y1, propsed_box(3), propsed_box(4)];
    det_scores(detections) = scores(1);
end

% im = im(propsed_box(1):propsed_box(3), propsed_box(2):propsed_box(4));
% if scores(1) > 0.55
%     im_height = size(im,1);
%     im_width = size(im,2);
%     % window size
%     win_x = 102; %102
%     win_y = 264; %264
%     % stride size
%     stride_x = win_x/2; 
%     stride_y = win_y/2;
%     % keep track of the position within the image
%     window = -1;
%     % max number of rows
%     max_ys = floor((im_height - win_y+stride_y)/ stride_y);
%     % max number of columns
%     max_xs = floor((im_width -win_x+stride_x) / stride_x);
% 
%     % while there is still more windows
%     while window > -2
%         % which position in the image are we up to
%         window = window + 1;
% 
%         % get current row
%         row = floor(window / (max_xs));
% 
%         % if within the image
%         if (row < max_ys)
% 
%             % get x, y coordinates in the image
%             x = (mod(window, max_xs)*stride_x);
%             y = (row * stride_y);
% 
%             scores = cnn_detect(im(y+1:(y+win_y),x+1:(x+win_x),:), net);
%             if scores(1) > 0.7
%                 % mark detection
%                 detections = detections + 1;
%                 det_bboxes(detections, :) = [propsed_box(1)+x, propsed_box(2)+y, win_x, win_y];
%                 det_scores(detections) = scores(1);
% 
%                 % decrease the stride
% 
%             else
%                 % increase the stride, skip a window
%                 window = window + 1;
%             end
%         else
%             % out of bounds, completed the window
%             fprintf('windows: %d\n', window);
%             window = -2;
%         end
%     end
% end
end




% Show boxes
% ShowRectsWithinImage(boxes, 5, 5, im);
% 
% % Show blobs which result from first similarity function
% hBlobs = RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{1}(end));
% ShowBlobs(hBlobs, 1, 1, im);
% figure;
% im = insertObjectAnnotation(im,'rectangle',boxes2, '', 'LineWidth', 2);
% imshow(im);

function scores = cnn_detect(im, net)    
    % make sure it is up to CNNs standard
    img_ = single(im); % 255  range
    img_ = imresize(img_, net.meta.normalization.imageSize(1:2)); 
    img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage); % cnn_mean vp_mean

    % extract CNN features
    res = vl_simplenn(net, img_);
    % extract scores
    scores = squeeze(gather(res(end).x));
    % select best scores
    % [bestScore, best] = max(scores);
end