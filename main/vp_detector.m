% fast rcnn example!!!!!!!!!!

% setup the program variables
run vp_vars.m;

% directories
inputDir = '../output/test-images/obj1.png';
model = '../output/mat/genius2.mat'; %'matconvnet/imagenet-vgg-f.mat';

% setup MatConvNet.
run(prog.files.matconvnet);
run(prog.files.vlfeat);

% load the image database
imdb = load(prog.files.inImgDb);

% obtain an image.
im = imread(inputDir);
%im = imresize(im, 0.4);
% -------------------------------------------------------------------------
% Prepare the model
net = vp_prep_detection_model(prog.files.outMat, 1, prog.net.drop6, prog.net.drop7);

% -------------------------------------------------------------------------
% 640x480 
im_height = size(im,1);
im_width = size(im,2);



% -------------------------------------------------------------------------   
% maximum box overlap
nonmax_treshold = 0.48;

% pre-allocate space
detections = 0;
det_bboxes = zeros(500, 4);
det_scores = zeros(500, 1);

% -------------------------------------------------------------------------
tic

% win_w = floor(im_width/(im_width/102));
% win_h = floor(im_height/(im_height/264));
% step_w = win_w - 20;
% step_h = win_h - 20;
% blocks = struct('img', cell(1, 10), 'coord', cell(1, 10));
% block_count = 0;
% 
% for x=1:step_w:(im_width-win_w)
%     for y=1:step_h:(im_height-win_h)
%         xw = x+win_w;
%         yh = y+win_h;
%         if (yh+step_h) >= im_height
%             yh = im_height;
%         end
%         if (xw+step_w) > im_width
%             xw = im_width;
%         end
%         
%         crop = im(y:yh, x:xw, :);
%         scores = cnn_detect(crop, net);
%         if scores(1) > 0.5
%             block_count = block_count + 1;
%             blocks(block_count).img = crop;
%             blocks(block_count).coord = [x, y, xw, yh];
%         end
%         
%     end
% end
% 
% 
% win_w = 102/2;
% win_h = 264/2;
% steps_w = floor(win_w/2);
% steps_h = floor(win_h/4);
% 
% for count=1:block_count
%     % sliding window
%     block = blocks(count).img;
%     for x=1:steps_w:(size(block,2)-win_w)
%         for y=1:steps_h:(size(block,1)-win_h)
%             scores = cnn_detect(block(y:(y+win_h), x:(x+win_w),:), net);
%             if scores(1) > 0.7 %&& strcmp(net.meta.classes{best}, 'people') == 1
%                 detections = detections + 1;
%                 
%                 det_bboxes(detections, :) = [blocks(count).coord(1)+x, blocks(count).coord(2)+y, win_w, win_h];
%                 det_scores(detections) = scores(1);
%                 %disp(bestScore);
%             end
%         end
%     end
% end
% -------------------------------------------------------------------------
% window size
% win_x = im_width/3; 
% win_y = im_height/3;
% stride_x = floor(win_x*0.7); 
% stride_y = floor(win_y*0.7);
% window = -1;
% max_ys = floor((im_height - win_y+stride_y)/ stride_y);
% max_xs = floor((im_width - win_x+stride_x) / stride_x);
% 
% % while there is still more windows
% while window > -2
%     % which position in the image are we up to
%     window = window + 1;
% 
%     % get current row
%     row = floor(window / (max_xs));
%     
%     % if within the image
%     if (row < max_ys)
%         
%         % get x, y coordinates in the image
%         x = (mod(window, max_xs)*stride_x);
%         y = (row * stride_y);
%         disp(x);
%         disp(y);
%         scores = cnn_detect(im(y+1:(y+win_y),x+1:(x+win_x),:), net);
%         if scores(1) > 0.7
%             figure
%             imshow(im(y+1:(y+win_y),x+1:(x+win_x),:));
%         end
%     else
%         % out of bounds, completed the window
%         disp(window);
%         window = -2;
%     end
% end




% -------------------------------------------------------------------------
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
            
        scores = cnn_detect(im(y+1:(y+win_y),x+1:(x+win_x),:), net);
        if scores(1) > 0.7
            % mark detection
            detections = detections + 1;
            det_bboxes(detections, :) = [x, y, win_x, win_y];
            det_scores(detections) = scores(1);
            
            % decrease the stride

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
%[det_bboxes, det_scores] = selectStrongestBbox(det_bboxes, det_scores, 'OverlapThreshold', 1-nonmax_treshold);

toc

% -------------------------------------------------------------------------
figure;
im = insertObjectAnnotation(im,'rectangle',det_bboxes,cellstr(num2str(det_scores*100)),'Color','green','TextBoxOpacity',0.99,'FontSize',12,'LineWidth',2);
imshow(im);



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