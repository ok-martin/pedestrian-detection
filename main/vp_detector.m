% fast rcnn example!!!!!!!!!!

% setup the program variables
run vp_vars.m;

% directories
inputDir = '../output/test-images/obj2.png';
model = '../output/mat/genius2.mat'; %'matconvnet/imagenet-vgg-f.mat';

% setup MatConvNet.
run matconvnet/matlab/vl_setupnn;

% load the image database
imdb = load(prog.files.inImgDb);

% obtain an image.
im = imread(inputDir);

% -------------------------------------------------------------------------
% Prepare the model
net = vp_detect_model(model, prog.net.drop6, prog.net.drop7);

% -------------------------------------------------------------------------

% image
im_height = size(im,1);
im_width = size(im,2);

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

input_image = im;
win_x = 102; 
win_y = 264;
spacingx = win_x/2; 
spacingy = win_y/2;
windows = ((im_width-win_x)/spacingx)*((im_height-win_y)/spacingy);
for index = 1:windows  
    
    maxPerRow = floor((im_width -win_x+spacingx) / spacingx);
    maxRows = floor((im_height - win_y+spacingy)/ spacingy);
    currentRow = floor(index / (maxPerRow));
    
    if (currentRow < maxRows)
        
        cutX = (mod(index,maxPerRow)*spacingx);
        cutY = (currentRow * spacingy);
            
        scores = cnn_detect(im(cutY+1:(cutY+win_y),cutX+1:(cutX+win_x),:), net);
        if scores(1) > 0.7 %&& strcmp(net.meta.classes{best}, 'people') == 1
            detections = detections + 1;
            det_bboxes(detections, :) = [cutX, cutY, win_x, win_y];
            det_scores(detections) = scores(1);
            %disp(bestScore);
        end
    else
        disp(index);
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