% fast rcnn example!!!!!!!!!!

% setup the program variables
run vp_vars.m;

% directories
inputDir = '../output/test-images/obj4.jpg';
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
height = size(im,1);
width = size(im,2);
steps = 30;
win_w = 101;
win_h = 264;

detections = 0;
det_bboxes = zeros(100,4);
det_scores = zeros(100, 1);

nonmax_treshold = 0.5;

tic
for x=1:steps:(width-win_w)
    for y=1:steps:(height-win_h)
        
        % crop
        crop = im(y:(y+win_h), x:(x+win_w),:,:);
        img = crop;
        
        % make sure it is up to CNNs standard
        img_ = single(img); % 255  range
        img_ = imresize(img_, net.meta.normalization.imageSize(1:2)); 
        img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage); % cnn_mean vp_mean
    
        % extract CNN features
        res = vl_simplenn(net, img_);
        % extract scores
        scores = squeeze(gather(res(end).x));
        % select best scores
        [bestScore, best] = max(scores);
        if scores(1) > 0.7555 %&& strcmp(net.meta.classes{best}, 'people') == 1
            detections = detections + 1;
            det_bboxes(detections, :) = [x, y, win_w, win_h];
            det_scores(detections) = bestScore;
            disp(bestScore);
        end
    end
end

% remove zero rows
det_bboxes = det_bboxes(1:detections,:);
det_scores = det_scores(1:detections,:);

% perform non-max supression
[det_bboxes, det_scores] = vp_nonmax_suppression(win_w, win_h, detections, det_bboxes, det_scores, nonmax_treshold);
%[det_bboxes, det_scores] = selectStrongestBbox(det_bboxes, det_scores, 'OverlapThreshold', nonmax_treshold);

toc

figure;
im = insertObjectAnnotation(im2,'rectangle',det_bboxes,cellstr(num2str(det_scores*100)),'Color','green','TextBoxOpacity',0.99,'FontSize',12,'LineWidth',2);
imshow(im);



