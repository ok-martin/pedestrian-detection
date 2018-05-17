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
person_w = 101;
person_h = 264;

rect_count = 0;
rect = zeros(100,4);
% crop = zeros(person_w,person_h);
fprintf('%d %d\n',width, height);
%im2col

tic
for x=1:steps:(width-person_w)
    for y=1:steps:(height-person_h)
        
        %fprintf('(%d %d), (%d %d)\n',x, y, x+person_w, y+person_h);
        
        crop = im(y:(y+person_h), x:(x+person_w),:,:);
        
        img = crop;
        
        % make sure it is up to CNNs standard
        img_ = single(img); % 255  range
        img_ = imresize(img_, net.meta.normalization.imageSize(1:2)); 
        img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage); % cnn_mean vp_mean
    
        
        
        % imshow(crop);
        
%         img_ = crop;
%         img_ = im2single(img_);
%         img_ = imresize(img_, net.meta.normalization.imageSize(1:2));
%         img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage);
%         
        
        %img_ = bsxfun(@minus, img_, imdb.meta.vp_mean);
        %img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage);
        
        %im_ = im_ - net.meta.normalization.averageImage;
        %imdb.meta.dataMean;
        %net.meta.normalization.averageImage ;
        
        res = vl_simplenn(net, img_);
        scores = squeeze(gather(res(end).x));
        [bestScore, best] = max(scores);
        if scores(1) > 0.7555 %&& strcmp(net.meta.classes{best}, 'people') == 1
            rect_count = rect_count + 1;
            rect(rect_count, :) = [x, y, person_w, person_h];
            disp(bestScore);
        end
    end
end
toc

figure;
imshow(im);
if(rect_count > 0)
    for i=1:rect_count
        rectangle('Position',rect(i, :), 'LineWidth',2, 'EdgeColor', 'red');
    end
end

% im_ = im2single(im) ; % note: 255 range
% im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
% 
% im_ = im_ - net.meta.normalization.averageImage ;
% 
% net.layers{end}.type = 'softmax';
% 
% % Run the CNN.
% res = vl_simplenn(net, im_) ;






%[outimg, bbox, score, probmap] = WinDetect(net, inputDir);
% ConvnetDetect(net, inputDir);

function ConvnetDetect(net, inputDir)
    % Obtain and preprocess an image.
    im = imread(inputDir) ;
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;
    
    % !!!!!!!!!!!!!! Change last layer to softmax
    net.layers{end}.type = 'softmax';
    
    % Run the CNN.
    res = vl_simplenn(net, im_) ;

    % Show the classification result.
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    figure(1) ; clf ; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
       net.meta.classes.description{best}, best, bestScore)) ;
end

function [outimg, bbox, score, probmap] = WinDetect(net, inputDir)
    % Test window_detector

    imdb = load('imdb.mat');

    % Change last layer to softmax
    net.layers{end}.type = 'softmax';

    % Ensure net stores imdb mean
    net.meta.dataMean = imdb.meta.dataMean;

    img = imread(inputDir);
    img = im2single(img);

    stride = 10;
    thresh = 0.5;
    nms = 1;
    windowsize = [102, 265];
    [outimg, bbox, score, probmap] = window_detector(img, net, stride, thresh, windowsize, nms);

end



