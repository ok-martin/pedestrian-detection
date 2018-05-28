% fast rcnn example!!!!!!!!!!

% setup the program variables
run vp_vars.m;

% directories
inputDir = '../output/test-images/obj3.jpg';
model = '../output/mat/genius2.mat'; %'matconvnet/imagenet-vgg-f.mat';

% setup MatConvNet.
run matconvnet/matlab/vl_setupnn;

% load the image database
imdb = load(prog.files.inImgDb);

% obtain an image.
im = imread(inputDir);
im = imresize(im,0.5);
% -------------------------------------------------------------------------
% Prepare the model
net = vp_detect_model(model, prog.net.drop6, prog.net.drop7);

% -------------------------------------------------------------------------
height = size(im,1);
width = size(im,2);
steps = 30;
person_w = 40;
person_h = 100;

rect_count = 0;
rect = zeros(100,4);
rect_scores = zeros(100);
% crop = zeros(person_w,person_h);
fprintf('%d %d\n',width, height);

tic
%bwim = rgb2gray(im);
window_matrix = im2col(im, [person_h person_w], 'sliding');
%window_matrix = im2col_sliding(im, [person_h person_w]);
[M,N] = size(window_matrix)
x_limit = floor(width / person_w);
y_limit = floor(height / person_h);
x_counter = 0;
y_counter = 0;
for x = 1:N
    B = window_matrix(:,x);
    %size(B,1)
    %temp_im = col2im(B, [size(B,1), 1], [person_h person_w], 'distinct');
    temp_im = reshape(B,[person_h person_w]);
    
    %make sure it is up to CNNs standard
    img_ = single(temp_im); % 255  range
    img_ = imresize(img_, net.meta.normalization.imageSize(1:2)); 
    img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage); % cnn_mean vp_mean
    
    res = vl_simplenn(net, img_);
    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores);
    if scores(1) > 0.65 %&& strcmp(net.meta.classes{best}, 'people') == 1
        rect_count = rect_count + 1;
        rect(rect_count, :) = [(person_w * x_counter), (person_h * y_counter), person_w, person_h];
        rect_scores(rect_count) = bestScore;
        disp(bestScore);
    end
    x_counter = x_counter + 1;
    if x_counter > x_limit
        x_counter = 0;
        y_counter = y_counter + 1;
    end
end
toc

% tic
% for x=1:steps:(width-person_w)
%     for y=1:steps:(height-person_h)
%         
%         fprintf('(%d %d), (%d %d)\n',x, y, x+person_w, y+person_h);
%         
%         crop = im(y:(y+person_h), x:(x+person_w),:,:);
%         
%         img = crop;
%         
%         make sure it is up to CNNs standard
%         img_ = single(img); % 255  range
%         img_ = imresize(img_, net.meta.normalization.imageSize(1:2)); 
%         img_ = bsxfun(@minus, img_, net.meta.normalization.averageImage); % cnn_mean vp_mean
%     
%         res = vl_simplenn(net, img_);
%         scores = squeeze(gather(res(end).x));
%         [bestScore, best] = max(scores);
%         if scores(1) > 0.65 %&& strcmp(net.meta.classes{best}, 'people') == 1
%             rect_count = rect_count + 1;
%             rect(rect_count, :) = [x, y, person_w, person_h];
%             rect_scores(rect_count) = bestScore;
%             disp(bestScore);
%         end
%     end
% end
% toc

%non-max suppression

% window_area = person_h * person_w;
% if(rect_count > 0)
%     for ii=1:rect_count-1
%         for jj=ii:rect_count
%             x_check = abs(rect(ii,1) - rect(jj,1));
%             y_check = abs(rect(ii,2) - rect(jj,2));
%             if x_check < person_w && y_check < person_h
%                 crossover_area = (person_w - x_check) * (person_h - y_check);
%                 persentage_crossover = crossover_area / window_area;
%                 if persentage_crossover > 0.45
%                     if rect_scores(ii) > rect_scores(jj) && rect_scores(ii) ~= 0 && rect_scores(jj) ~= 0
%                         rect_scores(jj) = 0;
%                     elseif rect_scores(ii) < rect_scores(jj) && rect_scores(ii) ~= 0 && rect_scores(jj) ~= 0
%                         rect_scores(ii) = 0;
%                     end
%                 end
%             end
%         end
%     end
% end
                    

figure;
imshow(im);
if(rect_count > 0)
    for ii=1:rect_count
        %if rect_scores(ii) > 0
            rectangle('Position',rect(ii, :), 'LineWidth',2, 'EdgeColor', 'red');
        %end
    end
end





