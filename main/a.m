inputDir = '../output/test-images/obj3.jpg';
model = '../output/mat/genius2.mat';

% setup the program variables
run vp_vars.m;

% setup MatConvNet.
run matconvnet/matlab/vl_setupnn;

% obtain the trained svm
svm = loadCompactModel('svm.mat');

% obtain an image
im = imread(inputDir);


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
        
        feats = vp_extract_cnn_features(img, net);
        z(1,:) = feats(1,:,:);
        
        label = predict(svm, z);
        if label == 1
            rect_count = rect_count + 1;
            rect(rect_count, :) = [x, y, person_w, person_h];
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
