% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%
addpath('Dependencies');

fprintf('Demo of how to run the code for:\n');
fprintf('   J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders\n');
fprintf('   Segmentation as Selective Search for Object Recognition\n');
fprintf('   IJCV 2013\n\n');

% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

if(~exist('mexCountWordsIndex'))
    mex Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made.\n');
%     fprintf('   
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 1000; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.7;

% As an example, use a single image
images = {'000016.jpg'};
im = imread(images{1});
im = imresize(im, [480, 640]);

tic
% Perform Selective Search
[boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
boxes = BoxRemoveDuplicates(boxes);
%boxes = selectStrongestBbox(boxes, zeros(size(boxes, 1), 1), 'OverlapThreshold', 0.5);

% x, y, w, h,
boxes2 = [boxes(:,2), boxes(:,1), boxes(:,4)-boxes(:,2), boxes(:,3)-boxes(:,1)];
boxes2(boxes2(:,3) > 306, :) = [];
boxes2(boxes2(:,4) > 528, :) = [];
boxes2(boxes2(:,3) < 51, :) = [];
boxes2(boxes2(:,4) < 132, :) = [];
boxes2 = boxes2(nms(boxes2, 0.5), :);


x1 = boxes2(:,1); x2 = x1 + boxes2(:,3);
y1 = boxes2(:,2); y2 = y1 + boxes2(:,4);

for aaa = 1:size(boxes2, 1)
    for bbb = aaa:size(boxes2, 1)
        if x1(aaa) > x1(bbb) && x1(aaa) < x2(bbb)
            boxes2(aaa,1) = x2(bbb);
            if y1(aaa) > y1(bbb)
                boxes2(bbb,2) = boxes2(aaa,2);
            elseif y1(aaa) < y1(bbb)
                boxes2(bbb,4) = boxes2(aaa,4);
            end
        elseif x2(aaa) > x1(bbb) && x2(aaa) < x2(bbb)
            boxes2(aaa,1) = x1(bbb);
            if y1(aaa) > y1(bbb)
                boxes2(bbb,2) = boxes2(aaa,2);
            elseif y1(aaa) < y1(bbb)
                boxes2(bbb,4) = boxes2(aaa,4);
            end
        end
    end
end

toc

% Show boxes
ShowRectsWithinImage(boxes, 5, 5, im);

% Show blobs which result from first similarity function
% hBlobs = RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{1});
% ShowBlobs(hBlobs, 5, 5, im);
figure;
im = insertObjectAnnotation(im,'rectangle',boxes2, '', 'LineWidth', 2);
imshow(im);