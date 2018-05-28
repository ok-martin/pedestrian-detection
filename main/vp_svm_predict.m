% setup the program variables
run vp_vars.m;

% prepare the CNN model
net = vp_prep_detection_model(prog.files.outMat, 2, prog.net.drop6, prog.net.drop7);

% prep the SVM model
svm_model = loadCompactModel('svm_model.mat');

% get the image
inputDir = '../output/test-images/obj3.jpg';
im = imread(inputDir); %imdb.images.data(:,:,:,2)

tic
ff(1,:) = vp_svm_extract_cnn_features(im, net);
[label,score] = predict(svm_model, ff);
toc