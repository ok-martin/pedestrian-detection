function vp_svm_train()
% setup the program variables
run vp_vars.m;

% setup MatConvNet and VlFeatvm
run(prog.files.matconvnet);
run(prog.files.vlfeat);

% load the image database
imdb = load(prog.files.inImgDb);

% prepare the CNN model
net = vp_prep_detection_model(prog.files.outMat, 2, prog.net.drop6, prog.net.drop7);

% prep svm data structures
data_size = size(imdb.images.data, 4);
% featuers = zeros(4096, data_size);
% labels = zeros(1, data_size);
features = zeros(data_size, 4096);
labels = zeros(data_size, 1);
% --------------------------------------------------------------------
%                                                          CNN Featues
% --------------------------------------------------------------------

% extract and save CNN features
for index = 1:data_size
    %features(:, index) = vp_svm_extract_cnn_features(imdb.images.data(:,:,:,index), net);
    %labels(1, index) = imdb.images.labels(index);
    features(index, :) = vp_svm_extract_cnn_features(imdb.images.data(:,:,:,index), net);
    labels(index, 1) = imdb.images.labels(index);
end

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------
% % regularization parameter
% lambda = 0.01;
% % max number of iterations
% maxIter = 1000;
% % train
% [w, b, info] = vl_svmtrain(featuers, labels, lambda, 'MaxNumIterations', maxIter);

% vision toolbox
svm_struct = fitcsvm(features, labels);

% save the svm
saveCompactModel(svm_struct, 'svm_model');

end
