% setup program variables
run vp_vars.m;
model = '../output/mat/genius2.mat'; %'matconvnet/imagenet-vgg-f.mat';

% -------------------------------------------------------------------------
% setup the cnn
net = vp_detect_model(model, prog.net.drop6, prog.net.drop7);

% load the image database
imdb = load(prog.files.inImgDb);

% -------------------------------------------------------------------------
% extract features
% prealocate space
data_size = size(imdb.images.data, 4);
feats = zeros(data_size, 1, 4096);

% read each image in the database and extract the cnn features
for i=1:data_size
    feats(i,:,:) = vp_extract_cnn_features(imdb.images.data(:,:,:,i), net);

end

% remove the extra x 1 x dimension
feats = squeeze(feats);


% -------------------------------------------------------------------------
% SVM
svmStruct = fitcsvm(feats, imdb.images.labels);
CVSVMModel = crossval(svmStruct);

% save the model
saveCompactModel(svmStruct, 'svm');

















% set_train_size = 0;
% set_test_size = 0;

%     if imdb.images.set(i) == 1
%         set_train_size = set_train_size + 1;
%     elseif imdb.images.set(i) == 2
%         set_test_size = set_test_size + 1;
%     end

% feats_train = zeros(set_train_size, 4096);
% feats_test = zeros(set_test_size, 4096);
% labels_train = zeros(set_train_size);
% labels_test = zeros(set_test_size);
% train_i = 1;
% test_i = 1;
% for i=1:data_size
%     if imdb.images.set(i) == 1
%         feats_train(train_i,:) = feats(i);
%         labels_train(train_i) = imdb.images.labels(i);
%         train_i = train_i + 1;
%     elseif imdb.images.set(i) == 2
%         feats_test(test_i,:) = feats(i);
%         labels_test(test_i) = imdb.images.labels(i);
%         test_i = test_i + 1;
%     end
% end


