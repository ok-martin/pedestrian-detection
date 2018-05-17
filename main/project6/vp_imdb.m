% 224 x 224 is the size required by the pre-trained CNN because that's how
% it has been trained before. Hence it can only accept those sizes (or
% bigger??)
% 
% resize?
% 
% Apply some filters before CNN training
% then after, to get some quicker & better recognition???
%
% cnn scale free?
%
% instead of replacing the classifier can we combine it all the other
% categoires into not human
%
% don't use the matconvnet -f as they are for fast trying out, use other
% ones. google-deep-net or something one?
%
% shouldn't matter if there is background in the people images submitted 
% because if we submit the non-people backgrounds only then the CNN should 
% learn to distinguish them, right? 
% so hence can apply just squares straight out of data, not deformed

% setup the program variables
run vp_vars.m;

people_path = prog.files.inPeople;
notpeople_path = prog.files.inNotppl;

people_dir = dir(people_path);
notpeople_dir = dir(notpeople_path);

num_people = size(people_dir,1);
num_notpeople = size(notpeople_dir,1);

% training = 70% validation = 15% test = 15%;
trai_p = 0.70;
vali_p = 0.15;
test_p = 0.15;

% divide the input data
[pp_trainInd, pp_valInd, pp_testInd] = dividerand(num_people, trai_p, vali_p, test_p);
[npp_trainInd, npp_valInd, npp_testInd] = dividerand(num_notpeople, trai_p, vali_p, test_p);

% training data
[trai_imgs, trai_labels] = FormData( pp_trainInd, npp_trainInd, ... 
    people_path, people_dir, notpeople_path, notpeople_dir);

% validation data
[vali_imgs, vali_labels] = FormData( pp_valInd, npp_valInd, ... 
    people_path, people_dir, notpeople_path, notpeople_dir);

% test data
[test_imgs, test_labels] = FormData( pp_testInd, npp_testInd, ... 
    people_path, people_dir, notpeople_path, notpeople_dir);


%% Concatenate
data = [];
data = cat(4, data, trai_imgs, vali_imgs, test_imgs);

labels = [trai_labels, vali_labels, test_labels];
%labels = single(labels);

set = [ones(1,size(trai_imgs,4)), 2*ones(1,size(vali_imgs,4)), 3*ones(1,size(test_imgs,4))];
set = int8(set);


%% Create imdb struct
imdb.images.data = data;
imdb.images.label = labels;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
%cellstr({'train', 'val', 'test'});
%{'train', 'val', 'test'};
imdb.meta.classes = {'people', 'non_people'}; 
% cellstr({'people', 'non_people'}).';
%{'people', 'non_people'}; 
imdb.meta.dataMean = mean(trai_imgs, 4);

% Normalize data
% imdb.images.data = imdb.images.data - repmat(imdb.meta.dataMean,1,1,1,size(imdb.images.data,4));
save imdb.mat -struct imdb

%% Creating data structs

function [data_struct, data_labels] = FormData(pp_Ind, npp_Ind, ... 
    obj_path, obj_dir, notobj_path, notobj_dir)

    % ini the data struct
    data_struct = [];

    % go through every file and add it to the struct
    % image files start at index 3 in the dir struct
    for i = 3:size(pp_Ind')
        % get path to the image
        img_path = fullfile(obj_path, obj_dir(pp_Ind(i)).name);    
        % add the image to the data struct
        data_struct = cat(4, data_struct, GetImg(img_path));
    end

    for i = 3:size(npp_Ind')
        % get path to the image
        img_path = fullfile(notobj_path, notobj_dir(npp_Ind(i)).name);    
        % add the image to the data struct
        data_struct = cat(4, data_struct, GetImg(img_path));
    end

    % Randomize
    [data_struct, data_labels] = Randomise(data_struct, pp_Ind, npp_Ind);
end

% reading of the images
function img = GetImg(path)
    % read the image
    img = imread(path);
    % convert/compress the image
    img = im2single(img);
end

% randomising the people / non people images in a struct
function [data_struct, data_labels] = Randomise(data_struct, object_i, notobject_i)
    indices = randperm(size(data_struct,4));
    data_struct = data_struct(:,:,:,indices);
    data_labels = [ones(1,size(object_i',1)) 2*ones(1,size(notobject_i',1))];
    data_labels = data_labels(indices);
end
