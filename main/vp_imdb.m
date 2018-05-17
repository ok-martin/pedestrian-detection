function vp_imdb()
    % setup the program variables
    run vp_vars.m;
    img_size = prog.cnn.imgSize;

    % The sets, and number of samples per label in each set
    sets = {'train', 'val'};
    categories = {'people', 'background'};

    category = 0;
    
    % percantage of data to be train and vali
    trai_p = 0.80;
    vali_p = 0.20;

    % folder structure
    people_path = prog.files.inPeople;
    notpeople_path = prog.files.inNotppl;

    % get the images
    dir_ppl = dir(people_path);
    dir_npl = dir(notpeople_path);

    % remove the folder structure (just want the images)
    dir_ppl(1:2) = [];
    dir_npl(1:2) = [];

    num_people = size(dir_ppl,1);
    num_notppl = size(dir_npl,1);
    nums = {num_people, num_notppl};

    % store the indices of images in the directory
    dataInd = [];
    dataInd = DistributeData(dataInd, num_people, trai_p, vali_p);
    dataInd = DistributeData(dataInd, num_notppl, trai_p, vali_p);
    % store folders
    data(1).path = people_path;
    data(1).dir = dir_ppl;
    data(2).path = notpeople_path;
    data(2).dir = dir_npl;


    % STORE TOGETHER
    if category == 0
        save_file = 'imdb.mat';

        % pre-alocate space 'single'
        total_samples = num_people + num_notppl;
%         images = zeros(img_size(1), img_size(2), img_size(3), total_samples);
%         labels = zeros(total_samples, 1);
%         set = ones(total_samples, 1);
        images = [];
        labels = [];
        set = [];
        sample = 1 ;
        
        % iterate through sets
        for s = 1:size(sets, 2)
            % iterate through labels / categories
            for l = 1:size(categories, 2)
                % fprintf('Adding Images %d\n category: %s\n set: %s\n path: %s\n', size(dataInd{s}{l},2), categories{l}, sets{s}, data(l).path);
                % add labled images
                [images, labels, set, sample] = AddCategoryImages(... 
                    s, l, data(l).path, data(l).dir, dataInd{l}{s}, ... 
                    images, labels, set, sample);
            end
        end
    else
        l = category;
        save_file = ['imdb-' categories{1,l} '.mat'];
        
        % pre-alocate space
        total_samples = nums{1, l};
        images = zeros(img_size(1), img_size(2), img_size(3), total_samples);
        labels = zeros(total_samples, 1);
        set = ones(total_samples, 1);%'single'
        sample = 1 ;
        
        % STORE SEPERATE
        for s = 1:size(sets, 2)
            % add labled images
            [images, labels, set, sample] = AddCategoryImages(... 
                s, l, data(l).path, data(l).dir, dataInd{l}{s}, ... 
                images, labels, set, sample);
        end
    end
    
    % show some random example images
    figure(2) ;
    x = randperm(total_samples, 10);
    montage(images(:,:,:,x)/256);
    disp(labels(x));

    % Remove mean over whole dataset
    % images = bsxfun(@minus, images, mean(images, 4));

    % Store results in the imdb struct
    imdb.images.data = images;
    imdb.images.labels = labels;
    imdb.images.set = set;
    imdb.meta.vp_mean = mean(images, 4);
    imdb.meta.sets = sets;
    imdb.meta.classes = categories;

    % save the imdb
    save(save_file, '-struct', 'imdb');
end

function [images, labels, set, sample] = AddCategoryImages(...
    s, l, path, directory, obj_ind, images, labels, set, sample)

    % iterate through samples
    for i = 1:size(obj_ind, 2)
        % get image path
        impath = fullfile(path, directory(obj_ind(i)).name);

        % read the image
        im = imread(impath);

        % store the image, its label and its set information
        images(:,:,:,sample) = single(im); %im2single
        %imshow(images(:,:,:,sample)/256);
        labels(sample) = l;
        set(sample) = s;
        
        sample = sample + 1;
    end
end

function dataInd = DistributeData(dataInd, num_obj, trai_p, vali_p)
    % randomly divide the input data
    [trainInd, valInd] = dividerand(num_obj, trai_p, vali_p, 0);
    dataInd{end+1} = {trainInd, valInd};
end