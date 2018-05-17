function imdb = proj6_part1_setup_data()
%code for Computer Vision, Georgia Tech by James Hays

% The train folder has 100 samples of each category
% Test has an arbitrary amount of each category.
SceneJPGsPath = '../../input-images/';
extension = '*.png';

img_w = 64;
img_h = 64;

sets = {'train', 'test'};
categories = {'people'};
cat_size = 1;

num_train_per_category = 100;
num_test_per_category  = 100; %can be up to 110
total_images = cat_size*num_train_per_category + cat_size* num_test_per_category;

image_size = [img_w img_h]; %downsampling data for speed and because it hurts
% accuracy surprisingly little

imdb.images.data   = zeros(image_size(1), image_size(2), 1, total_images, 'single');
imdb.images.labels = zeros(1, total_images, 'single');
imdb.images.set    = zeros(1, total_images, 'uint8');
image_counter = 1;


fprintf('Loading %d train and %d test images from each category\n', ...
          num_train_per_category, num_test_per_category)
fprintf('Each image will be resized to %d by %d\n', image_size(1),image_size(2));

%Read each image and resize it to image_size
for set = 1:length(sets)
    for category = 1:length(categories)
        cur_path = fullfile( SceneJPGsPath, categories{category}, sets{set});
        cur_images = dir( fullfile( cur_path,  extension) );
        
        if(set == 1)
            fprintf('Taking %d out of %d images in %s\n', num_train_per_category, length(cur_images), cur_path);
            cur_images = cur_images(1:num_train_per_category);
        elseif(set == 2)
            fprintf('Taking %d out of %d images in %s\n', num_test_per_category, length(cur_images), cur_path);
            cur_images = cur_images(1:num_test_per_category);
        end

        for i = 1:length(cur_images)

            cur_image = imread(fullfile(cur_path, cur_images(i).name));
            cur_image = single(cur_image);
            if(size(cur_image,3) > 1)
                fprintf('color image found %s\n', fullfile(cur_path, cur_images(i).name));
                cur_image = rgb2gray(cur_image);
            end
            cur_image = imresize(cur_image, image_size);
                       
            % Stack images into a large image_size x 1 x total_images matrix
            % images.data
            imdb.images.data(:,:,1,image_counter) = cur_image;            
            imdb.images.labels(  1,image_counter) = category;
            imdb.images.set(     1,image_counter) = set; %1 for train, 2 for test (val)
            
            image_counter = image_counter + 1;
        end
    end
end

