% Based on the script from the project site.
% source:
% http://teaching.csse.uwa.edu.au/units/CITS4402/labs/project/project2018.html
% setup the program variables
run vp_vars.m;

% input / output directories
baseDir = prog.predata.pnn;
annotDir = fullfile(baseDir, 'Annotation');
imgDir = fullfile(baseDir, 'PNGImages');
outputDirPeople = prog.files.inPeople;
outputDirNotPpl = prog.files.inNotppl;
objectName = 'object';
objectExtension = '.png';

% create the folder structure
mkdir(outputDirPeople);
mkdir(outputDirNotPpl);

% ouput image dimensions
img_size = [224, 224];

% get the pas functions
addpath pas-data

% used for calculations of the average
totalW = 0;
totalH = 0;
totalA = 0;
objectsNO = 0;

% get annotation files
files = dir(annotDir);
files(1:2) = [];
close all;

% go through every file
for ii = 1 : length(files)
    % open the annotation file
    fileName = fullfile(annotDir, files(ii).name);
    % get the info about images to get the objects (humans)
    record = PASreadrecord(fileName);
    
    % go through each object
    for jj = 1 : length(record.objects)
        % get the bounding box coord for the object
        bbox = record.objects(jj).bbox;
        
        % calculate the box
        width = bbox(3) - bbox(1);
        height = bbox(4) - bbox(2);
        
        % add the values for the total
        aratio = width/height;
        totalW = totalW + width;
        totalH = totalH + height;
        totalA = totalA + aratio;
        objectsNO = objectsNO + 1;
    end
end

% calculate the avarages
avgW = totalW/objectsNO;
avgH = totalH/objectsNO;
avgA = totalA/objectsNO;
avgR = avgW/avgH;

objectsNO = 0;
non_objectsNO = 0;

% get image files
imgFiles = dir(imgDir);
imgFiles(1:2) = [];
close all;

% go through every image
for ii = 1 : length(files)
    % open the annotation file
    fileName = fullfile(annotDir, files(ii).name);
    % get the info about images to get the objects (humans)
    record = PASreadrecord(fileName);
    % get the images
    img = imread(fullfile(imgDir, imgFiles(ii).name));
    
    % remeber people boxes
    boxes = [];
    boxes_count = 0;
    
    % go through every object
    for jj = 1 : length(record.objects)
        % extract the object
        bbox = record.objects(jj).bbox;
        % remember the box
        boxes_count = boxes_count + 1;
        boxes(boxes_count).box = bbox;
        
        % crop the image
        x = bbox(1);
        y = bbox(2);
        w = bbox(3) - bbox(1);
        h = bbox(4) - bbox(2);
        object = imcrop(img, [x y w h]);
        
        % resize the image
        %object = imresize(object, [avgH, avgW]);
        object = imresize(object, img_size);
        
        objectsNO = SaveImg(object, objectsNO, objectName, outputDirPeople, objectExtension);
        %objectsNO = SaveImg(flip(object, 2), objectsNO, objectName, outputDirPeople, objectExtension);
    end
    
    startX = 1; endX = avgW;
    startY = 1; endY = avgH;

    non_objectsNO = CropBackground(startX, startY, endX, endY, boxes, boxes_count, non_objectsNO, img, objectName, outputDirNotPpl, objectExtension);

    startX = size(img, 2)-avgW-1; endX = startX+avgW;
    startY = size(img, 1)-avgH-1; endY = startY+avgH;

    non_objectsNO = CropBackground(startX, startY, endX, endY, boxes, boxes_count, non_objectsNO, img, objectName, outputDirNotPpl, objectExtension);
    
end

rmpath pas-data


function non_objectsNO = CropBackground(startX, startY, endX, endY, boxes, boxes_count, non_objectsNO, img, objectName, outputDirNotPpl, objectExtension)
    all_g = true;
    
    for b = 1:boxes_count
        if (endX < boxes(b).box(1) || endY < boxes(b).box(2)) ...
            || (startX > boxes(b).box(3) || startY > boxes(b).box(4))
                all_g = true;
        else
            all_g = false;
        end
    end
    
    if all_g == true
        
       % fprintf("%d %d %d %d\n", startX, startY, (endX-startX), (endY-startY));
        
        non_object = imcrop(img, [startX startY (endX-startX) (endY-startY)]);
        non_object = imresize(non_object, [224, 224]);
        
        non_objectsNO = SaveImg(non_object, non_objectsNO, objectName, outputDirNotPpl, objectExtension);
        %non_objectsNO = SaveImg(flip(non_object, 2), non_objectsNO, objectName, outputDirNotPpl, objectExtension);
    end
end


function objectsNO = SaveImg(object, objectsNO, objectName, outputDirPeople, objectExtension)
    % image name
    objectsNO = objectsNO + 1;
    fileNO = sprintf('%s%d', objectName, objectsNO);
    finalFile = fullfile(outputDirPeople, [fileNO objectExtension]);

    % save the image
    imwrite(object, finalFile);
end