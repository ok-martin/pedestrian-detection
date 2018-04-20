baseDir = 'pas\';
annotDir = [baseDir 'PennFudanPed\Annotation\'];
imgDir = [baseDir 'PennFudanPed\PNGImages\'];

% files = dir(annotDir); files(1:2) = [];
% close all;
% for ii = 1 : length(files)
%     fileName = [annotDir files(ii).name];
%     record = PASreadrecord(fileName);
%     imshow([baseDir record.imgname]); hold on;
%     for jj = 1 : length(record.objects)
%         bbox = record.objects(jj).bbox;
%         bbox(3:4) = bbox(3:4) - bbox(1:2);
%         rectangle('Position', bbox, 'EdgeColor','y','LineWidth',2);
%     end
%     hold off;    
%     pause(0.5);
% end

totalW = 0;
totalH = 0;
totalA = 0;
objectsNO = 0;

% get annotation files
files = dir(annotDir); files(1:2) = [];
close all;

for ii = 1 : length(files)
    fileName = [annotDir files(ii).name];
    record = PASreadrecord(fileName);
    for jj = 1 : length(record.objects)
        bbox = record.objects(jj).bbox;
        width = bbox(3) - bbox(1);
        height = bbox(4) - bbox(2);
        aratio = width/height;
        totalW = totalW + width;
        totalH = totalH + height;
        totalA = totalA + aratio;
        objectsNO = objectsNO + 1;
    end
end

avgW = totalW/objectsNO;
avgH = totalH/objectsNO;
avgA = totalA/objectsNO;
avgR = avgW/avgH;


objectsNO = 0;

% get image files
imgFiles = dir(imgDir);
imgFiles(1:2) = [];
close all;

for ii = 1 : length(files)
    fileName = [annotDir files(ii).name];
    record = PASreadrecord(fileName);
    img = imread([imgDir imgFiles(ii).name]);
    for jj = 1 : length(record.objects)
        bbox = record.objects(jj).bbox;
        
        % crop the image
        x = bbox(1);
        y = bbox(2);
        w = bbox(3) - bbox(1);
        h = bbox(4) - bbox(2);
        person = imcrop(img, [x y w h]);
        
        
        % resize the image
        person = imresize(person, [avgH, avgW]);
        
        % image name
        objectsNO = objectsNO + 1;
        fileNO = sprintf('person%d', objectsNO);
        finalFile = ['cropped\' fileNO '.png'];
        
        % save image
        imwrite(person, finalFile);
    end
end


