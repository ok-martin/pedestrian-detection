% prog.imgSize = [224 224];

% file structure
% Data to be pre-processed
prog.predata.pnn = fullfile('..', 'data', 'data-PennFudanPed', '');

% MatConvNet & Vision Toolboxes
prog.files.matconvnet = fullfile('libs', 'matconvnet', 'matlab', 'vl_setupnn.m');
prog.files.vlfelat = fullfile('libs', 'vlfeat', 'toolbox', 'vl_setup.m');

% Learning Phase: Image Data
prog.files.inImg = fullfile('..', 'input', 'images');
prog.files.inPeople = fullfile(prog.files.inImg, 'people');
prog.files.inNotppl = fullfile(prog.files.inImg, 'not-people');

% Learning Phase: Image Database
prog.files.inImgDb = fullfile('imdb.mat');

% Learning Phase: The Pre-trained Model
prog.net.inMat = fullfile('..', 'input', 'mat', 'imagenet-vgg-m.mat');
prog.net.drop6 = 16;
prog.net.drop7 = 19;

% imagenet-matconvnet-vgg-f
% imagenet-vgg-verydeep-16
% imagenet-vgg-m.mat

% Output data
prog.files.outMat = fullfile('..','output', 'mat', 'genius2.mat');
prog.files.outEpch = fullfile('..', 'output', 'mat', 'epochs');

% info about input data
prog.cnn.imgSize = [224, 224, 3];
prog.img.imgSize = [102, 264, 3];

% training parameters
%prog.trainOpts.errorFunction = 'binary';
prog.trainOpts.learningRate = 0.0001;
prog.trainOpts.batchSize = 50;%50
prog.trainOpts.numEpochs = 12;%12
prog.trainOpts.continue = false;
prog.trainOpts.expDir = prog.files.outEpch;
prog.trainOpts.gpus = [];
