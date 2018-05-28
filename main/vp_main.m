function varargout = vp_main(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @vp_main_OpeningFcn, ...
                   'gui_OutputFcn',  @vp_main_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before vp_main is made visible.
function vp_main_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for vp_main
handles.output = hObject;

% -----------------------------------------------------------------------
% setup the program variables
run('vp_vars.m');
% -----------------------------------------------------------------------
% prepare the model
handles.net = vp_prep_detection_model(prog.files.outMat, 1, prog.net.drop6, prog.net.drop7);
% prep the SVM model
handles.svm_model = loadCompactModel('svm_model.mat');

% -----------------------------------------------------------------------
% -----------------------------------------------------------------------
% Update handles structure
guidata(hObject, handles);

function varargout = vp_main_OutputFcn(hObject, eventdata, handles) 
    varargout{1} = handles.output;


function button_load_image_Callback(hObject, eventdata, handles)
    try
        % get the path to this file
        currentDir = pwd;

        % check if the path is not empty
        if isempty(currentDir) == 0
            % set the picker to all files
            default = fullfile(pwd, '*.*');

            % open file manager to select an image
            [filename, pathname] = uigetfile(default);
            if isequal(filename, 0)
                DisplayMessage('Image selection canceled.');
            else
                % get the image path & image matrix
                imgPath = fullfile(pathname, filename);

                % load the image and save the data
                handles = LoadImg(hObject, handles, imgPath, filename);
                % update the data
                guidata(hObject, handles);
            end
        end
    catch
        DisplayMessage('Image selection failed.');
    end

function button_detect_image_Callback(hObject, eventdata, handles)
    try
        % the detection function
        [im, time_taken] = vp_detection(handles.net, handles.ogImg);
        % update the image
        DisplayImg(handles.presentation_axes, im, 'People!');
        % display the performance
        performance = sprintf('%0.3f s', time_taken);
        set(handles.detection_time_text, 'String', performance);
        guidata(hObject, handles);
    catch
        DisplayMessage('Error, load an image first');
    end

function button_video_detection_Callback(hObject, eventdata, handles)

function presentation_axes_CreateFcn(hObject, eventdata, handles)
    ax = gca;
    ax.Visible = 'off';

    
function handles = LoadImg(hObject, handles, imgPath, title)
    try
        % read the image
        img = imread(imgPath);
        % save the data to the program
        handles.imgPath = imgPath;
        handles.ogImg = img;
        handles.edImg = img;
        % get the axis for original (og) image display & show the selected image
        DisplayImg(handles.presentation_axes, img, title);
    catch
        % display the error
        DisplayMessage(['Error reading: ', imgPath]);
    end 

% Display the image (img) on the specified axis with the title (text)    
function DisplayImg(axis, img, text)
    try
        % get the axis for the edited image display
        axes(axis);
        % clear the current image
        cla;
        % show the selected image
        imshow(img);
        % show the title
        title(text);
    catch
        DisplayMessage(['Error when displaying the image: ', text]);
    end

% Choose the format of the output messages
function DisplayMessage(text)
    %disp(text);
    msgbox(text, '!');
