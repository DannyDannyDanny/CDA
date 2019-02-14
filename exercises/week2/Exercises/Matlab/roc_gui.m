function varargout = roc_gui(varargin)
% ROC_GUI MATLAB code for roc_gui.fig
%      ROC_GUI, by itself, creates a new ROC_GUI or raises the existing
%      singleton*.
%
%      H = ROC_GUI returns the handle to a new ROC_GUI or the handle to
%      the existing singleton*.
%
%      ROC_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ROC_GUI.M with the given input arguments.
%
%      ROC_GUI('Property','Value',...) creates a new ROC_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before roc_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to roc_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help roc_gui

% Last Modified by GUIDE v2.5 04-Sep-2011 15:10:40

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @roc_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @roc_gui_OutputFcn, ...
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

function update_plot(handles)
plot(handles.axes_obs, handles.r1(:,1),handles.r1(:,2),'r+',handles.r2(:,1),handles.r2(:,2),'bx')
h1 = refline(-handles.beta(1)/handles.beta(2), -handles.beta(3)/handles.beta(2));
set(h1, 'Color', 'Green');
h2 = refline(-handles.beta(1)/handles.beta(2), -(handles.beta(3) - handles.value)/handles.beta(2));
set(h2, 'Color', 'Black');
title(handles.axes_obs, 'Observations');
xlabel('x_1');
ylabel('x_2');

function update_confusion(handles)
labels = unique(handles.y);
beta = [handles.beta(1:end - 1) handles.beta(end) - handles.value]';
f = handles.X*beta(1:end-1) + beta(end);
tp = sum(f > 0 & handles.y == labels(2));
tn = sum(f < 0 & handles.y == labels(1));
fp = sum(f > 0 & handles.y == labels(1));
fn = sum(f < 0 & handles.y == labels(2));
set(handles.text_fp,'String',fp);
set(handles.text_fn,'String',fn);
set(handles.text_tp,'String',tp);
set(handles.text_tn,'String',tn);
set(handles.text_sens,'String',sprintf('%.2f', tp/(tp+fn)));
set(handles.text_spec,'String',sprintf('%.2f', tn/(tn+fp)));
set(handles.text_ppv,'String',sprintf('%.2f', tp/(tp+fp)));
set(handles.text_npv,'String',sprintf('%.2f', tn/(tn+fn)));

function [sens spec] = create_roc(handles)
beta = [handles.beta(1:end-1)'*ones(1,handles.n); (handles.beta(end) - handles.dist)'];
f = [handles.X ones(handles.n,1)]*beta;
labels = unique(handles.y);
tp = sum(f > 0 & (handles.y == labels(2))*ones(1,handles.n));
tn = sum(f < 0 & (handles.y == labels(1))*ones(1,handles.n));
fp = sum(f > 0 & (handles.y == labels(1))*ones(1,handles.n));
fn = sum(f < 0 & (handles.y == labels(2))*ones(1,handles.n));
sens = tp./(tp+fn);
spec = tn./(tn+fp);

function update_roc(handles)
plot(handles.axes_roc, 1 - handles.spec, handles.sens, 'k-');
axis(handles.axes_roc, [0 1 0 1]);
xlabel(handles.axes_roc, '1 - specificity');
ylabel(handles.axes_roc, 'sensitivity');
title(handles.axes_roc, 'ROC');

y_hat = handles.beta(end) + handles.X*handles.beta(1:end-1)';
[sens spec] = roc_data(y_hat, handles.y, handles.value);

hold(handles.axes_roc, 'on');
plot(handles.axes_roc, 1-spec,sens,'*')
hold(handles.axes_roc, 'off');

% --- Executes just before roc_gui is made visible.
function roc_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to roc_gui (see VARARGIN)

% Choose default command line output for roc_gui
handles.output = hObject;

% init data
handles.n1 = 20;
handles.n2 = 100;
handles.n = handles.n1 + handles.n2;
handles.mu1 = [2 3];
handles.mu2 = [3 5];
handles.SIGMA1 = [1 1.5; 1.5 3];
handles.SIGMA2 = [2 1; 1 1];
randn('seed',43)
handles.r1 = mvnrnd(handles.mu1,handles.SIGMA1,handles.n1);
handles.r2 = mvnrnd(handles.mu2,handles.SIGMA2,handles.n2);
handles.X = [handles.r1; handles.r2];
handles.y = [ones(handles.n1,1); -ones(handles.n2,1)];
handles.beta = lda(handles.X,handles.y);
handles.dist = sort(handles.X*handles.beta(1:end-1)' + handles.beta(end));
set(handles.slider_cut, 'Min', handles.dist(1));
set(handles.slider_cut, 'Max', handles.dist(end));
set(handles.slider_cut, 'Value', 0);
handles.value = 0;

[handles.sens handles.spec] = create_roc(handles);

% Update handles structure
guidata(hObject, handles);

update_plot(handles);
update_confusion(handles);
update_roc(handles)

% UIWAIT makes roc_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = roc_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider_cut_Callback(hObject, eventdata, handles)
% hObject    handle to slider_cut (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.value = get(hObject,'Value');
guidata(hObject, handles);
update_plot(handles);
update_confusion(handles);
update_roc(handles)

%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider_cut_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_cut (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
