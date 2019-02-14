%function GSdendrogram
%
%GSdendrogram builds a dendrogram using the statistical toolbox's
%dendrogram function. However, in case of spectral numbers, labels
%are used to indicate different groups of samples. There is a new labelling
%method as well, in which coloured bars below the dendrogram are plot,
%indicating different groups of spectra, as well as the amount of spectra
%in the different groups.
%
%Syntax:
%    Info = GSDendrogram (Z, Labels, Colors)
%
%With:
%    Z: The Z-matrix as normal input for dendrogram (see help dendrogram)
%    Labels: Labels of the different spectra
%    Colors: a optional colour matrix, indicating the individual colours
%        for the different sample groups
%    Info: an matrix containing dendrogram info about the spectra

%C 2005-2006, Kris De Gussem, Raman Spectroscopy Research group, Laboratory
%of Analytical Chemistry, Ghent University
%
%This code is free software; you may redistribute it and/or modify it under
%the terms of the GNU General Public License as published by the Free
%Software Foundation; either version 2.1, or (at your option) any later
%version.
%
%This is distributed in the hope that it will be useful, but without any
%warranty; without even the implied warranty of merchantability or fitness
%for a particular purpose. See the GNU General Public License for more
%details.
%
%You should have received a copy of the GNU General Public License with
%this software. If not, a copy of the GNU General Public License is
%available as /usr/share/common-licenses/GPL in the Debian GNU/Linux
%distribution or on the World Wide Web at the GNU web site. You can also
%write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
%Boston, MA 02111-1307, USA.

% changed by OWI for teaching in 02581

function Info = GSDendrogram (Z, Labels, colors , N_leafs ); % changed here to allow user defined N_leafs
addpath('additional')
switch nargin
    case 2
        colors = [];
        N_leafs = 30 ;
    case 3
        N_leafs = 30 ;
    case 4
    otherwise
        error ('Wrong number of input parameters...');
end

NewLabellingMethod = 1 ;
% NewLabellingMethod = questdlg ('New labelling method?', 'GSDendrogram', 'Yes', 'No', 'Yes');
% switch NewLabellingMethod
%     case 'Yes'
%         NewLabellingMethod = 1;
%     case 'No'
%         NewLabellingMethod = 0;
% end

%figure;
[H, T] = dendrogram( Z , N_leafs );  % changed here to allow user defined N_leafs
%xlabel ('samples');
ylabel ('distance');
set (gcf, 'PaperOrientation', 'portrait');
set (gcf, 'PaperPositionMode', 'auto');
set (gcf, 'Paperposition', [0 0 20.984 29.6774])

%if a RGB colour matrix isn't given: make one with a sufficient amount of
%colours
if isempty (colors)
    if NewLabellingMethod
        colors = [0 0 0; 0 0.5 0; 1 0 0; 0 0.75 0.75; 0.75 0 0.75; 0.75 0.75 0; 0.25 0.25 0.25];
        %black blue, green, red, lila, orange, darkgrey
        colors = [colors; 1 0 1; 1 1 0 ];
        %magenta, yellow, black
        colors = [colors; [151 176 133]./255 ; [253 251 144]./255;];
        %chartreuse, Light Yellow
        colors = [colors; [163 177 7]./255];  %PANTONE 383 CV
        colors = [colors; [189 135 135]./255];   %PANTONE 5005 CV
        colors = [colors; [249 142 152]./255]; %PANTONE 708 CV
        colors = [colors; [0 102 204]./255]; %RGB R0G102B204
        colors = [colors; [255 153 0]./255]; %RGB R255G153B0
    else
        colors = [0 0 0; 1 0 0; 0 0 1; 1 1 0; 1 0 1; 0.5 0.5 0.5]; % ; 0.75 0 0.750 0.5 0.5];0 1 0; 
        %     colors = [colors; [151 176 133]./255 ; ];
        %     %chartreuse, Light Yellow
        colors = [colors; [163 177 7]./255];  %PANTONE 383 CV
        colors = [colors; [189 135 135]./255];   %PANTONE 5005 CV
        colors = [colors; [249 142 152]./255]; %PANTONE 708 CV
        colors = [colors; [0 102 204]./255]; %RGB R0G102B204
        colors = [colors; [255 153 0]./255]; %RGB R255G153B0
    end
end

% do the real job: add correct sample labels 
Ax = get(gca); % get all properties of plot
nrs = Ax.XTickLabel; %get numbers of spectra
nrs = str2num(nrs);

leafs = cell(max(T),2);
for i=1:max(T)
    pos = find (T ==i);
    leafs{i,1} = i;
    leafs{i,2} = Labels(pos)';
end
for i=1:length (nrs) %generate new labels list
    Labels2(i,1) = leafs{nrs(i),2}(1,1);
end

%show labels
TheLabel = get(Ax.XLabel, 'String');
legend (TheLabel); %show x-axis-label as legend
set(Ax.XLabel, 'Visible', 'off');
Unit = get(gca, 'Units'); %so distance between dendrogram and labels is always the same
set(gca, 'Units', 'pixels');
Pos = get(gca, 'Position');

partx = Pos(1,3) / (length (nrs)+1); %distance of labels
if NewLabellingMethod == false
    for i=1:length (nrs) %normaal max. 30
        g(i) = text (i*partx, -5, Labels2(i,1), 'Rotation', 90, 'HorizontalAlignment', 'right', 'Units', 'Pixels');
    end
end

%generate a matrix with Info about clusters
Info = cell(length (nrs),3); %preallocate cell array for Info matrix
Info (:,1) = num2cell(nrs);
Info (:,2) = Labels2;
if NewLabellingMethod
    Info (:,3) = leafs(nrs,2);
    tree = BTree (Labels); %because originally the 30 leafs may not show all different values
else
    Info (:,3) = GetComb (leafs(nrs,2), false)';
    tree = BTree (Labels2);
end

%colour the labels or place small coloured bars
items = GetItems (tree);
ColorIndex = 0;
for i = 1:length (items)
    ColorIndex = ColorIndex + 1;
    if ColorIndex > size (colors, 1)
        ColorIndex = 1;
        warndlg ('More items than available colors.');
    end
    p = cat(1, items{i}.itemvalues{:});
    if NewLabellingMethod
        LabTable {i, 1} = items{i}.value;
        LabTable {i, 2} = colors(ColorIndex,:);
    else
        set (g(p), 'Color', colors(ColorIndex,:));
    end
end

%Show the info matrix on the screen
if NewLabellingMethod
    g = [];
    for i=1:length (nrs)
        ThisItems = Info {i,3};
        for j=1:length(ThisItems)
            itemstr = ThisItems{j};
            if isempty (itemstr)
                [co, ind] = LocateItem ('NoValue', LabTable, 2);
            else
                [co, ind] = LocateItem (itemstr, LabTable, 2);
            end
            g(length (g)+1) = text (i*partx, j*-5, '-', 'FontWeight', 'Bold', 'Fontsize', 30, 'Units', 'Pixels', 'color', co);
            hand(ind) = g(length (g));
        end
        
    end
    set(gca, 'XTick', [])
    Pos = get(gca, 'Position');
    xl = Pos(1)+10; %to the right of gca
    yl = Pos(4)-10; %beginning at top of gca
    
    for i = 1:size(LabTable,1)
        leghandle1(i) = text (xl, yl, '-', 'FontWeight', 'Bold', 'Fontsize', 30, 'Units', 'Pixels', 'color', LabTable {i, 2});
        leghandle2(i) = text (xl+15, yl, LabTable {i, 1}, 'Units', 'Pixels', 'color', LabTable {i, 2});
        yl = yl-12;
    end
    set (leghandle1, 'Units', Unit);
    set (leghandle2, 'Units', Unit);
else
    fprintf (1, 'Dendrogram Info Matrix:\n\nLeaf\t\tLabel\t\tLeaf items\n');
    for i=1:size (Info,1)
        fprintf (1, '%s', str(Info {i,1}));
        for j=2:size(Info,2)
            fprintf (1, '\t\t%s', str(Info {i,j}));
        end
        fprintf (1,'\n');
    end
end

%set units to correct initial state
set(gca, 'Units', Unit);
set(g, 'Units', Unit);
set(gcf, 'Units', Unit);
