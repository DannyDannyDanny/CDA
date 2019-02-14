%Binary tree object
%
%function Hp = ConvertToBTree(Hp, X)
%
%This function will convert a given numeric, cell or string array into a
%BTree. By building the binary tree, the items positions in the original
%list are saved. By doing this, one can easily get the number of different
%elements in the list and positions of these elements. 
%
%Input parameters:
%   Hp: an empty BTree-structure
%   X:  the numeric, cell or string array which needs to be converted in a
%       BTree
%
%Output parameters:
%   Hp: the created BTree-obect
%
%uses: trimstr, MakeVector

%C 2004-2005, Kris De Gussem, Raman Spectroscopy Research group, Laboratory
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


function Hp = ConvertToBTree(Hp, X);
if nargin ~= 2
    error ('Function requires two input parameters');
end

if ischar (X)
    %convert it to a cell array
    %must be done before calling MakeVector: essentially this is a
    %2-dimensional character array
    for i = 1:size (X,1)
        tmp{i,1} = X(i,:);
    end
    X = tmp;
    clear tmp;
end

if length (X) < 1
%     Hp.items = [];
    warning ('No data assigned to BTree object');
    return;
end
X = MakeVector (X, 'Input X must be a vector');

if isnumeric (X)
    %convert to a cell array
    for i = 1:length (X)
        tmp{i,1} = X(i);
    end
    X = tmp;
    clear tmp;
end

%character and numeric arrays are already converted to cell arrays
if iscell (X) == false
    error ('Input X must be a cell array');
end

%convert cell array to BTree
for i = 1: size (X,1)
    item = X{i};
    if isempty (item)
        Hp.emptyValues {length (Hp.emptyValues)+1} = i;
        Hp.emptyCount = Hp.emptyCount +1;
    else
        if isnumeric (item)
            Hp = Add (Hp, item, i);
        else
            %item = trimstr(item);
            while length (item) >= 1
                if (item (1) == ' ') || (item (1) == 0)
                    item(1) = [];
                else
                    break;
                end
            end
            l = length (item);
            while l >= 1
                if (item (l) == ' ') || (item (l) == 0)
                    item(l) = [];
                else
                    break;
                end
                l = l-1;
            end
            
            Hp = Add (Hp, item, i);
        end
    end
end
