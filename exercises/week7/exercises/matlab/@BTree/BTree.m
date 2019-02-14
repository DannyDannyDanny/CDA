%Binary tree object
%
%function Hp = BTree(X, description)
%
%This function initialises the binary tree. Each element can have a left
%branch (with a lower value, or less important than the element), and a
%right branch which has a higher value, or more important than the
%element). The binary tree is an efficient data-structure for sorting a
%list, obtaining the different values of a list and e.g. the positions of
%the different values. However, although code is implemented to ensure a
%certain degree to obtain a balanced tree, it is yet impossible to obtain a
%fully balanced tree with this implementation.
%
%Parameters:
%   Hp: the created btree-obect
%   X:  the numeric, cell or string array which needs to be converted in a
%       btree
%   description: optional string: a description for the data
%
%See also BTree/ConvertToBTree.

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

function Hp = BTree(X, description);
%generate the default structure
Hp = struct ('count', {0}, 'emptyValues', {{}}, 'emptyCount', {0}, 'info', {''}, 'items', {[]});
Hp = class(Hp,'BTree');

%check input + convert it to a BTree-structure
switch nargin
    case 0
        
    case 1
        if isa(X,'BTree')
            Hp = X;
        elseif iscell (X) || isnumeric (X) || ischar (X)
            %convert the array to a BTree
            Hp = ConvertToBTree (Hp, X);
        else
            error (sprintf ('''%s'': unsupported data input type. See help BTree for more information.', class (X)));
            
        end
        
        Hp.info = '';
    case 2
        %convert the cell array to a binary tree
        if ischar (description)
            Hp.info = description;
        else
            Hp.info = '';
        end
        Hp = ConvertToBTree (Hp, X);
        
    otherwise
        error ('Wrong number of input parameteres');
end
