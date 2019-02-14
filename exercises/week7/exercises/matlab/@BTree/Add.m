%Binary tree object
%
%function [Hp, succes] = Add(Hp, item, itemval)
%
%This function adds an item to the binary tree. It also adds an element
%describing this object. If the item is already in the list, then only
%its describing or characterising value is added to the list. As such a
%list of itemvalues is build, when the binary tree is filled. For instance
%if for each tree element the position in a list is given, then you easilly
%get the different elements in a list, the possitions of the different
%elements and the number of times each element is found in the original
%list.
%
%Input parameters:
%   Hp: the current BTree-obect
%   item: the cell (string, number) which must be attached to the binary tree
%   itemval: a value characterising item: e.g. the position in a list
%
%and returns:
%   Hp: the BTree-object with the added item
%   succes: a code representing
%      0: item already in the list: added itemval to the properties of the
%         item
%      1: you've add a new item to the BTree
%
%uses: mystrcmp

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

function [Hp, succes] = Add(Hp, item, itemval);
%the only simple method to ensure you have a fully balanced binary tree is
%to start from an already sorted list (of unique items), divide the number
%of items iteratively by two, assign one half to the left branch and the
%other half to the right branch.
%So the easiest way to become a fully balanced tree, is after every sample
%is added, to convert the tree to a list/array (using the getitems method)
%and then convert it back to a binary tree (just physically an other way of
%representing the same data, and as such an O(n) operation).

if nargin ~= 3
    error ('Add-function of the BTree-object requires three input parameters...');
end

[n p] = size (item);
if n ~= 1
    error ('Item must be a single data-element, not an array');
end

%add the element to the tree
if isempty (item)
    Hp.emptyValues {length (Hp.emptyValues)+1} = itemval;
    Hp.emptyCount = Hp.emptyCount +1;
    succes = 1;
else
    [Hp.items, succes] = AddItem(Hp.items, item, itemval);
    switch succes
        case 0
            %item already existed in the tree
        case 1
            Hp.count = Hp.count + 1;
        otherwise
            error (sprintf ('''%i'': unknown succescode', succes));
    end
end

function [list, succes, rebuilt] = AddItem (list, item, itemval);
%This function is called repeatedly in an iterative way.
%parameters:
%   list: the tree of items
%   item: the element which needs to be add to the list
%   itemval: the elements descriptive value
%   succes: a value representing the succes of addition

%generate a default tree item structure
%itemstruc = struct ('value', {[]}, 'itemvalues', {[]}, 'count', {0}, 'left', {[]}, 'right', {[]});
itemstruc.value=[];
itemstruc.itemvalues=[];
itemstruc.count=0;
itemstruc.left=[];
itemstruc.right=[];
itemstruc.leftbranches=0;
itemstruc.rightbranches=0;
itemstruc.nivleft=0;
itemstruc.nivright=0;
rebuilt = 0;

if isempty (list)
    list = itemstruc;
    %possibility 1: nothing in the list at the moment:
    list.value = item;
    list.itemvalues{1} = itemval;
    list.count = list.count + 1;
    succes = 1;
    %newniv = 1;
else
    %possibility 2: there are already elements in the list:
    %search further after comparison
   
    %compare the items: get a value indicating if the new element has
    %higher, lower or the same value as the element in the list
    tmp = DoCompare (item, list.value);
    
    %compare the items: do comparison
    if tmp == 0
        %items have the same value
        l = length(list.itemvalues);
        list.itemvalues{l+1} = itemval;
        list.count = list.count + 1; % the number of times the element is found in the original list
        succes = 0;
    elseif tmp < 0
        %new element has a lower value than the element in the list
        doadd=(isempty (list.right) == false) || (isempty (list.left));
        if doadd == false
            %right = not empty
            %compare this value with the right value
            tmp = DoCompare (item, list.left.value); %if same value: add to the left (calling additem as subroutine will just add 1 to the counter of the left value)
            doadd = (tmp == 0);
        end
        
        if doadd%to ensure a balanced tree
            %add to left branch
            [list.left, succes, rebuilt] = AddItem (list.left, item, itemval);
            if succes == 1
                list.nivleft = max (list.left.nivleft, list.left.nivright) + 1;
                %if newniv
                    %list.nivleft = list.nivleft + 1;
                    %end
                list.leftbranches = list.leftbranches + 1;
                ma = max (list.rightbranches, list.leftbranches);
                mi = min (list.rightbranches, list.leftbranches);                 
                diffniv = abs (list.nivleft - list.nivright); % the number of differing levels
                if (ma > (1.2 * mi + 2)) && (diffniv > 1) %determine if tree  has to be rebuilt
                    list = Rebuilt (list);
                    rebuilt = 1;
                end
            end
        else
            %do switch (if C < B: is equal to tmp == -1)    A         ===>     B
            %                                             B                  C   A
            if tmp == -1 %item and left value are not the same, if -1: then C<B and B<A
                list.right = itemstruc;
                list.right.count = list.count;
                list.right.value = list.value;
                list.right.itemvalues = list.itemvalues;
                
                list.value = list.left.value;
                list.itemvalues = list.left.itemvalues;
                list.count = list.left.count;
                list.rightbranches = 1;
                list.nivright = 1;
                if (isempty (list.left.left) == false) ||(isempty (list.left.right) == false)
                    error 
                    %this shouldn't happen if everything previously is
                    %executed correctly.
                end
                
                %add new item to the left
                list.left = itemstruc;
                list.left.value = item;
                list.left.itemvalues{1} = itemval;
                list.left.count = list.left.count + 1;
                succes = 1;
                list.leftbranches = 1;
                list.nivright = 1;
                
            else
                list.right = itemstruc;
                list.right.count = list.count;
                list.right.value = list.value;
                list.right.itemvalues = list.itemvalues;
                
                %add new item to the middle
                list.itemvalues = [];%clear it first
                list.value = item;
                list.itemvalues{1} = itemval;
                list.count = 1;            
                succes = 1;
                list.rightbranches = 1;
                list.nivright = 1;
            end
        end
    else
        %new element has a higher value than the element in the list
        doadd=(isempty (list.left) == false) || (isempty (list.right));
        if doadd == false
            %right = not empty
            %compare this value with the right value
            tmp = DoCompare (item, list.right.value);
            doadd = (tmp == 0);
        end
        
        if doadd%to ensure a balanced tree
            %add to right branch
            [list.right, succes, rebuilt] = AddItem (list.right, item, itemval);
            if succes == 1
                list.nivright = max (list.right.nivleft, list.right.nivright) + 1;
                %if newniv
                    %list.nivright = list.nivright + 1;
                    %end
                list.rightbranches = list.rightbranches + 1;
                ma = max (list.rightbranches, list.leftbranches);
                mi = min (list.rightbranches, list.leftbranches);
                diffniv = abs (list.nivleft - list.nivright); % the number of differing levels
                if (ma > (1.2 * mi + 2)) && (diffniv > 1) %determine if tree  has to be rebuilt
                    list = Rebuilt (list);
                    rebuilt = true;
                end
            end
        else
            %do switch  (if C > B : is equal to tmp == 1):  A         ===>     B
            %                                                 B              A   C
            if tmp == 1
                list.left = itemstruc;
                list.left.count = list.count;
                list.left.value = list.value;
                list.left.itemvalues = list.itemvalues;
                
                list.value = list.right.value;
                list.itemvalues = list.right.itemvalues;
                list.count = list.right.count;
                list.leftbranches = 1;
                list.nivleft = 1;
                if (isempty (list.right.left) == false) ||(isempty (list.right.right) == false)
                    error 
                    %this shouldn't happen if everything previously is
                    %executed correctly.
                end
                
                %add new item to the right
                list.right = itemstruc;
                list.right.value = item;
                list.right.itemvalues{1} = itemval;
                list.right.count = list.right.count + 1;
                succes = 1;
                list.rightbranches = 1;
                list.nivright = 1;
                
            else
                list.left = itemstruc;
                list.left.count = list.count;
                list.left.value = list.value;
                list.left.itemvalues = list.itemvalues;
                
                %add new item to the middle
                list.itemvalues = [];%clear it first
                list.value = item;
                list.itemvalues{1} = itemval;
                list.count = 1;            
                succes = 1;
                list.rightbranches = 1;
                list.nivright = 1;
            end
        end
    end
end
if rebuilt
list.nivleft = max (list.left.nivleft, list.left.nivright) + 1; %now this is doen twice: here it is placed to ensure correct number of levels when rebuilding the tree
list.nivright = max (list.right.nivleft, list.right.nivright) + 1;
end

function list = Rebuilt (list);
items = GetThem (list, []); %get the items in sorted order
list = DivideList (items);

function list = DivideList (items);
%determine optimal order
len = length (items);
mid = ceil (len /2);
toleft = 1:mid-1;
toright = mid+1:len;
nrleft = length(toleft);
nrright = length(toright);

%divide items into tree
ThisItem = items{mid};
list.value = ThisItem.value;
list.itemvalues = ThisItem.itemvalues;
list.count = ThisItem.count;
if nrleft
    list.left = DivideList (items(toleft));
    list.nivleft = max (list.left.nivleft, list.left.nivright) + 1;
else
    list.left = [];
    list.nivleft = 0;
end
if nrright
    list.right = DivideList (items(toright));
    list.nivright = max (list.right.nivleft, list.right.nivright) + 1;
else
    list.right = [];
    list.nivright = 0;
end
list.leftbranches = nrleft;
list.rightbranches = nrright;
