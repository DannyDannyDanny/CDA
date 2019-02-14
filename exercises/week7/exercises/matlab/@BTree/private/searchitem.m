%Binary tree object
%
%private function it = searchitem (list, str, retfullitem)
%
%This function searches for an item str in the tree of items (list). It is
%repeatedly called in an iterative way.
%
%Input parameters:
%   list: list of items in BTree structure
%   str: the item to look for
%   retfullitem: boolean value: true if the full elements structure
%      needs to be returned, false if searchitem returns a boolean value
%      indicating if the element is found
%
%Output parameters:
%   it: the structure of the item which is searched

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


function it = searchitem (list, str, retfullitem)
if isempty (list)
    %the element is not found in the list
    it = [];
    return;
else
    %Is it the element we look for?
    found = DoCompare (str, list.value);
    if found == 0
        if retfullitem
            tmp.value = list.value;
            tmp.count = list.count;
            tmp.itemvalues = list.itemvalues;
            it = tmp;
            return;
        else
            it = 1; %succes
        end
    elseif found < 0
        %look in the left branch
        it = searchitem (list.left, str, retfullitem);
    else
        %look in the right branch
        it = searchitem (list.right, str, retfullitem);
    end
end
