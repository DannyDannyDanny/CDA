%Binary tree object
%
% private function: listitems = GetThem (list, listitems);
%
%This function returns a list of all the non-empty items in the binary tree
%or part of the binary tree. This function returns the items of the binary
%tree only in the part of the tree as given by the structure list. GetThem
%is called iteratively.
%
%Input parameters:
%   list: the part of the binary tree of which the items must be returned.
%   listitems: the items already returned, the new items found will be
%      added to the list and returned. If the programmer calls this
%      function, this parameter must be empty.
%
%Output parameters:
%   listitems: the list of items already returned. The new found items are
%      added.
%
%See also Btree/add, Btree/ConvertToBtree.
%

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

function listitems = GetThem (list, listitems);
if isempty (list) == false
    listitems = GetThem (list.left, listitems);
    l = length (listitems);
    item.value = list.value;
    item.count = list.count;
    item.itemvalues = list.itemvalues;
    listitems{l+1} = item;
    listitems = GetThem (list.right, listitems);
end
