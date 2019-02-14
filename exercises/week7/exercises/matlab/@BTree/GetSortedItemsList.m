%Binary tree object
%
%SortedItemsList = GetSortedItemsList (tree);
%
%This function returns a sorted list of all individual elements from which
%the binary tree is build.
%
%See also BTree/GetItems.
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

function SortedItemsList = GetSortedItemsList (tree);
Items = GetItems (tree);

%     if isnumeric (Items{1}.value)
%         SortedItemsList = [];
%         %then the whole btree should be build from numeric elements?
%         for i = 1:length (Items)
%             nritem = length(Items{i}.itemvalues);
%             nrlist = length (SortedItemsList);
%             SortedItemsList (nrlist+1 : 1 : nrlist+nritem,1) = repmat (Items{i}.value, nritem, 1); 
%             
%         end
%         
%     else
SortedItemsList = {};
for i = 1:length (Items)
    
    nritem = length(Items{i}.itemvalues);
    nrlist = length (SortedItemsList);
    %the str command in the following line is necessary if str is a
    %number written as a string then matlab will automatically
    %convert is to a number, resulting in an error
    %tmp = repmat ({str(Items{i}.value)}, nritem, 1);
    tmp = repmat ({Items{i}.value}, nritem, 1);
    SortedItemsList (nrlist+1 : 1 : nrlist+nritem,1) = tmp;
    
end
%     end
