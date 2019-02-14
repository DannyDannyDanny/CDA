%Binary tree object
%
%function [Items, ItemsList, numlabels] = GetItems(Hp)
%
%This function returns a list of all items in the BTree.
%
%Input parameters:
%   Hp: the current BTree-obect
%
%Output parameters:
%   Items: a structure array containing the different elements, with their
%      itemvalues
%   ItemsList: a array of the elements
%   numlabels: a numeric array, different numbers indicate different
%      class membership of elements (corresponding to the original list; in
%      this case, the position of the element is given as third parameter
%      in the add function of the BTree object. Alternatlively the
%      BTree is build using the ConvertToBTree function.).
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

function [Items, ItemsList, numlabels] = GetItems(Hp);
Items = GetThem (Hp.items, []);
if isempty (Hp.emptyValues) == false
    ll = length(Items)+1;
    Items{ll}.value = 'NoValue';
    Items{ll}.count = Hp.emptyCcount;
    Items{ll}.itemvalues = Hp.emptyValues;
end

if nargout >=2
    for i = 1:length (Items)
        ItemsList{i} = Items{i}.value;
    end
end

%make an equivalent numerical array of itemslist
if nargout >= 3
    numlabels = [];
    for i = 1:length (Items)
        pos = Items{i}.itemvalues;
        for j=1:length (pos)
            numlabels( pos{j},1) = i;
        end
    end
end
