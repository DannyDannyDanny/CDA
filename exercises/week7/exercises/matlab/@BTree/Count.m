%Binary tree object
%
%function info = Count (Hp)
%
%This function returns an information structure about the BTree object
%Hp with the number of items, the number of left and right branches.
%
%Input parameters:
%   Hp: the current BTree-obect 
%
%Output parameters:
%   info: information structure containing the number of items, the number
%       of left and right branches.

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


function info = Count (Hp);
info.count = 0;
info.leftcount = 0;
info.rightcount = 0;

info = DoCount (Hp.items, info);

function info = DoCount (list, info)
if isempty (list.value) == false
    list.value
    info.count = info.count + 1;
    if isempty (list.left) == false
        list.left.value
        info.leftcount = info.leftcount + 1;
        info = DoCount (list.left, info);
    end
    if isempty (list.right) == false
        list.right.value
        info.rightcount = info.rightcount + 1;
        info = DoCount (list.right, info);
    end
end
