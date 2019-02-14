% function LocateItem
% 
% This function Locates an Item in a list or table. LocateItem returns the
% value in the nth column. This function is especially useful for use with
% conversion tables or lookup tables. As such it is very usable in e.g.
% import routines where coded and cryptic filenames need to be converted in
% useful sample descriptions.
% 
% Syntax: it = LocateItem (str, list, nrcol)
% 
% Input parameters:
%    str: string containing the item to search for
%    list: list of cell strings, first column containing the list items,
%       while the following columns can contain additional info, new
%       values for the old values in the first column, ...
%    nrcol: the number of the column (or field) which will be returned
%
% Output parameters:
%    it: the value in the nrcol'th column corresponding to str
%

%C 2005, Kris De Gussem, Raman Spectroscopy Research group, Laboratory
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


function [it, i] = LocateItem (str, list, nrcol);
OK = false;
for i = 1:size (list,1)
    listit = list{i,1};
    if strcmp (listit, str)
        %it = list{i,nrcol};
        %return;
        OK = true;
        break
    end
end
if OK %this is for speed optimisation
    it = list{i,nrcol};
else
    it = []; %in case str is not found
end
