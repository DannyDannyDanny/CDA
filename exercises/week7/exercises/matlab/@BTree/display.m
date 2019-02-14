%Binary tree object
%
%function display(Hp)
%
%This function is responsible for displaying the contents of the BTree
%object. Normally, you don't have to call this function.

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


function display(Hp); 
fprintf (1, '\n');
fprintf (1, '%s = \n', inputname(1));
fprintf('\tBTree object\n');

fprintf (1, '\t\tcontaining %s items\n', int2str(Hp.count));
fprintf (1, '\t\tDescription: %s\n', Hp.info);
fprintf (1, '\n\tList of items in the binary tree:\n\n');
t = GetItems (Hp);
for i=1:length (t)
    display(t{i})
end
