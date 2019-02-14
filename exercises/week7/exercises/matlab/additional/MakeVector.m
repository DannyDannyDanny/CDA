%function V = MakeVector (V, msg)
%
%MakeVector will check for an input matrix V if it's a vector, and if not
%it will display the error message msg. MakeVector will return the input
%matrix as a column vector.
%

%C 2003, Kris De Gussem, Raman Spectroscopy Research group, Laboratory
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


function V = MakeVector (V, msg);
if ndims(V) > 2
    disp ('The number of dimensions is:');
    ndims(V)
    error ('The number of dimensions is to big. A twodimensional array must be given...');
end

n = size (V);
if n(1,2 ) > 1
    if n(1,1) > 1
        error (msg);
    else
        V = V';
    end
end
clear n;
