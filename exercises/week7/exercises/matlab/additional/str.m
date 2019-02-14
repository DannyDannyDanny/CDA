%function str
%
%This function formats the input value as a string, regarding the type.
%Currently supported input values are logicals, doubles, chars and cells.
%It outputs the converted value.
%
%syntax:
%     s = str (value, domatrixconversion)
%     s = str (value)
%
%With:
%     value: a single logical, double string or cell value
%     domatrixconversion: if given and if it is 1, then value should be a
%         cell array and all elements in the 2-dimensional cell array value
%         are converted to strings, resulting in a cellstring array
%     s: the value (or cell array of values) converted into a string
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


function s = str (value, domatrixconversion);
if nargin == 1
    domatrixconversion = false;
end

switch class(value)
    case 'logical'
        s = int2str (value);
    case 'double'
        s = num2str (value);
    case 'char'
        s = value;
    case 'cell'
        if domatrixconversion == 1
            for i = 1:size (value,1)
                for j = 1:size (value, 2)
                    s{i,j} = str (value{i,j});
                end
            end
        else
            s = str (value{1});
        end
    otherwise
        error ('Unknown data type for conversion to string. Please contact the programmer of the biodata object.');
end
