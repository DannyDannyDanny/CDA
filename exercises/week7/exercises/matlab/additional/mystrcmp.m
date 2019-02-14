%function retval = mystrcmp (str1, str2)
%
%Mystrcmp will compare two strings in the same way as the standard
%c-functions do. In contrary to the matlab function, it will return a value
%of -1 of the first string has a 'lower value' than the second string, and
%+1 if the first string has a 'higher value'.
%The first different character will determine retval. In case the two
%strings have different lengths and only the longer part of the string is
%the difference, then this will determine retvals value.
%
%See also help strcmp
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


function retval = mystrcmp (str1, str2);
if ischar (str1) == false
    error ('String 1 is not a string');
end
if ischar (str2) == false
    error ('String 2 is not a string');
end

l1 = length (str1);
l2 = length (str2);
d = l1 - l2;
if d < 0
    str2 = str2(1:l1);
elseif d > 0
    str1 = str1(1:l2);
end

%pos = find(str1 ~= str2); %search for differing characters
str1b = double (str1);
str2b = double (str2);
pos = find(str1b - str2b); %search for differing characters: this isfaster than find(str1 ~= str2)

if isempty (pos)
    if d < 0
        retval = -1;
    elseif d == 0
        retval = 0;
    else
        retval = 1;
    end
else
    if str1 (pos(1)) < str2 (pos(1))
        retval = -1;
    else
        retval = 1;
    end
end
