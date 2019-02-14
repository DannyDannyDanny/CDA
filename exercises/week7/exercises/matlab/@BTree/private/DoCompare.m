%Binary tree object
%
%function res = DoCompare (item, value)
%
%This function compares two values (either strings or numbers) and returns
%a value indicating if the first value is lower, equal to or higher than
%the second.
%
%Input parameters:
%   item: first value
%   value: value to compare first with
%
%Output parameters:
%   res: if 
%           < 0: item has a lower value than value
%           = 0: item is equal to value
%           > 0: item has a higher value

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

function tmp = DoCompare (item, value);
itn = isnumeric (item);
itc = ischar (item);
van = isnumeric (value);
vac = ischar (value);

if ~(itn || itc)
    error (sprintf('Support for this type (''%s'')of data needs to be programmed in future ...', class (item)));
end

if ~(van || vac)
    error (sprintf('Support for this type (''%s'')of data needs to be programmed in future ...', class (value)));
end

if itn && van
    tmp = item - value;
    return;
end
if itn
    item = num2str(item);
end
if van
    value = num2str(value);
end

%tmp = mystrcmp (item, value);
if ischar (item) == false
    error ('String 1 is not a string');
end
if ischar (value) == false
    error ('String 2 is not a string');
end

l1 = length (item);
l2 = length (value);
d = l1 - l2;
if d < 0
    value = value(1:l1);
elseif d > 0
    item = item(1:l2);
end

itemb = double (item);
valueb = double (value);
pos = find(itemb - valueb); %search for differing characters: this isfaster than find(item ~= value)

if isempty (pos)
    if d < 0
        retval = -1;
    elseif d == 0
        retval = 0;
    else
        retval = 1;
    end
else
    if item (pos(1)) < value (pos(1))
        retval = -1;
    else
        retval = 1;
    end
end
tmp=retval;

% if %isnumeric (item)
%     if isnumeric (value)
%         tmp = item - value;
%     else
%         %a number and a string: convert number to string
%         tmp = mystrcmp (num2str(item), value);
%     end
% elseif ischar (item)
%     if isnumeric (value)
%         %a number and a string: convert number to string
%         tmp = mystrcmp (item, num2str(value));
%     else
%         tmp = mystrcmp (item, value);
%     end
% else
%     error (sprintf('Support for this type (''%s'')of data needs to be programmed in future ...', class (item)));
% end
