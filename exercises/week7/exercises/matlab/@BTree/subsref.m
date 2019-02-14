%Binary tree object
%
%function b = subsref(Hp,s)
%
%This function is responsible for correctly handling references to the
%BTree object. It controls the correct syntax, and returns the
%appropriate information. This function is automatically called and
%shouldn't be used by the user of the BTree object.
%
%see also help subsref for more information
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


function b = subsref(Hp,s);

nnargin = nargin;
error(nargchk(2,2,nnargin));   %give error if nargin is not appropriate

if isempty(s) error('  Invalid subscripting'); end;


switch s(1).type
case '.'
    switch s(1).subs
        case 'count'
            b = Hp.count;
        case 'info'
            b = Hp.info;
        otherwise
            error (strcat (s.subs, ': Undefined field!'));
    end
    %s(1) = [];
otherwise
   error('Undefined subsref')
end
pfield = s(1);
s(1) = [];
if isempty(s)    %nothing other than indexing into main object? just return reduced object
   %er werd alleen een veld opgevraagd
   return;
else
    error ('Referencing is to deep.');
end
