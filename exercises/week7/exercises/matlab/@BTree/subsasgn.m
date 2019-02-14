%Binary tree object
%
%function Hp=subsasgn(Hp, index, val)
%
%This function is responsible for correctly handling assignments to the
%BTree object. It controls the correct syntax, validation and assigns
%the appropriate information. This function is automatically called and
%shouldn't be used by the user of the BTree object.
%
%see also help subsasgn for more information
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


function Hp=subsasgn(Hp, index, val)

%check input
nnargin = nargin;
error(nargchk(3,3,nnargin));   %give error if nargin is not appropriate

if isempty(index); error('  Invalid subscripting'); end;

%check type of assignment
switch index(1).type
    case '.'
        switch index(1).subs
            case 'info'
                if ischar (val)
                    Hp.info = val;
                else
                    error ('Assignment to field info must be a string');
                end
            otherwise
                error ('Assignment is not valid.');
        end
        
    otherwise
        error ('  Invalid subscripting');
end
