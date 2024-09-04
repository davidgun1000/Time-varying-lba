function simout = dirichsim(par,varargin)
% simulates from a Dirichlet posterior density 

% input
% par ... parameter of the aposteriori density, the number of colums
%         determines the  number of categories
%
% for two argument:  varargin determines the number of draws, otherwise it assumed that a single darw is required  

% output: simout ... simulated values  (varargin times K)  array 

% Author: Sylvia Fruehwirth-Schnatter
% Last change: September 13, 2006

if nargin==2     
    M=varargin{1}; 
    if and(size(par,1)~=M,size(par,1)~=1)   warn(['Size disagreement in the variable par in function dirichsim']); fl=[]; return;     end 
else 
    M=size(par,1); 
end

cat=size(par,2);


if size(par,1)==1  %  a sequence of M draws from a single density  
    
    gam = gamrnd(repmat(par,M,1),1);
    simout = gam ./ repmat(sum(gam,2),1,cat);
    
    
else  % a single draw from a sequence of densities
    
    gam = gamrnd(par,1);
    simout = gam ./ repmat(sum(gam,2),1,cat);
    
end

