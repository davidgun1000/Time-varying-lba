function indx=rs_multinomial(w)
%this function is the multinomial resampling


N=length(w); % number of particles
indx=zeros(1,N); % preallocate 
Q=cumsum(w); % cumulative sum
u=sort(rand(1,N)); % random numbers

j=1;
for i=1:N
    while (Q(j)<u(i))
        j=j+1; % climb the ladder
    end
    indx(i)=j; % assign index
end

% if conditional
%     indx(1)=1; % static conditional index
% end