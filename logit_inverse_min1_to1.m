function logit_inv=logit_inverse_min1_to1(x)
u=(x+1)./2;
logit_inv=log(u./(1-u));
