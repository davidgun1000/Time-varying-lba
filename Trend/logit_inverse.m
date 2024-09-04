function logit_inv=logit_inverse(x)
%compute transformation for \phi_{\alpha}
logit_inv=log(x./(1-x));
end
