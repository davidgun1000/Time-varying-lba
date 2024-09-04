%function used to select the threshold depending on the experimental
%condition. 
function [theta_latent_b]=choose_b(data_cond,theta_latent_b1,theta_latent_b2,theta_latent_b3)

      if data_cond==1
         theta_latent_b=theta_latent_b1;
      end
      if data_cond==2
         theta_latent_b=theta_latent_b2;
      end
      if data_cond==3
         theta_latent_b=theta_latent_b3;
      end



end