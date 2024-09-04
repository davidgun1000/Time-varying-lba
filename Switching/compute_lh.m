%Function to compute the likelihood as part of algorithm 9b in main paper
function [lh]=compute_lh(data,theta_mu,theta_sig2,theta_latent,num_trials,j)
    
    %state1
    
    
    data_response_repmat = data.response{j,1};
    data_rt_repmat = data.rt{j,1};
    data_cond_repmat = data.cond{j,1};
    rnorm_theta_b1min_kron1 = kron(theta_latent(j,1),ones(num_trials,1));
    rnorm_theta_b2min_kron1 = kron(theta_latent(j,2),ones(num_trials,1));
    rnorm_theta_b3min_kron1 = kron(theta_latent(j,3),ones(num_trials,1));
    rnorm_theta_A_kron1 = kron(theta_latent(j,4),ones(num_trials,1));
    rnorm_theta_v1_kron1 = kron(theta_latent(j,5),ones(num_trials,1));
    rnorm_theta_v2_kron1 = kron(theta_latent(j,6),ones(num_trials,1));
    rnorm_theta_tau_kron1 = kron(theta_latent(j,7),ones(num_trials,1));
 
    [rnorm_theta_bmin_kron1]=reshape_b(data_cond_repmat,rnorm_theta_b1min_kron1,rnorm_theta_b2min_kron1,rnorm_theta_b3min_kron1);
    [rnorm_theta_v_kron1] = reshape_v(data_response_repmat,rnorm_theta_v1_kron1,rnorm_theta_v2_kron1);
    logw(:,1)= real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, rnorm_theta_A_kron1, rnorm_theta_bmin_kron1, rnorm_theta_v_kron1, ones(num_trials,1),rnorm_theta_tau_kron1)));

    rnorm_theta_b1min_kron2 = kron(theta_latent(j,8),ones(num_trials,1));
    rnorm_theta_b2min_kron2 = kron(theta_latent(j,9),ones(num_trials,1));
    rnorm_theta_b3min_kron2 = kron(theta_latent(j,10),ones(num_trials,1));
    rnorm_theta_A_kron2 = kron(theta_latent(j,11),ones(num_trials,1));
    rnorm_theta_v1_kron2 = kron(theta_latent(j,12),ones(num_trials,1));
    rnorm_theta_v2_kron2 = kron(theta_latent(j,13),ones(num_trials,1));
    rnorm_theta_tau_kron2 = kron(theta_latent(j,14),ones(num_trials,1));
    rnorm_theta_bmin_kron2=reshape_b(data_cond_repmat,rnorm_theta_b1min_kron2,rnorm_theta_b2min_kron2,rnorm_theta_b3min_kron2);
    [rnorm_theta_v_kron2] = reshape_v(data_response_repmat,rnorm_theta_v1_kron2,rnorm_theta_v2_kron2);
    logw(:,2) = real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, rnorm_theta_A_kron2, rnorm_theta_bmin_kron2, rnorm_theta_v_kron2, ones(num_trials,1),rnorm_theta_tau_kron2)));
    logw = logw';
    maxl = max(logw,[],1);
    lh = exp(logw - maxl(ones(2,1),:));
    
  
end