function [theta_latent_b1min,theta_latent_A,theta_latent_v1,theta_latent_v2,theta_latent_tau]=LBA_MC_IS_orig_real_log(data,param,num_subjects,num_trials,num_particles)

parfor j=1:num_subjects
     
    rnorm=mvnrnd(param.theta_mu',param.theta_sig2,num_particles);
    
    rnorm_theta_b1min=rnorm(:,1);
    rnorm_theta_A=rnorm(:,2);
    rnorm_theta_v1=rnorm(:,3);
    rnorm_theta_v2=rnorm(:,4);
    rnorm_theta_tau=rnorm(:,5);
    
    rnorm_theta_b1min_kron=kron(rnorm_theta_b1min,ones(num_trials(j,1),1));
    rnorm_theta_A_kron=kron(rnorm_theta_A,ones(num_trials(j,1),1));
    rnorm_theta_v1_kron=kron(rnorm_theta_v1,ones(num_trials(j,1),1));
    rnorm_theta_v2_kron=kron(rnorm_theta_v2,ones(num_trials(j,1),1));
    rnorm_theta_tau_kron=kron(rnorm_theta_tau,ones(num_trials(j,1),1));
    
    data_response_repmat=repmat(data.response{j,1}(:,1),num_particles,1);
    data_rt_repmat=repmat(data.rt{j,1}(:,1),num_particles,1);
    data_cond_repmat=repmat(data.cond{j,1}(:,1),num_particles,1);  
    
    %[rnorm_theta_bmin_kron]=reshape_b(data_cond_repmat,rnorm_theta_b1min_kron,rnorm_theta_b2min_kron,rnorm_theta_b3min_kron);
    [rnorm_theta_bmin_kron]=rnorm_theta_b1min_kron;
    
    [rnorm_theta_v_kron]=reshape_v(data_response_repmat,rnorm_theta_v1_kron,rnorm_theta_v2_kron);
    lw=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, rnorm_theta_A_kron, rnorm_theta_bmin_kron, rnorm_theta_v_kron, ones(num_particles*num_trials(j,1),1),rnorm_theta_tau_kron)));
    
    lw_reshape=reshape(lw,num_trials(j,1),num_particles);
    logw_first=sum(lw_reshape);
    logw=logw_first';
    
    
    id=imag(logw)~=0;
    id=1-id;
    id=logical(id);
    logw=logw(id,1); 
    logw=real(logw);

    if sum(isinf(logw))>0 | sum(isnan(logw))>0
     id=isinf(logw) | isnan(logw);
     id=1-id;
     id=logical(id);
     logw=logw(id,1);
    end
    
    max_logw=max(real(logw));
    weight=real(exp(logw-max_logw));
    weight=weight./sum(weight);
    if sum(weight<0)>0
        id=weight<0;
        id=1-id;
        id=logical(id);
        weight=weight(id,1);
    end
    Nw=length(weight);
    
    if Nw>0 
        ind=randsample(Nw,1,true,weight);
        theta_latent_b1min(j,1)=rnorm_theta_b1min(ind,1);
        theta_latent_A(j,1)=rnorm_theta_A(ind,1);
        theta_latent_v1(j,1)=rnorm_theta_v1(ind,1);
        theta_latent_v2(j,1)=rnorm_theta_v2(ind,1);
        theta_latent_tau(j,1)=rnorm_theta_tau(ind,1);
    end
        
%----------------------------------------------------------------------------------------------------------------------------------    
    
end

end


%       rnorm_b=normt_rnd(theta.b_mu,theta.b_sig2,theta.left_truncation,theta.right_truncation,num_particles);
%       rnorm_A=normt_rnd(theta.A_mu,theta.A_sig2,theta.left_truncation,theta.right_truncation,num_particles);
%       rnorm_v1=normt_rnd(theta.v1_mu,theta.v1_sig2,theta.left_truncation,theta.right_truncation,num_particles);
%       rnorm_v2=normt_rnd(theta.v2_mu,theta.v2_sig2,theta.left_truncation,theta.right_truncation,num_particles);
%       
%       rnorm_b(1,1)=latent_b(j,1);
%       rnorm_A(1,1)=latent_A(j,1);
%       rnorm_v1(1,1)=latent_v1(j,1);
%       rnorm_v2(1,1)=latent_v2(j,1);
%       
%       data_response_j=data.response{j,1};
%       data_rt_j=data.rt{j,1};
%       for k=1:num_particles
%       for i=1:num_trials
%           if data_response_j(i,1)==1
%              rnorm_v=[rnorm_v1(k,1),rnorm_v2(k,1)];
%           else
%              rnorm_v=[rnorm_v2(k,1),rnorm_v1(k,1)];
%           end
%           log_lik_ind(i,1)=real(log(LBA_n1PDF(data_rt_j(i,1),rnorm_A(k,1),rnorm_b(k,1),rnorm_v,1)));
%       end
%       lw(k,1)=sum(log_lik_ind);
%       end
%       max_lw=max(lw);
%       weight=real(exp(lw-max_lw));
%       weight=weight./sum(weight);
%       ind=randsample(num_particles,1,true,weight);
%       latent_b(j,1)=rnorm_b(ind,1);
%       latent_A(j,1)=rnorm_A(ind,1);
%       latent_v1(j,1)=rnorm_v1(ind,1);
%       latent_v2(j,1)=rnorm_v2(ind,1);