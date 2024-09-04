  function [theta_latent]=LBA_MC_IS_cond_orig_real_log_prop_usedmixchol3(data,param,...
theta_latent,...
num_subjects,num_trials,num_particles,mean_theta,covmat_theta,i,burn,adapt,j,S)

    epsilon=0.3;
    reference_par=theta_latent;
    num_randeffect=length(theta_latent);
    if i>=burn+adapt+1
    w1_mix=0.5;
    w2_mix=0.45;
    w3_mix=1-w1_mix-w2_mix;
    u=rand(num_particles,1);
    id1=(u<w1_mix);
    id2=(u>w1_mix) & (u<=(w1_mix+w2_mix));
    id3=(u>(w1_mix+w2_mix)) & (u<=(w1_mix+w2_mix+w3_mix));
    n1=sum(id1);
    n2=sum(id2);
    n3=sum(id3);
    id1=logical(id1);
    id2=logical(id2);
    id3=logical(id3);
    chol_theta_sig2=chol(param.theta_sig2,'lower');
    chol_theta_sig2_1=log(chol_theta_sig2(1,1));
    chol_theta_sig2_2=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
    chol_theta_sig2_3=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
    chol_theta_sig2_4=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
    chol_theta_sig2_5=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
    chol_theta_sig2_6=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
    chol_theta_sig2_7=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];
    chol_theta_sig2_8=[chol_theta_sig2(8,1:7),log(chol_theta_sig2(8,8))];
    chol_theta_sig2_9=[chol_theta_sig2(9,1:8),log(chol_theta_sig2(9,9))];
    chol_theta_sig2_10=[chol_theta_sig2(10,1:9),log(chol_theta_sig2(10,10))];
    chol_theta_sig2_11=[chol_theta_sig2(11,1:10),log(chol_theta_sig2(11,11))];
    chol_theta_sig2_12=[chol_theta_sig2(12,1:11),log(chol_theta_sig2(12,12))];
    chol_theta_sig2_13=[chol_theta_sig2(13,1:12),log(chol_theta_sig2(13,13))];
    chol_theta_sig2_14=[chol_theta_sig2(14,1:13),log(chol_theta_sig2(14,14))];
    
    xx=[param.theta_mu';chol_theta_sig2_1';chol_theta_sig2_2';chol_theta_sig2_3';...
        chol_theta_sig2_4';chol_theta_sig2_5';chol_theta_sig2_6';chol_theta_sig2_7';
        chol_theta_sig2_8';chol_theta_sig2_9';chol_theta_sig2_10';chol_theta_sig2_11';
        chol_theta_sig2_12';chol_theta_sig2_13';chol_theta_sig2_14'];
    cond_mean=mean_theta(j,1:num_randeffect)'+covmat_theta(1:num_randeffect,num_randeffect+1:end,j)*((covmat_theta(num_randeffect+1:end,num_randeffect+1:end,j))\(xx-mean_theta(j,num_randeffect+1:end)'));
    cond_mean_ref=reference_par';
    cond_var=covmat_theta(1:num_randeffect,1:num_randeffect,j)-covmat_theta(1:num_randeffect,num_randeffect+1:end,j)*(covmat_theta(num_randeffect+1:end,num_randeffect+1:end,j)\covmat_theta(num_randeffect+1:end,1:num_randeffect,j));
    chol_cond_var=chol(cond_var,'lower');
    rnorm1=cond_mean+chol_cond_var*randn(num_randeffect,n1);
    rnorm2=cond_mean_ref+epsilon.*chol_cond_var*randn(num_randeffect,n2);
    chol_covmat=chol(param.theta_sig2,'lower');
    rnorm3=param.theta_mu'+chol_covmat*randn(num_randeffect,n3);
    rnorm=[rnorm1,rnorm2,rnorm3];
    rnorm=rnorm';
    else
    w_mix=0.95;
    u=rand(num_particles,1);
    id1=(u<w_mix);
    n1=sum(id1);
    n2=num_particles-n1;
    chol_covmat=chol(param.theta_sig2,'lower');
    rnorm1=reference_par'+epsilon.*chol_covmat*randn(num_randeffect,n1);
    rnorm2=param.theta_mu'+chol_covmat*randn(num_randeffect,n2);
    
    rnorm=[rnorm1,rnorm2];
    rnorm=rnorm';
    end   
    
    rnorm_theta_b1min1=rnorm(:,1);
    rnorm_theta_b2min1=rnorm(:,2);
    rnorm_theta_b3min1=rnorm(:,3);
    rnorm_theta_A1=rnorm(:,4);
    rnorm_theta_v1_1=rnorm(:,5);
    rnorm_theta_v2_1=rnorm(:,6);
    rnorm_theta_tau1=rnorm(:,7);
    
    rnorm_theta_b1min2=rnorm(:,8);
    rnorm_theta_b2min2=rnorm(:,9);
    rnorm_theta_b3min2=rnorm(:,10);
    rnorm_theta_A2=rnorm(:,11);
    rnorm_theta_v1_2=rnorm(:,12);
    rnorm_theta_v2_2=rnorm(:,13);
    rnorm_theta_tau2=rnorm(:,14);

    rnorm_theta_b1min1(1,1)=reference_par(1,1);
    rnorm_theta_b2min1(1,1)=reference_par(1,2);
    rnorm_theta_b3min1(1,1)=reference_par(1,3);
    rnorm_theta_A1(1,1)=reference_par(1,4);
    rnorm_theta_v1_1(1,1)=reference_par(1,5);
    rnorm_theta_v2_1(1,1)=reference_par(1,6);
    rnorm_theta_tau1(1,1)=reference_par(1,7);

    rnorm_theta_b1min2(1,1)=reference_par(1,8);
    rnorm_theta_b2min2(1,1)=reference_par(1,9);
    rnorm_theta_b3min2(1,1)=reference_par(1,10);
    rnorm_theta_A2(1,1)=reference_par(1,11);
    rnorm_theta_v1_2(1,1)=reference_par(1,12);
    rnorm_theta_v2_2(1,1)=reference_par(1,13);
    rnorm_theta_tau2(1,1)=reference_par(1,14);    
    
    rnorm_theta_b1min = [rnorm_theta_b1min1,rnorm_theta_b1min2];
    rnorm_theta_b2min = [rnorm_theta_b2min1,rnorm_theta_b2min2];
    rnorm_theta_b3min = [rnorm_theta_b3min1,rnorm_theta_b3min2];
    rnorm_theta_A = [rnorm_theta_A1,rnorm_theta_A2];
    rnorm_theta_v1 = [rnorm_theta_v1_1,rnorm_theta_v1_2];
    rnorm_theta_v2 = [rnorm_theta_v2_1,rnorm_theta_v2_2];
    rnorm_theta_tau = [rnorm_theta_tau1,rnorm_theta_tau2];
    
    rnorm_theta_b1min_used = rnorm_theta_b1min(:,S);
    rnorm_theta_b2min_used = rnorm_theta_b2min(:,S);
    rnorm_theta_b3min_used = rnorm_theta_b3min(:,S);
    rnorm_theta_A_used = rnorm_theta_A(:,S);
    rnorm_theta_v1_used = rnorm_theta_v1(:,S);
    rnorm_theta_v2_used = rnorm_theta_v2(:,S);
    rnorm_theta_tau_used = rnorm_theta_tau(:,S);
    
    rnorm_theta_b1min_kron=reshape(rnorm_theta_b1min_used',num_trials*num_particles,1);
    rnorm_theta_b2min_kron=reshape(rnorm_theta_b2min_used',num_trials*num_particles,1);
    rnorm_theta_b3min_kron=reshape(rnorm_theta_b3min_used',num_trials*num_particles,1);
    rnorm_theta_A_kron=reshape(rnorm_theta_A_used',num_trials*num_particles,1);
    rnorm_theta_v1_kron=reshape(rnorm_theta_v1_used',num_trials*num_particles,1);
    rnorm_theta_v2_kron=reshape(rnorm_theta_v2_used',num_trials*num_particles,1);
    rnorm_theta_tau_kron=reshape(rnorm_theta_tau_used',num_trials*num_particles,1);
     
    rnorm_theta=[rnorm_theta_b1min1,rnorm_theta_b2min1,rnorm_theta_b3min1,rnorm_theta_A1,rnorm_theta_v1_1,rnorm_theta_v2_1,rnorm_theta_tau1,...
                rnorm_theta_b1min2,rnorm_theta_b2min2,rnorm_theta_b3min2,rnorm_theta_A2,rnorm_theta_v1_2,rnorm_theta_v2_2,rnorm_theta_tau2];
     
    data_response_repmat=repmat(data.response{j,1}(:,1),num_particles,1);
    data_rt_repmat=repmat(data.rt{j,1}(:,1),num_particles,1);
    data_cond_repmat=repmat(data.cond{j,1}(:,1),num_particles,1);  
     
    [rnorm_theta_bmin_kron]=reshape_b(data_cond_repmat,rnorm_theta_b1min_kron,rnorm_theta_b2min_kron,rnorm_theta_b3min_kron);
    [rnorm_theta_v_kron]=reshape_v(data_response_repmat,rnorm_theta_v1_kron,rnorm_theta_v2_kron);
    lw=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, rnorm_theta_A_kron, rnorm_theta_bmin_kron, rnorm_theta_v_kron, ones(num_particles*num_trials(1,1),1),rnorm_theta_tau_kron)));
     
    lw_reshape=reshape(lw,num_trials,num_particles);
    logw_first=sum(lw_reshape);
    if  i>=burn+adapt+1
        logw_second=(logmvnpdf(rnorm_theta,param.theta_mu,chol_covmat*chol_covmat'));
        temp = [log(w1_mix)+logmvnpdf(rnorm_theta,cond_mean',chol_cond_var*chol_cond_var');
                log(w2_mix)+logmvnpdf(rnorm_theta,cond_mean_ref',(epsilon^2).*chol_cond_var*chol_cond_var');
                log(w3_mix)+logmvnpdf(rnorm_theta,param.theta_mu,chol_covmat*chol_covmat')];
        temp_w = exp(temp-max(temp));
        logw_third = log(sum(temp_w)) + max(temp);  
%         logw_third=log(w1_mix.*mvnpdf(rnorm_theta,cond_mean',chol_cond_var*chol_cond_var')+...
%             (w2_mix).*mvnpdf(rnorm_theta,cond_mean_ref',chol_cond_var*chol_cond_var')+...
%             (w3_mix).*mvnpdf(rnorm_theta,param.theta_mu,chol_covmat*chol_covmat'));
        logw=logw_first'+logw_second'-logw_third';
    else
        logw_second=(logmvnpdf(rnorm_theta,param.theta_mu,chol_covmat*chol_covmat'));
        temp = [log(w_mix)+logmvnpdf(rnorm_theta,reference_par,(epsilon^2).*(chol_covmat*chol_covmat'));
                log(1-w_mix)+logmvnpdf(rnorm_theta,param.theta_mu,chol_covmat*chol_covmat')];
        temp_w = exp(temp-max(temp));
        logw_third = log(sum(temp_w)) + max(temp);      
%         logw_third=log(w_mix.*mvnpdf(rnorm_theta,reference_par,(epsilon^2).*(chol_covmat*chol_covmat'))+...
%             (1-w_mix).*mvnpdf(rnorm_theta,param.theta_mu,chol_covmat*chol_covmat'));
        logw=logw_first'+logw_second'-logw_third';
    end
     
    id=imag(logw)~=0;
    id=1-id;
    id=logical(id);
    logw=logw(id,1); 
    logw=real(logw);
    rnorm_theta=rnorm_theta(id,:);
    
    if sum(isinf(logw))>0 | sum(isnan(logw))>0
     id=isinf(logw) | isnan(logw);
     id=1-id;
     id=logical(id);
     logw=logw(id,1);
     rnorm_theta=rnorm_theta(id,:);
    end
    
    max_logw=max(real(logw));
    weight=real(exp(logw-max_logw));
%     llh_i(j) = max_logw+log(mean(weight)); 
%     llh_i(j) = real(llh_i(j)); 	
    weight=weight./sum(weight);
    if sum(weight<0)>0
       id=weight<0;
       id=1-id;
       id=logical(id);
       weight=weight(id,1);
       rnorm_theta=rnorm_theta(id,:);
    end
    Nw=length(weight);
     
    if Nw>0 
        ind=randsample(Nw,1,true,weight);
        theta_latent=rnorm_theta(ind,:);
    end
        
%----------------------------------------------------------------------------------------------------------------------------------    
    
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