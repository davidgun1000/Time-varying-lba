function [theta_latent]=LBA_CSMC_prior_prop_deterministic(data_response,data_rt,data_cond,covariates,param,theta_latent,num_subjects,num_block,num_within_block,num_particles,num_randeffect,...
    mean_param,covmat_param)
    %this is the function for conditional Monte Carlo algorithm for TV-LBA
    %(II)
    %with the mixture proposal for the sampling stage as in Gunawan et al
    %(2019)
    u=zeros(num_particles,1); % preallocations
    id1=zeros(num_particles,1); % preallocations
    id2=zeros(num_particles,1); % preallocations
    id3=zeros(num_particles,1); % preallocations
    %set the mixture weights
    %-----------------------
    w1_mix=0.8;
    w2_mix=0.1;
    w3_mix=1-w1_mix-w2_mix;
    %-----------------------
    particles=zeros(num_randeffect,num_particles,num_block); % preallocations for the particles
    w=zeros(num_block,num_particles); % preallocations for the weights
    particles(:,1,:)=theta_latent;%set the first particles to the values of random effects from the previous iterations of
                                   %the MCMC for conditioning, for all
                                   %blocks.
    
    for t=1:num_block
    % generating the proposals for random effects from the mixture
    % distribution in the sampling stage as outlined in the Gunawan et al
    %(2019)
    u(2:end,1)=sort(rand(num_particles-1,1));
    id1(2:end,1)=(u(2:end,1)<w1_mix);
    id2(2:end,1)=(u(2:end,1)>w1_mix) & (u(2:end,1)<=(w1_mix+w2_mix));
    id3(2:end,1)=(u(2:end,1)>(w1_mix+w2_mix)) & (u(2:end,1)<=(w1_mix+w2_mix+w3_mix));
    n1=sum(id1);%the number of proposals from the first components
    n2=sum(id2);%the number of proposals from the second components
    n3=sum(id3);%the number of proposals from the third components
    id1=logical(id1);
    id2=logical(id2);
    id3=logical(id3);
    reference_particle=theta_latent(:,t);%the set of random effects from previous iteration of MCMC for block t.
    scale_covmat=1;
    chol_theta_sig2=chol(param.theta_sig2,'lower');% the cholesky factorisation of the covariance matrix \Sigma_{\alpha} needed to construct the proposal for the random effects
    chol_theta_sig2_1=log(chol_theta_sig2(1,1));
    chol_theta_sig2_2=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
    chol_theta_sig2_3=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
    chol_theta_sig2_4=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
    chol_theta_sig2_5=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
    chol_theta_sig2_6=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
    chol_theta_sig2_7=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];
    %the vector xx below contains the parameter \beta, the cholesky
    %factor of covariance matrix \Sigma_{\alpha}
    xx=[param.theta_beta;chol_theta_sig2_1';chol_theta_sig2_2';chol_theta_sig2_3';chol_theta_sig2_4';
        chol_theta_sig2_5';chol_theta_sig2_6';chol_theta_sig2_7']; % we need this to compute the conditional mean of the proposal for the random  effects in the sampling stage at time t
    cond_mean=mean_param(t,1:num_randeffect)'+covmat_param(1:num_randeffect,num_randeffect+1:end,t)*((covmat_param(num_randeffect+1:end,num_randeffect+1:end,t))\(xx-mean_param(t,num_randeffect+1:end)')); %computing the mean of the proposal for the random effects at time t for the first mixture component
    cond_var=covmat_param(1:num_randeffect,1:num_randeffect,t)-covmat_param(1:num_randeffect,num_randeffect+1:end,t)*(covmat_param(num_randeffect+1:end,num_randeffect+1:end,t)\covmat_param(num_randeffect+1:end,1:num_randeffect,t)); % compute the covariance matrix for the proposal at time t for the first and thirt mixture components
    cond_var=topdm(cond_var);
    chol_cond_var=chol(cond_var,'lower');
    cov_times_beta=covariates(:,:,t)*param.theta_beta;
    particles_temp1=chol_cond_var*randn(num_randeffect,n1)+cond_mean; %generating the proposal from the first components
    particles_temp2=(mvnrnd(cov_times_beta',(param.theta_sig2),n2))'; %generating the proposal from the second components: the prior of random effects
    particles_temp3=sqrt(scale_covmat)*chol_cond_var*randn(num_randeffect,n3)+reference_particle; %generating the proposal from the third components
    particles(:,2:num_particles,t)=[particles_temp1,particles_temp2,particles_temp3];
    %------------------------------------------
    
    %list of the random effects of the LBA model, we have 7 random effects
    %for the Forstmann dataset  
    theta_latent_b1min(1,:)=particles(1,:,t);
    theta_latent_b2min(1,:)=particles(2,:,t);
    theta_latent_b3min(1,:)=particles(3,:,t);
    theta_latent_A(1,:)=particles(4,:,t);
    theta_latent_v1(1,:)=particles(5,:,t);
    theta_latent_v2(1,:)=particles(6,:,t);
    theta_latent_tau(1,:)=particles(7,:,t);
    
    %adjust the size of the vector of random effects
    theta_latent_b1min_kron=kron(theta_latent_b1min',ones(num_within_block{t,1},1));
    theta_latent_b2min_kron=kron(theta_latent_b2min',ones(num_within_block{t,1},1));
    theta_latent_b3min_kron=kron(theta_latent_b3min',ones(num_within_block{t,1},1));
    theta_latent_A_kron=kron(theta_latent_A',ones(num_within_block{t,1},1));
    theta_latent_v1_kron=kron(theta_latent_v1',ones(num_within_block{t,1},1));
    theta_latent_v2_kron=kron(theta_latent_v2',ones(num_within_block{t,1},1));
    theta_latent_tau_kron=kron(theta_latent_tau',ones(num_within_block{t,1},1));
    
    %adjust the size of the dataset.
    data_response_repmat=repmat(data_response{t,1}(:,1),num_particles,1);
    data_rt_repmat=repmat(data_rt{t,1}(:,1),num_particles,1);
    data_cond_repmat=repmat(data_cond{t,1}(:,1),num_particles,1);
    
    [theta_latent_bmin_kron]=reshape_b(data_cond_repmat,theta_latent_b1min_kron,theta_latent_b2min_kron,theta_latent_b3min_kron); % choose the threshold particles to match with the  conditions of the experiments at block t
    [theta_latent_v_kron]=reshape_v(data_response_repmat,theta_latent_v1_kron,theta_latent_v2_kron); % set the drift rate particles to match with the response at block t
    
    %compute the log weights
    %------------------------
    logw_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(num_particles*num_within_block{t,1},1),theta_latent_tau_kron)));
    logw_reshape=reshape(logw_temp,num_within_block{t,1},num_particles);
    logw_first=sum(logw_reshape);
    logw_second=(logmvnpdf(particles(:,:,t)',cov_times_beta',param.theta_sig2))';
    logw_third=log(w1_mix.*mvnpdf(particles(:,:,t)',cond_mean',chol_cond_var*chol_cond_var')+...
            w2_mix.*mvnpdf(particles(:,:,t)',cov_times_beta',param.theta_sig2)+...
            w3_mix.*mvnpdf(particles(:,:,t)',reference_particle',scale_covmat*(chol_cond_var*chol_cond_var')));    
    logw=logw_first'+logw_second-logw_third;
     %------------------------
    w(t,:)=exp(logw-max(logw));
    w(t,:)=w(t,:)/sum(w(t,:));%compute the normalised weights
    
    ind=randsample(num_particles,1,true,w(t,:)); %choose one index at random based on weight at time t
    theta_latent(:,t)=particles(:,ind,t); %choose the random effect based on index 'ind'
    end
end



%     for t=2:num_block
%         indx(t,:)=rs_multinomial_cond(w(t-1,:));
%         reference_particle=theta_latent(:,t);
%         u(2:end,1)=sort(rand(num_particles-1,1));
%         id1(2:end,1)=(u(2:end,1)<w1_mix);
%         id2(2:end,1)=(u(2:end,1)>w1_mix) & (u(2:end,1)<=(w1_mix+w2_mix));
%         id3(2:end,1)=(u(2:end,1)>(w1_mix+w2_mix)) & (u(2:end,1)<=(w1_mix+w2_mix+w3_mix));
%         n1=sum(id1);
%         n2=sum(id2);
%         n3=sum(id3);
%         id1=logical(id1);
%         id2=logical(id2);
%         id3=logical(id3);
%         particles_res(:,1:num_particles)=particles(:,indx(t,1:num_particles),t-1);
%              
%         xx1=[particles_res(:,1:num_particles);param.theta_mu*ones(1,num_particles);
%             chol_theta_sig2_1'*ones(1,num_particles);chol_theta_sig2_2'*ones(1,num_particles);
%             chol_theta_sig2_3'*ones(1,num_particles);chol_theta_sig2_4'*ones(1,num_particles);
%             chol_theta_sig2_5'*ones(1,num_particles);chol_theta_sig2_6'*ones(1,num_particles);
%             chol_theta_sig2_7'*ones(1,num_particles);logit_inverse(param.theta_phi)*ones(1,num_particles)];
%         xx1_reshape=reshape(xx1,43,1,num_particles);
%         xx1_minus_meanparam=xx1_reshape-mean_param(t,8:end)';
%         temp1=multiprod((covmat_param(1:7,8:end,t)/covmat_param(8:end,8:end,t)),xx1_minus_meanparam);
%         temp1_reshape=reshape(temp1,7,num_particles);
%         cond_mean=mean_param(t,1:7)'+temp1_reshape;
%         cond_mean_ref=reference_particle+temp1_reshape;
%         covmat_param(:,:,t)=topdm(covmat_param(:,:,t));
%         cond_var=covmat_param(1:7,1:7,t)-covmat_param(1:7,8:end,t)*(covmat_param(8:end,8:end,t)\covmat_param(8:end,1:7,t));
%         cond_var=topdm(cond_var);
%         chol_cond_var=chol(real(cond_var),'lower');
%         particles_temp1=chol_cond_var*randn(num_randeffect,n1)+cond_mean(:,id1);
%         particles_temp2=param.theta_mu+(param.theta_phi*ones(num_randeffect,1)).*(particles_res(:,id2)-param.theta_mu)+...
%                  chol(param.theta_sig2,'lower')*randn(num_randeffect,n2);
%         particles_temp3=sqrt(scale_covmat)*chol_cond_var*randn(num_randeffect,n3)+cond_mean_ref(:,id3);
%         particles(:,2:num_particles,t)=[particles_temp1,particles_temp2,particles_temp3];
%         
%         theta_latent_b1min(1,:)=particles(1,:,t);
%         theta_latent_b2min(1,:)=particles(2,:,t);
%         theta_latent_b3min(1,:)=particles(3,:,t);
%         theta_latent_A(1,:)=particles(4,:,t);
%         theta_latent_v1(1,:)=particles(5,:,t);
%         theta_latent_v2(1,:)=particles(6,:,t);
%         theta_latent_tau(1,:)=particles(7,:,t);
%         
%         theta_latent_b1min_kron=kron(theta_latent_b1min',ones(num_within_block{t,1},1));
%         theta_latent_b2min_kron=kron(theta_latent_b2min',ones(num_within_block{t,1},1));
%         theta_latent_b3min_kron=kron(theta_latent_b3min',ones(num_within_block{t,1},1));
%         
%         theta_latent_A_kron=kron(theta_latent_A',ones(num_within_block{t,1},1));
%         theta_latent_v1_kron=kron(theta_latent_v1',ones(num_within_block{t,1},1));
%         theta_latent_v2_kron=kron(theta_latent_v2',ones(num_within_block{t,1},1));
%         theta_latent_tau_kron=kron(theta_latent_tau',ones(num_within_block{t,1},1));
%     
%         data_response_repmat=repmat(data_response{t,1}(:,1),num_particles,1);
%         data_rt_repmat=repmat(data_rt{t,1}(:,1),num_particles,1);
%         data_cond_repmat=repmat(data_cond{t,1}(:,1),num_particles,1);
%     
%         [theta_latent_bmin_kron]=reshape_b(data_cond_repmat,theta_latent_b1min_kron,theta_latent_b2min_kron,theta_latent_b3min_kron);
%         [theta_latent_v_kron]=reshape_v(data_response_repmat,theta_latent_v1_kron,theta_latent_v2_kron);
%         
%         logw_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(num_particles*num_within_block{t,1},1),theta_latent_tau_kron)));
%         logw_reshape=reshape(logw_temp,num_within_block{t,1},num_particles);
%         logw_first=sum(logw_reshape);
%         mean_com2=(param.theta_mu+(param.theta_phi*ones(num_randeffect,1)).*(particles(:,indx(t,1:num_particles),t-1)-param.theta_mu))'; 
%         logw_second=(logmvnpdf(particles(:,:,t)',mean_com2,param.theta_sig2))';
%         %logw_third=(logmvnpdf(particles(:,:,t)',cond_mean',scale_covmat*(chol_cond_var*chol_cond_var')))';
%         logw_third=log(w1_mix.*mvnpdf(particles(:,:,t)',cond_mean',chol_cond_var*chol_cond_var')+...
%             w2_mix.*mvnpdf(particles(:,:,t)',mean_com2,param.theta_sig2)+...
%             w3_mix.*mvnpdf(particles(:,:,t)',cond_mean_ref',scale_covmat*(chol_cond_var*chol_cond_var')));    
%         logw=logw_first'+logw_second-logw_third;
%         w(t,:)=exp(logw-max(logw));
%         w(t,:)=w(t,:)/sum(w(t,:));  
%     end
