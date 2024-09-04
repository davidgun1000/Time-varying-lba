function [particles,w,indx]=LBA_CSMC_prior_rw_min_block_v3_diffphi(data_response,data_rt,data_cond,param,theta_latent,num_subjects,num_block,num_within_block,num_particles,num_randeffect)
    % this is the function for conditional Sequential Monte Carlo algorithm
    % with the mixture proposal from the prior density p(\alpha_t|\theta) and
    % 'random walk' type. 
    u=zeros(num_particles,1);%preallocations
    id1=zeros(num_particles,1);%preallocations
    id2=zeros(num_particles,1);%preallocations
    scale_factor1=1;
    scale_factor2=1;
    w1_mix=0.1;%set the mixture weights
    w2_mix=1-w1_mix;
    t=1;
    particles=zeros(num_randeffect,num_particles,num_block);%preallocations for the particles
    w=zeros(num_block,num_particles);%preallocations for the weights
    indx=zeros(num_block,num_particles);%preallocations for the index
    particles(:,1,:)=theta_latent; %set the first particles to the values of random effects from the previous iterations of
                                   %the MCMC for conditioning, for all
                                   %blocks.
    reference_particle=theta_latent(:,t);%the set of random effects from previous iteration of MCMC for block t.
    %generating the proposals from the mixture distribution at the burn in
    %and initial sampling stage at time t
    u(2:end,1)=sort(rand(num_particles-1,1));
    id1(2:end,1)=(u(2:end,1)<w1_mix);
    id2(2:end,1)=(u(2:end,1)>w1_mix) & (u(2:end,1)<=(w1_mix+w2_mix));
    n1=sum(id1);%the number of proposals from the first component
    n2=sum(id2);%the number of proposals from the second component
    id1=logical(id1);
    id2=logical(id2);
    
    particles_temp1=(mvnrnd(param.theta_mu',(param.theta_sig2),n1))';%generate proposals from the first component
    particles_temp2=(mvnrnd(reference_particle',scale_factor1*(param.theta_sig2),n2))';%generate proposals from the second component
    particles(:,2:num_particles,t)=[particles_temp1,particles_temp2];
    %list of random effects of the LBA model, we have 7 random effects for
    %Forstmann dataset
    theta_latent_b1min(1,:)=particles(1,:,t);
    theta_latent_b2min(1,:)=particles(2,:,t);
    theta_latent_b3min(1,:)=particles(3,:,t);
    theta_latent_A(1,:)=particles(4,:,t);
    theta_latent_v1(1,:)=particles(5,:,t);
    theta_latent_v2(1,:)=particles(6,:,t);
    theta_latent_tau(1,:)=particles(7,:,t);
    %adjust the size of the vectors of random effects
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
    
    [theta_latent_bmin_kron]=reshape_b(data_cond_repmat,theta_latent_b1min_kron,theta_latent_b2min_kron,theta_latent_b3min_kron);% choose the threshold particles to match with the  conditions of the experiments at block t
    [theta_latent_v_kron]=reshape_v(data_response_repmat,theta_latent_v1_kron,theta_latent_v2_kron);% set the drift rate particles to match with the response at block t
    %computing the log of weights
    %-----------------------------------------
    logw_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(num_particles*num_within_block{t,1},1),theta_latent_tau_kron)));
    logw_reshape=reshape(logw_temp,num_within_block{t,1},num_particles);
    logw_first=sum(logw_reshape);
    logw_second=(logmvnpdf(particles(:,:,t)',param.theta_mu',param.theta_sig2))';
    logw_third=log(w1_mix.*mvnpdf(particles(:,:,t)',param.theta_mu',(param.theta_sig2))+...
            w2_mix.*mvnpdf(particles(:,:,t)',reference_particle',scale_factor1*(param.theta_sig2)));    
    logw=logw_first'+logw_second-logw_third;
    w(t,:)=exp(logw-max(logw));
    w(t,:)=w(t,:)/sum(w(t,:));%compute the normalised weights
    %------------------------------------------
    
    for t=2:num_block
        indx(t,:)=rs_multinomial_cond(w(t-1,:));%resample the particles to avoid degeneracy
        reference_particle=theta_latent(:,t);%the set of random effects from previous iteration of MCMC for block t.
        past_reference_particle=theta_latent(:,t-1);
        
        w1_mix=0.2;
        w2_mix=1-w1_mix;
        %generating the proposals from the mixture distribution at the burn in
        %and initial sampling stage as outlined the manuscript at time t.
        %-------------------------------
        u(2:end,1)=sort(rand(num_particles-1,1));
        id1(2:end,1)=(u(2:end,1)<w1_mix);
        id2(2:end,1)=(u(2:end,1)>w1_mix) & (u(2:end,1)<=(w1_mix+w2_mix));
        
        n1=sum(id1);
        n2=sum(id2);
        id1=logical(id1);
        id2=logical(id2);
        particles_res(:,1:num_particles)=particles(:,indx(t,1:num_particles),t-1);
        particles_temp1=param.theta_mu+(param.theta_phi).*(particles_res(:,id1)-param.theta_mu)+...
                 chol(param.theta_sig2,'lower')*randn(num_randeffect,n1);
        
        particles_temp2=reference_particle+sqrt(scale_factor2)*chol(param.theta_sig2,'lower')*randn(num_randeffect,n2);
        particles(:,2:num_particles,t)=[particles_temp1,particles_temp2];
        %-----------------------------
        %list of random effects of the LBA model, we have 7 random effects for
        %Forstmann dataset
        
        theta_latent_b1min(1,:)=particles(1,:,t);
        theta_latent_b2min(1,:)=particles(2,:,t);
        theta_latent_b3min(1,:)=particles(3,:,t);
        
        theta_latent_A(1,:)=particles(4,:,t);
        theta_latent_v1(1,:)=particles(5,:,t);
        theta_latent_v2(1,:)=particles(6,:,t);
        theta_latent_tau(1,:)=particles(7,:,t);
        %adjust the size of the vectors of random effects
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
    
        [theta_latent_bmin_kron]=reshape_b(data_cond_repmat,theta_latent_b1min_kron,theta_latent_b2min_kron,theta_latent_b3min_kron);% choose the threshold particles to match with the  conditions of the experiments at block t
        [theta_latent_v_kron]=reshape_v(data_response_repmat,theta_latent_v1_kron,theta_latent_v2_kron);% set the drift rate particles to match with the response at block t
        %computing the log of weights
        %-----------------------------------------
        logw_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(num_particles*num_within_block{t,1},1),theta_latent_tau_kron)));
        logw_reshape=reshape(logw_temp,num_within_block{t,1},num_particles);
        logw_first=sum(logw_reshape);
        mean_com2=(param.theta_mu+(param.theta_phi).*(particles(:,indx(t,1:num_particles),t-1)-param.theta_mu))'; 
        logw_second=(logmvnpdf(particles(:,:,t)',mean_com2,param.theta_sig2))';
        
        
        logw_third=log(w1_mix.*mvnpdf(particles(:,:,t)',mean_com2,param.theta_sig2)+...
            w2_mix.*mvnpdf(particles(:,:,t)',reference_particle',scale_factor2*param.theta_sig2));
        logw=logw_first'+logw_second-logw_third;
        w(t,:)=exp(logw-max(logw));
        w(t,:)=w(t,:)/sum(w(t,:));  
    end

end
