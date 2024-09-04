function [sirhat]=refreshed_backward_simulation_LBA_min_block_prior_rw_diffphi(particles,w,indx,param,data_response,data_rt,data_cond,num_randeffect,num_within_block)
% this function is the refreshed backward simulation algorithm adapted to
% LBA model
T=size(w,1);
N=size(w,2);
outndx=NaN(1,T);
outndx(T)=randsample(N,1,true,w(T,:));
sirhat(:,T)=particles(:,outndx(T),T);
scale_factor=1;
u=zeros(N,1);%preallocations
id1=zeros(N,1);%preallocations
id2=zeros(N,1);%preallocations

for t=T-1:-1:1
    
    indx_old=indx(t+1,outndx(t+1));%fixed the first index to the ancestor of the random effects at t+1
    indx_star=rs_multinomial(w(t,:));%obtain the resampling index according to weight of the particles at time t
    indx_temp1=indx(t,indx_old);
    indx_temp2=indx(t,indx_star(1,2:end));
    indx(t,:)=[indx_temp1,indx_temp2];
    particles_temp1=particles(:,indx_old,t);%select the particles based on 'indx_old'
    particles_temp2=particles(:,indx_star(1,2:end),t);%select the particles based on 'indx_star'
    particles(:,:,t)=[particles_temp1,particles_temp2]; 
    particles_star(:,1)=sirhat(:,t+1);
    reference_particle=sirhat(:,t+1);
    %past_reference_particle=particles(:,indx_old,t);
    w1_mix=0.8;
    w2_mix=1-w1_mix;
    u(2:end,1)=sort(rand(N-1,1));
    id1(2:end,1)=(u(2:end,1)<w1_mix);
    id2(2:end,1)=(u(2:end,1)>w1_mix) & (u(2:end,1)<=(w1_mix+w2_mix));
    n1=sum(id1);%the number of proposals for the first mixture component
    n2=sum(id2);%the number of proposals for the second mixture component
    id1=logical(id1);
    id2=logical(id2);
    particles_res(:,1:N)=particles(:,:,t);
    particles_temp1=sqrt(scale_factor)*chol(param.theta_sig2,'lower')*randn(num_randeffect,n1)+reference_particle;%generating the proposals from the first mixture component
    particles_temp2=param.theta_mu+(param.theta_phi).*(particles_res(:,id2)-param.theta_mu)+...
              chol(param.theta_sig2,'lower')*randn(num_randeffect,n2);%generating the proposals from the second mixture component
    particles_star(:,2:N)=[particles_temp1,particles_temp2];
    %list of random effects in the LBA model
    theta_latent_b1min(1,:)=particles_star(1,:);
    theta_latent_b2min(1,:)=particles_star(2,:);
    theta_latent_b3min(1,:)=particles_star(3,:);
    
    theta_latent_A(1,:)=particles_star(4,:);
    theta_latent_v1(1,:)=particles_star(5,:);
    theta_latent_v2(1,:)=particles_star(6,:);
    theta_latent_tau(1,:)=particles_star(7,:);
    %adjust the size of vector of the random effects
    theta_latent_b1min_kron=kron(theta_latent_b1min',ones(num_within_block{t+1,1},1));
    theta_latent_b2min_kron=kron(theta_latent_b2min',ones(num_within_block{t+1,1},1));
    theta_latent_b3min_kron=kron(theta_latent_b3min',ones(num_within_block{t+1,1},1));
    
    theta_latent_A_kron=kron(theta_latent_A',ones(num_within_block{t+1,1},1));
    theta_latent_v1_kron=kron(theta_latent_v1',ones(num_within_block{t+1,1},1));
    theta_latent_v2_kron=kron(theta_latent_v2',ones(num_within_block{t+1,1},1));
    theta_latent_tau_kron=kron(theta_latent_tau',ones(num_within_block{t+1,1},1));
    %adjust the size of the dataset
    data_response_repmat=repmat(data_response{t+1,1}(:,1),N,1);
    data_rt_repmat=repmat(data_rt{t+1,1}(:,1),N,1);
    data_cond_repmat=repmat(data_cond{t+1,1}(:,1),N,1);

    [theta_latent_bmin_kron]=reshape_b(data_cond_repmat,theta_latent_b1min_kron,theta_latent_b2min_kron,theta_latent_b3min_kron);% choose the threshold particles to match with the  conditions of the experiments at block (t+1)
    [theta_latent_v_kron]=reshape_v(data_response_repmat,theta_latent_v1_kron,theta_latent_v2_kron);% set the drift rate particles to match with the response at block (t+1)
    %computing the log weights
    if t==T-1
       log_weight_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(N*num_within_block{t+1,1},1),theta_latent_tau_kron)));
       log_weight_reshape=reshape(log_weight_temp,num_within_block{t+1,1},N);
       log_weight1=sum(log_weight_reshape);
       mean_com_temp2=(param.theta_mu+(param.theta_phi).*(particles(:,:,t)-param.theta_mu))';
       %mean_com_temp3=(param.theta_mu+(param.theta_phi*ones(num_randeffect,1)).*(past_reference_particle*ones(1,N)-param.theta_mu))';
       log_weight2=(logmvnpdf(particles_star',mean_com_temp2,param.theta_sig2))';
       log_weight3=log(w1_mix.*mvnpdf(particles_star',reference_particle',scale_factor.*param.theta_sig2)+...
                    w2_mix.*mvnpdf(particles_star',mean_com_temp2,param.theta_sig2));    
       log_weight=log_weight1'+log_weight2-log_weight3;
             
    else
       log_weight_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(N*num_within_block{t+1,1},1),theta_latent_tau_kron)));
       log_weight_reshape=reshape(log_weight_temp,num_within_block{t+1,1},N);
       log_weight1=sum(log_weight_reshape);
       mean_com2=(param.theta_mu+(param.theta_phi).*(particles_star-param.theta_mu))';
       mean_com3=(param.theta_mu+(param.theta_phi).*(particles(:,:,t)-param.theta_mu))';
      %mean_com4=(param.theta_mu+(param.theta_phi*ones(num_randeffect,1)).*(past_reference_particle*ones(1,N)-param.theta_mu))';
       log_weight2=(logmvnpdf(sirhat(:,t+2)',mean_com2,param.theta_sig2))';
       log_weight3=(logmvnpdf(particles_star',mean_com3,param.theta_sig2))';
       log_weight4=log(w1_mix.*mvnpdf(particles_star',reference_particle',scale_factor.*param.theta_sig2)+...
                    w2_mix.*mvnpdf(particles_star',mean_com3,param.theta_sig2));  
       log_weight=log_weight1'+log_weight2+log_weight3-log_weight4;
             
                   
     end
     weight=exp(log_weight-max(log_weight));   
     weight=weight./sum(weight);
     indx_choose=find(rand(1) < cumsum(weight),1,'first');
     sirhat(:,t+1)=particles_star(:,indx_choose);
     sirhat(:,t)=particles(:,indx_choose,t);
     outndx(t)=indx_choose;

end

end

