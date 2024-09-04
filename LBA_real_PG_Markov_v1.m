%estimating the Time-varying hierarchical LBA (AR) model using PMwG method for the Forstmann (2008) dataset
%The AR model specification can be found in the paper. The data is stored
%in the Matlab file 'LBA_realdata_Forstmann_block.mat', It has three
%components: data.cond contains the conditions of the experiments, data.rt
%contains the response time, and data.response: response=1 for incorrect
%response and response=2 for correct response.
load('LBA_realdata_Fortsmann_block.mat'); %load the dataset
num_subjects=length(data.rt); %number of subjects
for j=1:num_subjects
    num_block(j,1)=length(data.rt{j,1});%computing the number of block for each subject
    for k=1:num_block(j,1)
    num_within_block{j,1}{k,1}=length(data.rt{j,1}{k,1});%computing the number of trials within each block for each subject 
    end
end
num_particles=250; %number of particles used in the conditional sequential Monte Carlo algorithm
%parpool(28)%number of multi-processors used


num_randeffect=7;%the number of random effects in the LBA model
param.theta_mu=[0.2484,0.5481,0.1,-0.3260,-0.1182,1.0858,-1.9437]'; % the initial values for the parameter \mu
param.theta_sig2=iwishrnd(eye(num_randeffect),20); %the initial values for the parameter \Sigma
param.theta_phi=0.9*ones(num_randeffect,1);  %initial value for the parameter \phi
param.sv=1; % the standard deviation of the drift rate and is assumed to be 1
param.num_randeffect=7;% the number of random effects in the LBA model 


num_choice=2;%the number of choice in the dataset
burn=1000;%the burn in replications
adapt=4000;%the initial sampling stage
sampling=20000;%the sampling stage
s=burn+adapt+sampling; %the total number of iterations

prior.mu_mean=zeros(num_randeffect,1); %the prior mean for \mu
prior.mu_sig2=eye(num_randeffect);%the prior variance for \mu
prior.a_phi=20;%the prior parameter for \phi_
prior.b_phi=1.5;%the prior parameter for \phi_
prior.v0_sigma=20;%the prior parameter for \Sigma
prior.s0_sigma=eye(num_randeffect);%the prior parameter for \Sigma

% generating initial trajectory for the random effect for each subject
for j=1:num_subjects
    theta_latent{j,1}(:,1)=(mvnrnd(param.theta_mu,param.theta_sig2))';
    for k=2:num_block(j,1)
        theta_latent{j,1}(:,k)=param.theta_mu+param.theta_phi.*(theta_latent{j,1}(:,k-1)-param.theta_mu)+chol(param.theta_sig2,'lower')*randn(num_randeffect,1); 
    end
end

temp=1;
i=1;

while i<=s
    i
    %tic
    
    %sample \mu from the full conditional distribution.
       theta_sig2_inv=inv(param.theta_sig2);
       var_mu=inv(num_subjects*theta_sig2_inv+sum(num_block-1)*((eye(num_randeffect)-diag(param.theta_phi))*theta_sig2_inv*(eye(num_randeffect)-diag(param.theta_phi)))+inv(prior.mu_sig2));
       chol_var_mu=chol(var_mu,'lower');
       term1=zeros(num_randeffect,1);
       term2=zeros(num_randeffect,1);
       for j=1:num_subjects
           term1=term1+theta_sig2_inv*theta_latent{j,1}(:,1);
           repmat_sig2=repmat(theta_sig2_inv,1,1,num_block(j,1)-1);
           current_theta_latent=reshape(theta_latent{j,1}(:,2:num_block(j,1)),num_randeffect,1,num_block(j,1)-1);
           lag_theta_latent=reshape(theta_latent{j,1}(:,1:num_block(j,1)-1),num_randeffect,1,num_block(j,1)-1);
           repmat_diag_phi=repmat(diag(param.theta_phi),1,1,num_block(j,1)-1);
           repmat_eye_minus_diag_phi=repmat(eye(num_randeffect)-diag(param.theta_phi),1,1,num_block(j,1)-1);
           term2_1=multiprod(multiprod(repmat_eye_minus_diag_phi,repmat_sig2),current_theta_latent);
           term2_2=multiprod(multiprod(repmat_eye_minus_diag_phi,repmat_sig2),multiprod(repmat_diag_phi,lag_theta_latent));
           term2=term2+sum(term2_1-term2_2,3);
                  
       end
       mu_temp=(term1+term2);
       mean_mu=var_mu*mu_temp;
       param.theta_mu=(mvnrnd(mean_mu',chol_var_mu*chol_var_mu'))';

    %sample \Sigma from the full conditional distribution.
     k_half=prior.v0_sigma+sum(num_block);
     cov_temp=zeros(num_randeffect,num_randeffect);
     for j=1:num_subjects
         theta_latent_temp1=(theta_latent{j,1}(:,1)-param.theta_mu);
         theta_latent_temp2=(theta_latent{j,1}(:,2:end)-param.theta_mu-(theta_latent{j,1}(:,1:end-1)-param.theta_mu).*param.theta_phi);
         temp1=(theta_latent_temp1*theta_latent_temp1');
         temp2=theta_latent_temp2*theta_latent_temp2';
         cov_temp=cov_temp+temp1+temp2;
     end
     B_half=prior.s0_sigma+cov_temp;
     param.theta_sig2=iwishrnd(B_half,k_half);

%------------------------------------------------------------------------------------------------------------    
    
    %sample \phi from full conditional distribution

     theta_sig2_inv=inv(param.theta_sig2);
     var_phi_temp=0;
     mean_phi_temp=0;
     for j=1:num_subjects
         phi_temp1=theta_latent{j,1}(:,1:num_block(j,1)-1)-param.theta_mu;
         phi_temp2=reshape(phi_temp1,num_randeffect,1,(num_block(j,1)-1));
         for t=1:(num_block(j,1)-1)
            phi_temp2_diag{j,1}(:,:,t)=diag(phi_temp2(:,:,t));
         end
         repmat_sig2=repmat(theta_sig2_inv,1,1,num_block(j,1)-1);
         mult_temp1=multiprod(phi_temp2_diag{j,1},repmat_sig2);
         mult_temp2=sum(multiprod(mult_temp1,phi_temp2_diag{j,1}),3);
         var_phi_temp=var_phi_temp+mult_temp2;         
         phi_mean2_reshape=reshape(theta_latent{j,1}(:,2:end),num_randeffect,1,num_block(j,1)-1);
         mult_temp1_mean=multiprod(repmat_sig2,phi_mean2_reshape);
         mult_temp2_mean=multiprod(phi_temp2_diag{j,1},mult_temp1_mean);
         theta_mu_reshape=repmat(param.theta_mu,1,1,(num_block(j,1)-1));
         mult_temp3_mean=multiprod(repmat_sig2,theta_mu_reshape);
         mult_temp4_mean=multiprod(phi_temp2_diag{j,1},mult_temp3_mean);
         mean_phi_temp=mean_phi_temp+(sum(mult_temp2_mean-mult_temp4_mean,3));
         
     end
     var_phi=inv(var_phi_temp);
     var_phi=topdm(var_phi);
     mean_phi=var_phi*mean_phi_temp;
     phi_star=(mvnrnd(mean_phi',var_phi))';
     if sum(phi_star<0.999)==num_randeffect & sum(phi_star>-0.999)==num_randeffect
        param.theta_phi=phi_star;
     end


%--------------------------------------------------------------------------------------------------------
    %%conditional Sequential Monte Carlo algorithm to generate trajectory of
    %random effects for each subject
    %if we are in the burn in and initial adaptation stage, we use less
    %efficient prior that is based on the mixture of prior of random
    %effects and 'random walk', otherwise we use better proposal as
    %outlined in the paper
     if i<=burn+adapt
         parfor j=1:num_subjects    
              [X,W,A]=LBA_CSMC_prior_rw_min_block_v3_diffphi(data.response{j,1},data.rt{j,1},data.cond{j,1},param,theta_latent{j,1}(:,:,1),num_subjects,num_block(j,1),num_within_block{j,1},num_particles,num_randeffect);
              [theta_latent{j,1}(:,:,1)]=refreshed_backward_simulation_LBA_min_block_prior_rw_diffphi(X,W,A,param,data.response{j,1},data.rt{j,1},data.cond{j,1},num_randeffect,num_within_block{j,1});          
         end
     end
      
     if i>=burn+adapt+1
           parfor j=1:num_subjects
%       
            [X,W,A]=LBA_CSMC_prior_prop_min_block3_v3_diffphi(data.response{j,1},data.rt{j,1},data.cond{j,1},param,theta_latent{j,1}(:,:,1),num_subjects,num_block(j,1),num_within_block{j,1},num_particles,num_randeffect,...
            mean_param_1{j,1},covmat_param_1{j,1},mean_param{j,1},covmat_param{j,1});
            [theta_latent{j,1}(:,:,1)]=refreshed_backward_simulation_LBA_min_block_prop_prior3_diffphi(X,W,A,param,data.response{j,1},data.rt{j,1},data.cond{j,1},num_randeffect,num_within_block{j,1},...
            mean_param{j,1},covmat_param{j,1});    
           end
     end
      
      if i>burn
      chol_theta_sig2=chol(param.theta_sig2,'lower');
      chol_theta_sig2_store1(i-burn,:)=log(chol_theta_sig2(1,1));
      chol_theta_sig2_store2(i-burn,:)=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
      chol_theta_sig2_store3(i-burn,:)=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
      chol_theta_sig2_store4(i-burn,:)=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
      chol_theta_sig2_store5(i-burn,:)=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
      chol_theta_sig2_store6(i-burn,:)=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
      chol_theta_sig2_store7(i-burn,:)=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];
      
      theta_mu_store(i-burn,:)=param.theta_mu';
      theta_sig2_store1(i-burn,:)=param.theta_sig2(1,:);
      theta_sig2_store2(i-burn,:)=param.theta_sig2(2,2:end);
      theta_sig2_store3(i-burn,:)=param.theta_sig2(3,3:end);
      theta_sig2_store4(i-burn,:)=param.theta_sig2(4,4:end);
      theta_sig2_store5(i-burn,:)=param.theta_sig2(5,5:end);
      theta_sig2_store6(i-burn,:)=param.theta_sig2(6,6:end);
      theta_sig2_store7(i-burn,:)=param.theta_sig2(7,7:end);
      
      theta_phi_store(i-burn,:)=param.theta_phi;
      
      
       for j=1:num_subjects
          theta_latent_b1_store{j,1}(i-burn,:)=theta_latent{j,1}(1,:);
          theta_latent_b2_store{j,1}(i-burn,:)=theta_latent{j,1}(2,:);
          theta_latent_b3_store{j,1}(i-burn,:)=theta_latent{j,1}(3,:);          
          theta_latent_A_store{j,1}(i-burn,:)=theta_latent{j,1}(4,:);
          theta_latent_v1_store{j,1}(i-burn,:)=theta_latent{j,1}(5,:);
          theta_latent_v2_store{j,1}(i-burn,:)=theta_latent{j,1}(6,:);
          theta_latent_tau_store{j,1}(i-burn,:)=theta_latent{j,1}(7,:);  

      end
      end
      
      %obtain proposals
       if i>=burn+adapt & mod(i,100)==0 
       for j=1:num_subjects
           length_draws=length(theta_mu_store);
           for t=1:num_block(j,1)
           if t==1
             xx=[theta_latent_b1_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_b2_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_b3_store{j,1}(length_draws-(adapt-1):length_draws,t),...
                 theta_latent_A_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_v1_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_v2_store{j,1}(length_draws-(adapt-1):length_draws,t),...
                 theta_latent_tau_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_mu_store(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store1(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store2(length_draws-(adapt-1):length_draws,:),...
                 chol_theta_sig2_store3(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store4(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store5(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store6(length_draws-(adapt-1):length_draws,:),...
                 chol_theta_sig2_store7(length_draws-(adapt-1):length_draws,:),logit_inverse_min1_to1(theta_phi_store(length_draws-(adapt-1):length_draws,:))];
             mean_param_1{j,1}=mean(xx);
             cov_temp=cov(xx);
             cov_temp=topdm(cov_temp);
             cov_temp=chol(cov_temp,'lower');
             covmat_param_1{j,1}=cov_temp*cov_temp';
             
           else
             xx=[theta_latent_b1_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_b2_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_b3_store{j,1}(length_draws-(adapt-1):length_draws,t),...
                 theta_latent_A_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_v1_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_v2_store{j,1}(length_draws-(adapt-1):length_draws,t),...
                 theta_latent_tau_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_b1_store{j,1}(length_draws-(adapt-1):length_draws,t-1),theta_latent_b2_store{j,1}(length_draws-(adapt-1):length_draws,t-1),...
                 theta_latent_b3_store{j,1}(length_draws-(adapt-1):length_draws,t-1),theta_latent_A_store{j,1}(length_draws-(adapt-1):length_draws,t-1),theta_latent_v1_store{j,1}(length_draws-(adapt-1):length_draws,t-1),theta_latent_v2_store{j,1}(length_draws-(adapt-1):length_draws,t-1),...
                 theta_latent_tau_store{j,1}(length_draws-(adapt-1):length_draws,t-1),theta_mu_store(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store1(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store2(length_draws-(adapt-1):length_draws,:),...
                 chol_theta_sig2_store3(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store4(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store5(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store6(length_draws-(adapt-1):length_draws,:),...
                 chol_theta_sig2_store7(length_draws-(adapt-1):length_draws,:),logit_inverse_min1_to1(theta_phi_store(length_draws-(adapt-1):length_draws,:))];
             mean_param{j,1}(t,:)=mean(xx);
             cov_temp=cov(xx);
             cov_temp=topdm(cov_temp);
             cov_temp=chol(cov_temp,'lower');
             covmat_param{j,1}(:,:,t)=cov_temp*cov_temp';
           end     
           end
       end
       end
      

      
    %save the output to your directory
 
         if mod(i,2000)==0   
            save('/short/jz21/dg2271/LBA_real_forstmanndata_PG_Markov_v3_diffphi_uniform.mat','theta_mu_store','theta_sig2_store1','theta_sig2_store2','theta_sig2_store3','theta_sig2_store4',...
                 'theta_sig2_store5','theta_sig2_store6','theta_sig2_store7','theta_phi_store','theta_latent_b1_store',...
                 'theta_latent_b2_store','theta_latent_b3_store','theta_latent_A_store','theta_latent_v1_store','theta_latent_v2_store','theta_latent_tau_store',...
                 'chol_theta_sig2_store1','chol_theta_sig2_store2','chol_theta_sig2_store3','chol_theta_sig2_store4',...
                 'chol_theta_sig2_store5','chol_theta_sig2_store6','chol_theta_sig2_store7');
         end
     i=i+1;   
     
end
      save_draws=13000;
      chol_theta_sig2_store1=chol_theta_sig2_store1(save_draws:end,:);
      chol_theta_sig2_store2=chol_theta_sig2_store2(save_draws:end,:);
      chol_theta_sig2_store3=chol_theta_sig2_store3(save_draws:end,:);
      chol_theta_sig2_store4=chol_theta_sig2_store4(save_draws:end,:);
      chol_theta_sig2_store5=chol_theta_sig2_store5(save_draws:end,:);
      chol_theta_sig2_store6=chol_theta_sig2_store6(save_draws:end,:);
      chol_theta_sig2_store7=chol_theta_sig2_store7(save_draws:end,:);
      
      
      theta_phi_store=theta_phi_store(save_draws:end,:);
      theta_mu_store=theta_mu_store(save_draws:end,:);
      theta_sig2_store1=theta_sig2_store1(save_draws:end,:);
      theta_sig2_store2=theta_sig2_store2(save_draws:end,:);
      theta_sig2_store3=theta_sig2_store3(save_draws:end,:);
      theta_sig2_store4=theta_sig2_store4(save_draws:end,:);
      theta_sig2_store5=theta_sig2_store5(save_draws:end,:);
      theta_sig2_store6=theta_sig2_store6(save_draws:end,:);
      theta_sig2_store7=theta_sig2_store7(save_draws:end,:);
      
      
      for j=1:num_subjects
          theta_latent_b1_store{j,1}=theta_latent_b1_store{j,1}(save_draws:end,:);
          theta_latent_b2_store{j,1}=theta_latent_b2_store{j,1}(save_draws:end,:);
          theta_latent_b3_store{j,1}=theta_latent_b3_store{j,1}(save_draws:end,:);
          theta_latent_A_store{j,1}=theta_latent_A_store{j,1}(save_draws:end,:);
          theta_latent_v1_store{j,1}=theta_latent_v1_store{j,1}(save_draws:end,:);
          theta_latent_v2_store{j,1}=theta_latent_v2_store{j,1}(save_draws:end,:);
          theta_latent_tau_store{j,1}=theta_latent_tau_store{j,1}(save_draws:end,:);  
      end

save('LBA_real_forstmanndata_PG_Markov_v3_diffphi_uniform.mat','theta_mu_store','theta_sig2_store1','theta_sig2_store2','theta_sig2_store3','theta_sig2_store4',...
                 'theta_sig2_store5','theta_sig2_store6','theta_sig2_store7','theta_phi_store','theta_latent_b1_store',...
                 'theta_latent_b2_store','theta_latent_b3_store','theta_latent_A_store','theta_latent_v1_store','theta_latent_v2_store','theta_latent_tau_store',...
                 'chol_theta_sig2_store1','chol_theta_sig2_store2','chol_theta_sig2_store3','chol_theta_sig2_store4',...
                 'chol_theta_sig2_store5','chol_theta_sig2_store6','chol_theta_sig2_store7');











%[X,W,A]=LBA_CSMC_rw_min_block(data.response{j,1},data.rt{j,1},data.cond{j,1},param,theta_latent{j,1}(:,:,1),num_subjects,num_block(j,1),num_within_block,num_particles,num_randeffect);
%[theta_latent{j,1}(:,:,1)]=refreshed_backward_simulation_LBA_min_block_prior_rw(X,W,A,param,data.response{j,1},data.rt{j,1},data.cond{j,1},num_randeffect,num_within_block);       
%           theta_latent_b1_store(i-burn,:)=theta_latent(1,:,j);
%           theta_latent_b2_store(i-burn,:)=theta_latent(2,:,j);
%           theta_latent_b3_store(i-burn,:)=theta_latent(3,:,j);
%           theta_latent_A_store(i-burn,:)=theta_latent(4,:,j);
%           theta_latent_v1_store(i-burn,:)=theta_latent(5,:,j);
%           theta_latent_v2_store(i-burn,:)=theta_latent(6,:,j);
%           theta_latent_tau_store(i-burn,:)=theta_latent(7,:,j);  
%save('/srv/scratch/z3512791/LBA_real_PG_Markov_boot_Mod1.mat','theta_mu_store','theta_sig2_store1','theta_sig2_store2','theta_sig2_store3','theta_sig2_store4',...
%'theta_sig2_store5','theta_sig2_store6','theta_sig2_store7','theta_phi_store'); 
%save('/srv/scratch/z3512791/LBA_sim_PG_Markov_rw_min.mat','theta_mu_store','theta_sig2_store1','theta_sig2_store2','theta_sig2_store3','theta_sig2_store4',...
%       'theta_sig2_store5','theta_sig2_store6','theta_sig2_store7','theta_phi_store','theta_latent_b1_store',...
%       'theta_latent_b2_store','theta_latent_b3_store','theta_latent_A_store','theta_latent_v1_store','theta_latent_v2_store','theta_latent_tau_store'); 

     
%save the output to your directory
%save('/srv/scratch/z3512791/LBA_real_PG_log_halft1_propN_adaptmixchol3_2.mat','theta_mu_store','theta_sig2_store1','theta_sig2_store2','theta_sig2_store3','theta_sig2_store4',...
%             'theta_sig2_store5','theta_sig2_store6','theta_sig2_store7','theta_latent_b1_store','theta_latent_b2_store','theta_latent_b3_store',...
%             'theta_latent_A_store','theta_latent_v1_store','theta_latent_v2_store','theta_latent_tau_store','CPU_time'); 
     
   
%      
%     %sample sigma
%     theta_latent_temp1=zeros(num_randeffect,num_trials(1,1),num_subjects);
%     k_half=v_half+num_randeffect-1+num_subjects*num_trials(1,1);
%     theta_latent_temp1(:,1,:)=multiprod(sqrt((eye(num_randeffect)-(param.theta_phi^2)*eye(num_randeffect))),(theta_latent(:,1,:)-param.theta_mu));
%     theta_latent_temp1(:,2:end,:)=(theta_latent(:,2:end,:)-param.theta_mu-multiprod(param.theta_phi*eye(num_randeffect),(theta_latent(:,1:num_trials-1,:)-param.theta_mu)));
%     theta_latent_reshape=reshape(theta_latent_temp1,num_randeffect,1,num_trials(1,1)*num_subjects);
%     theta_latent_reshape_transpose=multitransp(theta_latent_reshape);
%     cov_temp=sum(multiprod(theta_latent_reshape,theta_latent_reshape_transpose),3);
%     B_half=2*v_half*diag([1/a1_half;1/a2_half;1/a3_half;1/a4_half;1/a5_half;1/a6_half;1/a7_half])+cov_temp;
%     param.theta_sig2=iwishrnd(B_half,k_half);

%     repmat_sig2_term1=repmat(theta_sig2_inv,1,1,num_subjects);
%     theta_latent_term1=(1-param.theta_phi^2).*theta_latent(:,1,:);
%     term1=sum(multiprod(repmat_sig2_term1,theta_latent_term1),3);
%     repmat_sig2_term2=repmat(theta_sig2_inv,1,1,num_subjects*(num_trials(1,1)-1));
%     current_theta_latent_reshape=reshape(theta_latent(:,2:end,:),num_randeffect,1,num_subjects*(num_trials(1,1)-1));
%     lag_theta_latent_reshape=reshape(theta_latent(:,1:num_trials(1,1)-1,:),num_randeffect,1,num_subjects*(num_trials(1,1)-1));
%     term2_1=multiprod(repmat_sig2_term2,current_theta_latent_reshape);
%     term2_2=multiprod(param.theta_phi.*repmat_sig2_term2,current_theta_latent_reshape);
%     term2_3=multiprod((param.theta_phi^2).*repmat_sig2_term2,lag_theta_latent_reshape);
%     term2_4=multiprod((param.theta_phi).*repmat_sig2_term2,lag_theta_latent_reshape);
%     term2=sum(term2_1-term2_2+term2_3-term2_4,3);
%     mean_mu=var_mu*(term1+term2);
%     param.theta_mu=(mvnrnd(mean_mu',chol_var_mu*chol_var_mu'))';

%      phi_temp1=theta_latent(:,2:end,:)-param.theta_mu;
%      phi_temp2=reshape(phi_temp1,num_randeffect,1,(num_trials(1,1)-1)*num_subjects);
%      phi_temp2_transpose=multitransp(phi_temp2);
%      repmat_sig2=repmat(inv(param.theta_sig2),1,1,(num_trials(1,1)-1)*num_subjects);
%      mult_temp1=multiprod(phi_temp2_transpose,repmat_sig2);
%      mult_temp2=multiprod(mult_temp1,phi_temp2);
%      var_phi=1/(sum(mult_temp2,3));
%      phi_mean1=theta_latent(:,1:num_trials(1,1)-1,:)-param.theta_mu;
%      phi_mean1_reshape=reshape(phi_mean1,num_randeffect,1,(num_trials(1,1)-1)*num_subjects);
%      phi_mean2_reshape=multitransp(reshape(theta_latent(:,2:end,:),num_randeffect,1,(num_trials(1,1)-1)*num_subjects));
%      mult_temp1_mean=multiprod(phi_mean2_reshape,repmat_sig2);
%      mult_temp2_mean=multiprod(mult_temp1_mean,phi_mean1_reshape);
%      theta_mu_reshape=multitransp(repmat(param.theta_mu,1,1,(num_trials(1,1)-1)*num_subjects));
%      mult_temp3_mean=multiprod(theta_mu_reshape,repmat_sig2);
%      mult_temp4_mean=multiprod(mult_temp3_mean,phi_mean1_reshape);
%      mean_phi=var_phi*(sum(mult_temp2_mean-mult_temp4_mean,3));
%      phi_star=normt_rnd(mean_phi,var_phi,0,0.999);
%      prior=log(betapdf((1+param.theta_phi)/2,a_phi,b_phi));
%      prior_star=log(betapdf((1+phi_star)/2,a_phi,b_phi));
%      num_phi=(num_subjects/2)*log(1-phi_star^2);
%      den_phi=(num_subjects/2)*log(1-param.theta_phi^2);
%      r2=exp(num_phi-den_phi+prior_star-prior);
%      C2=min(1,r2);
%      A2=rand();
%      if A2<=C2
%         param.theta_phi=phi_star;
%      end
%       
%     %sample a1,...,ap