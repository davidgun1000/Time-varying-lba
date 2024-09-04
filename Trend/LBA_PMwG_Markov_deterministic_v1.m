%estimating the Time-varying hierarchical LBA (Trend) model using PMwG method for the Forstmann (2008) dataset
%The Trend model specification can be found in the paper. The data is stored
%in the Matlab file 'LBA_realdata_Forstmann_block.mat', It has three
%components: data.cond contains the conditions of the experiments, data.rt
%contains the response time, and data.response: response=1 for incorrect
%response and response=2 for correct response.
load('LBA_realdata_Fortsmann_block.mat'); %load the dataset, see an example in the 'LBA_realdata_Forstmann_block.mat'
num_subjects=length(data.rt); %number of subjects

% Note: The Trend model is referred to as the 'deterministic' model below and in the related functions

for j=1:num_subjects
    num_block(j,1)=length(data.rt{j,1}); %computing the number of block for each subject
    for k=1:num_block(j,1)
    num_within_block{j,1}{k,1}=length(data.rt{j,1}{k,1}); %computing the number of trials within each block for each subject
    end
end
num_particles=250; %number of particles used in the conditional sequential Monte Carlo algorithm
parpool(28) %number of multi-processors used

num_randeffect=7; %the number of random effects in the LBA model
num_covariates=3; %the number of covariates 1, t, and t^2
param.theta_beta=[0.2484,0,0,0.5481,0,0,0.1,0,0,-0.3260,0,0,-0.1182,0,0,1.0858,0,0,-1.9437,0,0]'; % the initial values for the parameter \beta
param.theta_sig2=iwishrnd(eye(num_randeffect),20); %the initial values for the parameter \Sigma_{\alpha}
param.sv=1; % the standard deviation of the drift rate and is assumed to be 1
param.num_randeffect=7; % the number of random effects in the LBA model 

num_choice=2; %the number of choice in the dataset
burn=1000; %the burn in replications
adapt=4000; %the initial sampling stage
sampling=20000; %the sampling stage
s=burn+adapt+sampling; %the total number of iterations

%values of the hyperparameters for lower level parameters
prior.beta_mean=zeros(num_covariates*num_randeffect,1); %the prior mean for \beta
prior.beta_sig2=eye(num_covariates*num_randeffect); %the prior variance for \beta
prior.a_phi=20; %the prior parameter for \phi_{\alpha}
prior.b_phi=1.5; %the prior parameter for \phi_{\alpha}
prior.v0_sigma=20; %the prior parameter for \Sigma_{\alpha}
prior.s0_sigma=eye(num_randeffect); %the prior parameter for \Sigma_{\alpha}

%constructing the covariates (1,t,t^2) for each subject, random effects,
%and blocks
for j=1:num_subjects
    for k=1:num_block(j,1)
        x_cov=[1,k,k.^2];
        covariates{j,1}(:,:,k)=[x_cov,zeros(1,18);
                   zeros(1,3),x_cov,zeros(1,15);
                   zeros(1,6),x_cov,zeros(1,12);
                   zeros(1,9),x_cov,zeros(1,9);
                   zeros(1,12),x_cov,zeros(1,6);
                   zeros(1,15),x_cov,zeros(1,3);
                   zeros(1,18),x_cov];
    end
end
% generating initial trajectory for the random effect for each subject
for j=1:num_subjects
    for k=1:num_block(j,1)
        cov_times_beta=covariates{j,1}(:,:,k)*param.theta_beta;
        theta_latent{j,1}(:,k)=(mvnrnd(cov_times_beta',param.theta_sig2))';
    end
end

temp=1;
i=1;

while i<=s
    i
    tic
    
    % sample \beta from the full conditional distribution. See the
    % paper Gunawan et al (2019)
    theta_sig2_inv=inv(param.theta_sig2);
    temp_var_beta=zeros(num_covariates*num_randeffect,num_covariates*num_randeffect);
    temp_mu_beta=zeros(num_covariates*num_randeffect,1);
    for j=1:num_subjects
        repmat_sig2_inv=repmat(theta_sig2_inv,1,1,num_block(j,1));
        covariates_transpose=multitransp(covariates{j,1});
        temp_var1=multiprod(covariates_transpose,repmat_sig2_inv);
        temp_var2=multiprod(temp_var1,covariates{j,1});
        temp_var_beta=temp_var_beta+sum(temp_var2,3);
        current_theta_latent=reshape(theta_latent{j,1}(:,1:num_block(j,1)),num_randeffect,1,num_block(j,1));
        temp_mu1=multiprod(temp_var1,current_theta_latent);
        temp_mu_beta=temp_mu_beta+sum(temp_mu1,3);
    end
    
    var_beta=inv(temp_var_beta+eye(num_covariates*num_randeffect));
    chol_var_beta=chol(var_beta,'lower');
    mu_beta=var_beta*(temp_mu_beta);
    param.theta_beta=(mvnrnd(mu_beta',chol_var_beta*chol_var_beta'))';
    
    %sample \Sigma_{\alpha} from the full conditional distribution. See the
    %paper Gunawan et al. (2019)
    k_half=prior.v0_sigma+sum(num_block);
    cov_temp=zeros(num_randeffect,num_randeffect);
    for j=1:num_subjects
        repmat_beta=repmat(param.theta_beta,1,1,num_block(j,1));
        cov_times_beta_temp=multiprod(covariates{j,1},param.theta_beta);
        cov_times_beta=reshape(cov_times_beta_temp,num_randeffect,num_block(j,1));
        theta_latent_temp=(theta_latent{j,1}-cov_times_beta);
        temp=theta_latent_temp*theta_latent_temp';
        cov_temp=cov_temp+temp;
    end
    B_half=prior.s0_sigma+cov_temp;
    param.theta_sig2=iwishrnd(B_half,k_half);
    
%--------------------------------------------------------------------------------------------------------
    %conditional Sequential Monte Carlo algorithm to generate trajectory of
    %random effects for each subject
    %if we are in the burn in and initial adaptation stage, we use less
    %efficient prior that is based on the mixture of prior of random
    %effects and 'random walk', otherwise we use better proposal as
    %outlined in the Gunawan et al (2019)
    
        if i<=burn+adapt
           parfor j=1:num_subjects
               [theta_latent{j,1}(:,:,1)]=LBA_CSMC_prior_rw_deterministic(data.response{j,1},data.rt{j,1},data.cond{j,1},covariates{j,1},param,theta_latent{j,1}(:,:,1),num_subjects,num_block(j,1),num_within_block{j,1},num_particles,num_randeffect);               
           end            
        end
    
        if i>=burn+adapt+1
           parfor j=1:num_subjects 
               [theta_latent{j,1}(:,:,1)]=LBA_CSMC_prior_prop_deterministic(data.response{j,1},data.rt{j,1},data.cond{j,1},covariates{j,1},param,theta_latent{j,1}(:,:,1),num_subjects,num_block(j,1),num_within_block{j,1},num_particles,num_randeffect,...
                mean_param{j,1},covmat_param{j,1})
           end
        end
   
      %storing the MCMC posterior draws  
      
      if i>burn
      %storing the cholesky factor of the \Sigma_{\alpha}
      chol_theta_sig2=chol(param.theta_sig2,'lower');
      chol_theta_sig2_store1(i-burn,:)=log(chol_theta_sig2(1,1));
      chol_theta_sig2_store2(i-burn,:)=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
      chol_theta_sig2_store3(i-burn,:)=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
      chol_theta_sig2_store4(i-burn,:)=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
      chol_theta_sig2_store5(i-burn,:)=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
      chol_theta_sig2_store6(i-burn,:)=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
      chol_theta_sig2_store7(i-burn,:)=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];
      
      theta_beta_store(i-burn,:)=param.theta_beta'; %storing the parameter \beta
      theta_sig2_store1(i-burn,:)=param.theta_sig2(1,:); %storing the parameter \Sigma_{\alpha}
      theta_sig2_store2(i-burn,:)=param.theta_sig2(2,2:end); %storing the parameter \Sigma_{\alpha}
      theta_sig2_store3(i-burn,:)=param.theta_sig2(3,3:end); %storing the parameter \Sigma_{\alpha}
      theta_sig2_store4(i-burn,:)=param.theta_sig2(4,4:end); %storing the parameter \Sigma_{\alpha}
      theta_sig2_store5(i-burn,:)=param.theta_sig2(5,5:end); %storing the parameter \Sigma_{\alpha}
      theta_sig2_store6(i-burn,:)=param.theta_sig2(6,6:end); %storing the parameter \Sigma_{\alpha}
      theta_sig2_store7(i-burn,:)=param.theta_sig2(7,7:end); %storing the parameter \Sigma_{\alpha}
      
      %storing the random effects for each subject
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
      
     %obtain proposals, training the proposals for conditional Monte Carlo
      %algorithm, we can adapt this proposal for every iteration of MCMC,
      %but to speed up, we adapt this proposal every 100 iteration of MCMC.
      %This proposal is computed in the sampling stage. We need the
      %proposal for each subject and for each block.
      if i>=burn+adapt & mod(i,100)==0 
      for j=1:num_subjects
          length_draws=length(theta_beta_store(:,1));
          for t=1:num_block(j,1)
              % in the large matrix called xx below, you have to list (1) all your random effects in the LBA model at time t, in the case of Forstmann
            % you have \alpha_{b_1t}, \alpha_{b_2t}, \alpha_{b_3t}, \alpha_{A_t}, \alpha_{v_1t}, \alpha_{v_2t}, \alpha_{\taut} at time t, (2) followed by parameters \beta, cholesky factor 
            % (lower triangular matrix) of the covariance matrix
            % \Sigma_{\alpha}, the transformations are necessary to make sure
            % that all the parameters are lies on the real line. The
            % transformations are especially important for \Sigma_{\alpha}
            
            xx=[theta_latent_b1_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_b2_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_b3_store{j,1}(length_draws-(adapt-1):length_draws,t),...
                theta_latent_A_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_v1_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_latent_v2_store{j,1}(length_draws-(adapt-1):length_draws,t),...
                theta_latent_tau_store{j,1}(length_draws-(adapt-1):length_draws,t),theta_beta_store(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store1(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store2(length_draws-(adapt-1):length_draws,:),...
                chol_theta_sig2_store3(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store4(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store5(length_draws-(adapt-1):length_draws,:),chol_theta_sig2_store6(length_draws-(adapt-1):length_draws,:),...
                chol_theta_sig2_store7(length_draws-(adapt-1):length_draws,:)];
            mean_param{j,1}(t,:)=mean(xx); %computing the sample mean for the joint random effects at time t and parameters \beta, \Sigma_{\alpha},
            cov_temp=cov(xx);
            cov_temp=topdm(cov_temp);
            cov_temp=chol(cov_temp,'lower');
            covmat_param{j,1}(:,:,t)=cov_temp*cov_temp'; %computing the sample covariance matrix for the joint random effects at time t and parameters \beta, \Sigma_{\alpha},      
          end
      end
      end
      

      
         %save the output to your directory every 2000 iterations  
         
         if mod(i,2000)==0   
            save('/srv/scratch/z3512791/LBA_Fortsmann_Markov_deterministic.mat','theta_beta_store','theta_sig2_store1','theta_sig2_store2','theta_sig2_store3','theta_sig2_store4',...
                 'theta_sig2_store5','theta_sig2_store6','theta_sig2_store7','theta_latent_b1_store',...
                 'theta_latent_b2_store','theta_latent_b3_store','theta_latent_A_store','theta_latent_v1_store','theta_latent_v2_store','theta_latent_tau_store',...
                 'chol_theta_sig2_store1','chol_theta_sig2_store2','chol_theta_sig2_store3','chol_theta_sig2_store4',...
                 'chol_theta_sig2_store5','chol_theta_sig2_store6','chol_theta_sig2_store7');
         end
     i=i+1;   
     toc
end

%keep the last 10000 draws for further analysis
length_draws=length(chol_theta_sig2_store1);
chol_theta_sig2_store1=chol_theta_sig2_store1(length_draws-9999:end,:);
chol_theta_sig2_store2=chol_theta_sig2_store2(length_draws-9999:end,:);
chol_theta_sig2_store3=chol_theta_sig2_store3(length_draws-9999:end,:);
chol_theta_sig2_store4=chol_theta_sig2_store4(length_draws-9999:end,:);
chol_theta_sig2_store5=chol_theta_sig2_store5(length_draws-9999:end,:);
chol_theta_sig2_store6=chol_theta_sig2_store6(length_draws-9999:end,:);
chol_theta_sig2_store7=chol_theta_sig2_store7(length_draws-9999:end,:);

theta_beta_store=theta_beta_store(length_draws-9999:end,:);
theta_sig2_store1=theta_sig2_store1(length_draws-9999:end,:);
theta_sig2_store2=theta_sig2_store2(length_draws-9999:end,:);
theta_sig2_store3=theta_sig2_store3(length_draws-9999:end,:);
theta_sig2_store4=theta_sig2_store4(length_draws-9999:end,:);
theta_sig2_store5=theta_sig2_store5(length_draws-9999:end,:);
theta_sig2_store6=theta_sig2_store6(length_draws-9999:end,:);
theta_sig2_store7=theta_sig2_store7(length_draws-9999:end,:);

for j=1:num_subjects
    theta_latent_b1_store{j,1}=theta_latent_b1_store{j,1}(length_draws-9999:end,:);
    theta_latent_b2_store{j,1}=theta_latent_b2_store{j,1}(length_draws-9999:end,:);
    theta_latent_b3_store{j,1}=theta_latent_b3_store{j,1}(length_draws-9999:end,:);
    theta_latent_A_store{j,1}=theta_latent_A_store{j,1}(length_draws-9999:end,:);
    theta_latent_v1_store{j,1}=theta_latent_v1_store{j,1}(length_draws-9999:end,:);
    theta_latent_v2_store{j,1}=theta_latent_v2_store{j,1}(length_draws-9999:end,:);
    theta_latent_tau_store{j,1}=theta_latent_tau_store{j,1}(length_draws-9999:end,:);
end

save('/srv/scratch/z3512791/LBA_Fortsmann_Markov_deterministic.mat','theta_beta_store','theta_sig2_store1','theta_sig2_store2','theta_sig2_store3','theta_sig2_store4',...
                 'theta_sig2_store5','theta_sig2_store6','theta_sig2_store7','theta_latent_b1_store',...
                 'theta_latent_b2_store','theta_latent_b3_store','theta_latent_A_store','theta_latent_v1_store','theta_latent_v2_store','theta_latent_tau_store',...
                 'chol_theta_sig2_store1','chol_theta_sig2_store2','chol_theta_sig2_store3','chol_theta_sig2_store4',...
                 'chol_theta_sig2_store5','chol_theta_sig2_store6','chol_theta_sig2_store7');











