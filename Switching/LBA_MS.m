%estimating the Switching LBA model using PMwG method for the Forstmann (2008) dataset
%The Switching LBA model specification can be found in the paper. The data is stored
%in the Matlab file 'LBA_realdata.mat', It has three
%components: data.cond contains the conditions of the experiments, data.rt
%contains the response time, and data.response: response=1 for incorrect
%response and response=2 for correct response.


load('LBA_realdata.mat'); %load the dataset, see an example in the 'LBA_realdata.mat'
%parpool(28)  %number of multi-processors used
load('init_value_Forstmann.mat'); % initial values for the forstmann dataset for the random effects
num_randeffect=14; %number of random effects
num_subjects=length(data.rt); % number of subjects
for j=1:num_subjects
    num_trials(j,1)=length(data.rt{j,1}); %number of trials 
end
num_particles=100; %number of particles used in the conditional sequential Monte Carlo algorithm
prior.mu_mean=0; %the prior mean for \mu
prior.mu_sig2=eye(num_randeffect); %the prior variance for \mu 
prior.sigma_v0 = 30; %the prior parameter for \Sigma
prior.sigma_s0 = eye(num_randeffect); %the prior parameter for \Sigma
prior.trans_prob = [4,1;1,4]; %the prior parameter for the elements of transition matrix 

num_cat=2; % number of states 
theta_sig2 = iwishrnd(eye(num_randeffect),prior.sigma_v0); %initial values for sigma
p0=[0.5;0.5];
smoothdens = false;
pinit = ones(num_cat,1)/num_cat;
for j=1:num_subjects
    
    trans_prob(:,:,j) = [0.9,0.1;0.2,0.8]; %initial values for the element of the transition probability 
end
for j=1:num_subjects
    S0{j,1} = sum(cumsum(p0) < rand)+1;
end

for j=1:num_subjects %initial values for hidden Markov states s
    for t=1:num_trials(j,1)
        if t>1
           p_pred = trans_prob(S{j,1}(1,t-1),:)'; 
        else
           p_pred = trans_prob(S0{j,1},:)'; 
        end
        S{j,1}(1,t) = sum(cumsum(p_pred)<rand) +1;
    end
end

burn=1000; %the burn in replications 
adapt=2000; %the initial sampling stage
nit=20000; %the sampling stage
s=burn+nit+adapt; %the total number of iterations

theta_latent = [theta_latent_b1min_true1,theta_latent_b2min_true1,theta_latent_b3min_true1,theta_latent_A_true1,theta_latent_v1_true1,theta_latent_v2_true1,theta_latent_tau_true1,...
                theta_latent_b1min_true2,theta_latent_b2min_true2,theta_latent_b3min_true2,theta_latent_A_true2,theta_latent_v1_true2,theta_latent_v2_true2,theta_latent_tau_true2];
theta_mu = mean(theta_latent); %initial values for mu

for j=1:num_subjects
    Post.S{j,1}=zeros(1,num_trials(j,1));    
end

mean_theta=[];
covmat_theta=[];
count=1;
for i=1:s
    i
    
    param.theta_mu = theta_mu;
    param.theta_sig2 = theta_sig2;
    %sampling the random effects for the two states
    parfor j=1:num_subjects
       [theta_latent(j,:)] = LBA_MC_IS_cond_orig_real_log_prop_usedmixchol3(data,param,theta_latent(j,:),num_subjects,num_trials(j,1),num_particles,mean_theta,covmat_theta,i,burn,adapt,j,S{j,1}); 
    end
 
    %Order Restriction (optional)
    for j=1:num_subjects
        if  (exp(theta_latent(j,1))+exp(theta_latent(j,4))) >= (exp(theta_latent(j,8))+exp(theta_latent(j,11))) | ...
            (exp(theta_latent(j,2))+exp(theta_latent(j,4))) >= (exp(theta_latent(j,9))+exp(theta_latent(j,11))) | ...
            (exp(theta_latent(j,3))+exp(theta_latent(j,4))) >= (exp(theta_latent(j,10))+exp(theta_latent(j,11)))
            
            qq = theta_latent(j,8:14);
            theta_latent(j,8:14) = theta_latent(j,1:7);
            theta_latent(j,1:7) = qq;
        end
    end
    % sample \mu from the full conditional distribution        
    var_mu=inv(num_subjects*inv(theta_sig2)+inv(prior.mu_sig2));
    mean_mu=var_mu*(inv(theta_sig2)*sum(theta_latent,1)');
    chol_var_mu = chol(var_mu,'lower');
    theta_mu = mvnrnd(mean_mu,chol_var_mu*chol_var_mu');
        
    % sample \sigma from the full conditional distribution
    k_half=prior.sigma_v0+num_subjects;
    theta_latent_minus_mu=theta_latent'-theta_mu';
    cov_temp = theta_latent_minus_mu*theta_latent_minus_mu';
    B_half=prior.sigma_s0+cov_temp;
    theta_sig2=iwishrnd(B_half,k_half);
        
    %sampling the path indicators for each subject, the hidden Markov
    %process s
    
    parfor j=1:num_subjects
        [lh] = compute_lh(data,theta_mu,theta_sig2,theta_latent,num_trials(j,1),j);
        [S_temp,  ~, ~,  ~, ~]= simstate_ms(trans_prob(:,:,j),lh,pinit,smoothdens);
        S0{j,1} = S_temp(1,1);
        S{j,1} = S_temp(1,2:end);
    end
   
    
    for ii=1:num_subjects
        S_com = [S0{ii,1},S{ii,1}];
        temp_trans = statecount(S_com,num_cat) + prior.trans_prob;
        trans_prob(:,:,ii) = dirichsim(temp_trans);    
    end

     %obtain proposals, training the proposals for conditional Monte Carlo
      %algorithm, we can adapt this proposal for every iteration of MCMC,
      %but to speed up, we adapt this proposal every 100 iteration of MCMC.
      %This proposal is computed in the sampling stage. We need the
      %proposal for each subject.

    if i>=burn+adapt & mod(i,100)==0
       for j=1:num_subjects 
           % in the large matrix called theta_mix below, you have to list (1) all
            % you random effects in the switching LBA model of state1, in the case of Forstmann
            % you have \alpha_{b_11}, \alpha_{b_21}, \alpha_{b_31}, \alpha_{A1}, \alpha_{v_11}, \alpha_{v_21}, \alpha_{\taut1} , (2) followed by the random effects 
            % of state 2, in the case of Forstmann
            % you have \alpha_{b_12}, \alpha_{b_22}, \alpha_{b_32},
            % \alpha_{A2}, \alpha_{v_12}, \alpha_{v_22},
            % \alpha_{\tau_2}, (3) followed by parameters \mu, cholesky factor 
            % (lower triangular matrix) of the covariance matrix
            % \Sigma_{\alpha}
           theta_mix = [theta_latent_b1_store1(burn:end,j),theta_latent_b2_store1(burn:end,j),theta_latent_b3_store1(burn:end,j),...
             theta_latent_A_store1(burn:end,j),theta_latent_v1_store1(burn:end,j),theta_latent_v2_store1(burn:end,j),theta_latent_tau_store1(burn:end,j),...
             theta_latent_b1_store2(burn:end,j),theta_latent_b2_store2(burn:end,j),theta_latent_b3_store2(burn:end,j),...
             theta_latent_A_store2(burn:end,j),theta_latent_v1_store2(burn:end,j),theta_latent_v2_store2(burn:end,j),theta_latent_tau_store2(burn:end,j),...
             Post.theta_mu(burn:end,:),Post.chol_theta_sig2_store{1,1}(burn:end,:),Post.chol_theta_sig2_store{2,1}(burn:end,:),Post.chol_theta_sig2_store{3,1}(burn:end,:),...
             Post.chol_theta_sig2_store{4,1}(burn:end,:),Post.chol_theta_sig2_store{5,1}(burn:end,:),Post.chol_theta_sig2_store{6,1}(burn:end,:),Post.chol_theta_sig2_store{7,1}(burn:end,:),...
             Post.chol_theta_sig2_store{8,1}(burn:end,:),Post.chol_theta_sig2_store{9,1}(burn:end,:),Post.chol_theta_sig2_store{10,1}(burn:end,:),Post.chol_theta_sig2_store{11,1}(burn:end,:),...
             Post.chol_theta_sig2_store{12,1}(burn:end,:),Post.chol_theta_sig2_store{13,1}(burn:end,:),Post.chol_theta_sig2_store{14,1}(burn:end,:)];
             covmat_theta(:,:,j)=cov(theta_mix); %computing the sample covariance matrix for the joint random effects from states 1 and 2 and parameters \mu, \Sigma, and \phi
             covmat_theta(:,:,j)=topdm(covmat_theta(:,:,j));
             mean_theta(j,:)=mean(theta_mix); %computing the sample mean for the joint random effects of states 1 and 2 and parameters \mu, \Sigma 
        
       end 

    end

    %storing the MCMC parameter draws
    %storing the elements of transition matrix for each subject
    for j=1:num_subjects
        Post.trans_prob_11{j,1}(i,1) = trans_prob(1,1,j);
        Post.trans_prob_22{j,1}(i,1) = trans_prob(2,2,j);
    end
    Post.theta_mu(i,:) = theta_mu; %storing the \mu
    
    %storing the cholesky factor of the \Sigma
    chol_theta_sig2=chol(theta_sig2(:,:,1),'lower');
    Post.chol_theta_sig2_store{1,1}(i,:)=log(chol_theta_sig2(1,1));
    Post.chol_theta_sig2_store{2,1}(i,:)=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
    Post.chol_theta_sig2_store{3,1}(i,:)=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
    Post.chol_theta_sig2_store{4,1}(i,:)=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
    Post.chol_theta_sig2_store{5,1}(i,:)=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
    Post.chol_theta_sig2_store{6,1}(i,:)=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
    Post.chol_theta_sig2_store{7,1}(i,:)=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];
    Post.chol_theta_sig2_store{8,1}(i,:)=[chol_theta_sig2(8,1:7),log(chol_theta_sig2(8,8))];
    Post.chol_theta_sig2_store{9,1}(i,:)=[chol_theta_sig2(9,1:8),log(chol_theta_sig2(9,9))];
    Post.chol_theta_sig2_store{10,1}(i,:)=[chol_theta_sig2(10,1:9),log(chol_theta_sig2(10,10))];
    Post.chol_theta_sig2_store{11,1}(i,:)=[chol_theta_sig2(11,1:10),log(chol_theta_sig2(11,11))];
    Post.chol_theta_sig2_store{12,1}(i,:)=[chol_theta_sig2(12,1:11),log(chol_theta_sig2(12,12))];
    Post.chol_theta_sig2_store{13,1}(i,:)=[chol_theta_sig2(13,1:12),log(chol_theta_sig2(13,13))];
    Post.chol_theta_sig2_store{14,1}(i,:)=[chol_theta_sig2(14,1:13),log(chol_theta_sig2(14,14))];
    
    
    
    %storing the hidden markov state S for each subject
    for j=1:num_subjects
        Sind = S{j,1}==1;
        Post.S{j,1}(1,:) = Post.S{j,1}(1,:)+double(Sind);
    end
    %storing the random effects of state1 for each subject
    theta_latent_b1_store1(i,:) = theta_latent(:,1)';
    theta_latent_b2_store1(i,:) = theta_latent(:,2)';
    theta_latent_b3_store1(i,:) = theta_latent(:,3)';    
    theta_latent_A_store1(i,:) = theta_latent(:,4)';
    theta_latent_v1_store1(i,:) = theta_latent(:,5)';
    theta_latent_v2_store1(i,:) = theta_latent(:,6)';
    theta_latent_tau_store1(i,:) = theta_latent(:,7)';
    %storing the random effects of state 2 for each subject
    theta_latent_b1_store2(i,:) = theta_latent(:,8)';
    theta_latent_b2_store2(i,:) = theta_latent(:,9)';
    theta_latent_b3_store2(i,:) = theta_latent(:,10)';  
    theta_latent_A_store2(i,:) = theta_latent(:,11)';
    theta_latent_v1_store2(i,:) = theta_latent(:,12)';
    theta_latent_v2_store2(i,:) = theta_latent(:,13)';
    theta_latent_tau_store2(i,:) = theta_latent(:,14)';
     %save the output to your directory every 250 iterations  
    if mod(i,250)==0
        save('/scratch/jz21/dg2271/LBA_MS_Forstmann_latest_correct.mat','Post','theta_latent_b1_store1','theta_latent_b2_store1','theta_latent_b3_store1','theta_latent_A_store1','theta_latent_v1_store1','theta_latent_v2_store1','theta_latent_tau_store1',...
            'theta_latent_b1_store2','theta_latent_b2_store2','theta_latent_b3_store2','theta_latent_A_store2','theta_latent_v1_store2','theta_latent_v2_store2','theta_latent_tau_store2'); 
    end
    
    %generate posterior predictive
    
    if i>=burn+adapt & mod(i,100)==0
       
       for j=1:num_subjects
           for k=1:num_trials(j,1)
               
               if S{j,1}(1,k)==1
               temp_b1_pred = theta_latent_b1_store1(i,j);
               temp_b2_pred = theta_latent_b2_store1(i,j);
               temp_b3_pred = theta_latent_b3_store1(i,j);
               temp_A_pred = theta_latent_A_store1(i,j);
               temp_v1_pred = theta_latent_v1_store1(i,j);
               temp_v2_pred = theta_latent_v2_store1(i,j);
               temp_tau_pred = theta_latent_tau_store1(i,j);
               else
               temp_b1_pred = theta_latent_b1_store2(i,j);
               temp_b2_pred = theta_latent_b2_store2(i,j);
               temp_b3_pred = theta_latent_b3_store2(i,j);
               temp_A_pred = theta_latent_A_store2(i,j);
               temp_v1_pred = theta_latent_v1_store2(i,j);
               temp_v2_pred = theta_latent_v2_store2(i,j);
               temp_tau_pred = theta_latent_tau_store2(i,j);
               end
               
               theta_latent_b_pred = choose_b(data.cond{j,1}(k,1),temp_b1_pred,temp_b2_pred,temp_b3_pred);
               [tmp_response,tmp_rt]=LBA_trial_MS_LBA(exp(temp_A_pred),...
                    exp(theta_latent_b_pred)+exp(temp_A_pred),[exp(temp_v1_pred),exp(temp_v2_pred)],...
                    exp(temp_tau_pred),1,2);
                
                synthetic_data.cond{j,1}(k,count)=data.cond{j,1}(k,1);
                synthetic_data.response{j,1}(k,count)=tmp_response;
                synthetic_data.rt{j,1}(k,count)=tmp_rt;
           
           end
       end
       count=count+1; 
    end
    
    
end
%save the results, you need to keep only the draws in the sampling stage
save('/scratch/jz21/dg2271/pred_Forstmann_MS_LBA_latest_correct.mat','synthetic_data');
