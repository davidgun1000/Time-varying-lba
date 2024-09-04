%keep the draws in the sampling stage
load('LBA_MS_Forstmann_v2.mat');
num_subjects=19;
length_draws=length(theta_latent_A_store1);
theta_latent_A_store1=theta_latent_A_store1(length_draws-9999:end,:);
theta_latent_b1_store1=theta_latent_b1_store1(length_draws-9999:end,:);
theta_latent_b2_store1=theta_latent_b2_store1(length_draws-9999:end,:);
theta_latent_b3_store1=theta_latent_b3_store1(length_draws-9999:end,:);
theta_latent_v1_store1=theta_latent_v1_store1(length_draws-9999:end,:);
theta_latent_v2_store1=theta_latent_v2_store1(length_draws-9999:end,:);
theta_latent_tau_store1=theta_latent_tau_store1(length_draws-9999:end,:);

theta_latent_A_store2=theta_latent_A_store2(length_draws-9999:end,:);
theta_latent_b1_store2=theta_latent_b1_store2(length_draws-9999:end,:);
theta_latent_b2_store2=theta_latent_b2_store2(length_draws-9999:end,:);
theta_latent_b3_store2=theta_latent_b3_store2(length_draws-9999:end,:);
theta_latent_v1_store2=theta_latent_v1_store2(length_draws-9999:end,:);
theta_latent_v2_store2=theta_latent_v2_store2(length_draws-9999:end,:);
theta_latent_tau_store2=theta_latent_tau_store2(length_draws-9999:end,:);

Post.theta_mu=Post.theta_mu(length_draws-9999:end,:);
Post.chol_theta_sig2_store{1,1}=Post.chol_theta_sig2_store{1,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{2,1}=Post.chol_theta_sig2_store{2,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{3,1}=Post.chol_theta_sig2_store{3,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{4,1}=Post.chol_theta_sig2_store{4,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{5,1}=Post.chol_theta_sig2_store{5,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{6,1}=Post.chol_theta_sig2_store{6,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{7,1}=Post.chol_theta_sig2_store{7,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{8,1}=Post.chol_theta_sig2_store{8,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{9,1}=Post.chol_theta_sig2_store{9,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{10,1}=Post.chol_theta_sig2_store{10,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{11,1}=Post.chol_theta_sig2_store{11,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{12,1}=Post.chol_theta_sig2_store{12,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{13,1}=Post.chol_theta_sig2_store{13,1}(length_draws-9999:end,:);
Post.chol_theta_sig2_store{14,1}=Post.chol_theta_sig2_store{14,1}(length_draws-9999:end,:);

for j=1:num_subjects
    Post.trans_prob_11{j,1} = Post.trans_prob_11{j,1}(length_draws-9999:end,:);
    Post.trans_prob_22{j,1} = Post.trans_prob_22{j,1}(length_draws-9999:end,:);
    
end

for j=1:num_subjects
    Post.S{j,1} = Post.S{j,1}/23000;
end

save('LBA_MS_Forstmann_v2.mat','Post','theta_latent_b1_store1','theta_latent_b2_store1','theta_latent_b3_store1','theta_latent_A_store1','theta_latent_v1_store1','theta_latent_v2_store1','theta_latent_tau_store1',...
            'theta_latent_b1_store2','theta_latent_b2_store2','theta_latent_b3_store2','theta_latent_A_store2','theta_latent_v1_store2','theta_latent_v2_store2','theta_latent_tau_store2'); 
