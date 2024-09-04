function [sirhat]=refreshed_backward_simulation_LBA_min_block_prop_prior3_diffphi(particles,w,indx,param,data_response,data_rt,data_cond,num_randeffect,num_within_block,...
    mean_param,covmat_param)

% this function is the refreshed backward simulation algorithm adapted to
% LBA model
T=size(w,1);
N=size(w,2);
outndx=NaN(1,T);
outndx(T)=randsample(N,1,true,w(T,:));
sirhat(:,T)=particles(:,outndx(T),T);
scale_covmat=1;
chol_theta_sig2=chol(param.theta_sig2,'lower');
chol_theta_sig2_1=log(chol_theta_sig2(1,1));
chol_theta_sig2_2=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
chol_theta_sig2_3=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
chol_theta_sig2_4=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
chol_theta_sig2_5=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
chol_theta_sig2_6=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
chol_theta_sig2_7=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];

u=zeros(N,1);
id1=zeros(N,1);
id2=zeros(N,1);
id3=zeros(N,1);
for t=T-1:-1:1
         
         indx_old=indx(t+1,outndx(t+1));% fixed the first index to the ancestor of the random effects t+1
         indx_star=rs_multinomial(w(t,:));% obtain the resampling index according to the weight of the particles at time t
         indx_temp1=indx(t,indx_old);
         indx_temp2=indx(t,indx_star(1,2:end));
         indx(t,:)=[indx_temp1,indx_temp2];
         particles_temp1=particles(:,indx_old,t);%select the particles based on 'indx_old'
         particles_temp2=particles(:,indx_star(1,2:end),t);%select the particles based on 'indx_star'
         particles(:,:,t)=[particles_temp1,particles_temp2];             
         particles_star(:,1)=sirhat(:,t+1);%set the first particles to the chosen random effects at time t+1 for conditioning
         reference_particle=sirhat(:,t+1);
         %the matrix xx1 below contains the random effects at time t-1,
         %parameters \mu, the cholesky factor of covariance matrix
         %\Sigma, and logistic transformation for the \phi
         xx1=[particles(:,:,t);param.theta_mu*ones(1,N);
            chol_theta_sig2_1'*ones(1,N);chol_theta_sig2_2'*ones(1,N);
            chol_theta_sig2_3'*ones(1,N);chol_theta_sig2_4'*ones(1,N);
            chol_theta_sig2_5'*ones(1,N);chol_theta_sig2_6'*ones(1,N);
            chol_theta_sig2_7'*ones(1,N);logit_inverse_min1_to1(param.theta_phi)*ones(1,N)];
         xx1_reshape=reshape(xx1,49,1,N);
         xx1_minus_meanparam=xx1_reshape-mean_param(t+1,num_randeffect+1:end)';
         temp1=multiprod((covmat_param(1:num_randeffect,num_randeffect+1:end,t+1)/covmat_param(num_randeffect+1:end,num_randeffect+1:end,t+1)),xx1_minus_meanparam);
         temp1_reshape=reshape(temp1,num_randeffect,N);
         cond_mean=mean_param(t+1,1:num_randeffect)'+temp1_reshape; %mean of the proposal for the first component
         cond_mean_ref=reference_particle+temp1_reshape; %mean of the proposal for the third component
         covmat_param(:,:,t+1)=topdm(covmat_param(:,:,t+1));
         cond_var=covmat_param(1:num_randeffect,1:num_randeffect,t+1)-covmat_param(1:num_randeffect,num_randeffect+1:end,t+1)*(covmat_param(num_randeffect+1:end,num_randeffect+1:end,t+1)\covmat_param(num_randeffect+1:end,1:num_randeffect,t+1));
         cond_var=topdm(cond_var);
         chol_cond_var=chol(real(cond_var),'lower');
         w1_mix=0.6;%set the mixture weights
         w2_mix=0.1;%the number of proposals from the second component
         w3_mix=1-w1_mix-w2_mix;%the number of proposals from the third component
         u(2:end,1)=sort(rand(N-1,1));
         id1(2:end,1)=(u(2:end,1)<w1_mix);
         id2(2:end,1)=(u(2:end,1)>w1_mix) & (u(2:end,1)<=(w1_mix+w2_mix));
         id3(2:end,1)=(u(2:end,1)>(w1_mix+w2_mix)) & (u(2:end,1)<=(w1_mix+w2_mix+w3_mix));
         n1=sum(id1);%the number of proposals from the first component
         n2=sum(id2);
         n3=sum(id3);
         id1=logical(id1);
         id2=logical(id2);
         id3=logical(id3);
         particles_res(:,1:N)=particles(:,:,t);
         particles_temp1=chol_cond_var*randn(num_randeffect,n1)+cond_mean(:,id1);
         particles_temp2=param.theta_mu+(param.theta_phi).*(particles_res(:,id2)-param.theta_mu)+...
                 chol(param.theta_sig2,'lower')*randn(num_randeffect,n2);
         particles_temp3=sqrt(scale_covmat)*chol_cond_var*randn(num_randeffect,n3)+cond_mean_ref(:,id3);
         particles_star(:,2:N)=[particles_temp1,particles_temp2,particles_temp3];
         %list of the random effects in the LBA model
         theta_latent_b1min(1,:)=particles_star(1,:);
         theta_latent_b2min(1,:)=particles_star(2,:);
         theta_latent_b3min(1,:)=particles_star(3,:);
         theta_latent_A(1,:)=particles_star(4,:);
         theta_latent_v1(1,:)=particles_star(5,:);
         theta_latent_v2(1,:)=particles_star(6,:);
         theta_latent_tau(1,:)=particles_star(7,:);
         
         %adjust the size of the vectors of the random effects 
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
         
         [theta_latent_bmin_kron]=reshape_b(data_cond_repmat,theta_latent_b1min_kron,theta_latent_b2min_kron,theta_latent_b3min_kron); %choose the threshold particles to match with the conditions of the experiments at block (t+1)
         [theta_latent_v_kron]=reshape_v(data_response_repmat,theta_latent_v1_kron,theta_latent_v2_kron); % set the drift rate particles to match with the response at block (t+1)
         %computing the log of weights
         if t==T-1
            log_weight_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(N*num_within_block{t+1,1},1),theta_latent_tau_kron)));
            log_weight_reshape=reshape(log_weight_temp,num_within_block{t+1,1},N);
            log_weight1=sum(log_weight_reshape);
            mean_com_temp2=(param.theta_mu+(param.theta_phi).*(particles(:,:,t)-param.theta_mu))';
            %mean_com_temp3=(reference_particles+(param.theta_phi*ones(num_randeffect,1)).*(particles(:,:,t)-reference_particles))';
            log_weight2=(logmvnpdf(particles_star',mean_com_temp2,param.theta_sig2))';
            log_weight3=log(w1_mix.*mvnpdf(particles_star',cond_mean',chol_cond_var*chol_cond_var')+...
                    w2_mix.*mvnpdf(particles_star',mean_com_temp2,param.theta_sig2)+...
                    w3_mix.*mvnpdf(particles_star',cond_mean_ref',scale_covmat*(chol_cond_var*chol_cond_var')));    
            log_weight=log_weight1'+log_weight2-log_weight3;
            
         else
            log_weight_temp=real(log(LBA_n1PDF_reparam_real_min(data_rt_repmat, theta_latent_A_kron,theta_latent_bmin_kron, theta_latent_v_kron, ones(N*num_within_block{t+1,1},1),theta_latent_tau_kron)));
            log_weight_reshape=reshape(log_weight_temp,num_within_block{t+1,1},N);
            log_weight1=sum(log_weight_reshape);
            mean_com2=(param.theta_mu+(param.theta_phi).*(particles_star-param.theta_mu))';
            mean_com3=(param.theta_mu+(param.theta_phi).*(particles(:,:,t)-param.theta_mu))';
            %mean_com4=(reference_particles+(param.theta_phi*ones(num_randeffect,1)).*(particles(:,:,t)-reference_particles))';
            log_weight2=(logmvnpdf(sirhat(:,t+2)',mean_com2,param.theta_sig2))';
            log_weight3=(logmvnpdf(particles_star',mean_com3,param.theta_sig2))';
            log_weight4=log(w1_mix.*mvnpdf(particles_star',cond_mean',chol_cond_var*chol_cond_var')+...
                    w2_mix.*mvnpdf(particles_star',mean_com3,param.theta_sig2)+...
                    w3_mix.*mvnpdf(particles_star',cond_mean_ref',(scale_covmat)*(chol_cond_var*chol_cond_var'))); 
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

%         xx1=[particles(:,:,t);mu*ones(1,N);logit_inverse(phi)*ones(1,N);log(tau)*ones(1,N)];
%         xx1_reshape=reshape(xx1,4,1,N);
%         xx1_minus_meanparam=xx1_reshape-mean_param(t+1,2:end)';
%         temp1=multiprod((covmat_param(1:1,2:end,t+1)/covmat_param(2:end,2:end,t+1)),xx1_minus_meanparam);
%         temp1_reshape=reshape(temp1,1,N);
%         cond_mean=mean_param(t+1,1)+temp1_reshape;
%         cond_var=covmat_param(1,1,t+1)-covmat_param(1:1,2:end,t+1)*(covmat_param(2:end,2:end,t+1)\covmat_param(2:end,1:1,t+1));
%         particles_star(1,2:N)=sqrt(cond_var)*randn(1,N-1)+cond_mean(1,2:end);
%         if t==T-1
%            log_weight1=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2);
%            log_weight2=-0.5*log(2*pi)-0.5.*log(tau)-0.5.*(1./tau).*((particles_star-mu-phi.*(particles(:,:,t)-mu)).^2);
%            log_weight3=-0.5*log(2*pi)-0.5.*log(cond_var)-0.5.*(1./cond_var).*((particles_star-cond_mean).^2);
%            log_weight=log_weight1+log_weight2-log_weight3;
%         else
%            log_weight1=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2);
%            log_weight2=-0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*((sirhat(:,t+2)-mu-phi.*(particles_star-mu)).^2);
%            log_weight3=-0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*((particles_star-mu-phi.*(particles(:,:,t)-mu)).^2);
%            log_weight4=-0.5*log(2*pi)-0.5.*log(cond_var)-0.5.*(1./cond_var).*((particles_star-cond_mean).^2);
%            log_weight=log_weight1+log_weight2+log_weight3-log_weight4;
%         end
%         weight=exp(log_weight-max(log_weight));   
%         weight=weight./sum(weight);
%         indx_choose=find(rand(1) < cumsum(weight),1,'first');
%         sirhat(1,t+1)=particles_star(1,indx_choose);
%         sirhat(1,t)=particles(:,indx_choose,t);
%         outndx(t)=indx_choose; 



















%             indx_star=rs_multinomial(w(t,:));
%             indx_star(1,1)=indx(t+1,outndx(t+1));
%             indx(t,:)=indx(t,indx_star);
%             particles(:,:,t)=particles(:,indx_star,t);
%             particles_star(1,1)=sirhat(1,t+1);  
%             particles_star(1,2:N)=mu+phi*(particles(:,2:end,t)-mu)+sqrt(tau)*randn(1,N-1);
%             if t==T-1
%                 log_weight=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2);
%             else
%                 log_weight=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2)-...
%                     0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*((sirhat(:,t+2)-mu-phi.*(particles_star-mu)).^2);
%             end
%             weight=exp(log_weight-max(log_weight));   
%             weight=weight./sum(weight);
%             indx_choose=randsample(N,1,true,weight);
%             sirhat(1,t+1)=particles_star(1,indx_choose);
%             sirhat(1,t)=particles(:,indx_choose,t);
%             outndx(t)=indx_choose;


%           indx_old=indx(t+1,outndx(t+1));
%           indx_star=rs_multinomial_cond(w(t,:));
%           indx_temp1=indx(t,indx_old);
%           indx_temp2=indx(t,indx_star(1,2:end));
%           indx(t,:)=[indx_temp1,indx_temp2];
%           particles_temp1=particles(:,indx_old,t);
%           particles_temp2=particles(:,indx_star(1,2:end),t);
%           particles(:,:,t)=[particles_temp1,particles_temp2];
%           particles_star(1,1)=sirhat(1,t+1);        
%           particles_star(1,2:N)=mu+phi*(particles(:,2:end,t)-mu)+sqrt(tau)*randn(1,N-1);
%          if t==T-1
%             log_weight=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2);
%          else
%             log_weight=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2)-...
%                 0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*((sirhat(:,t+2)-mu-phi.*(particles_star-mu)).^2);
%          end
%          weight=exp(log_weight-max(log_weight));   
%          weight=weight./sum(weight);
%          indx_choose=find(rand(1) < cumsum(weight),1,'first');
%          sirhat(1,t+1)=particles_star(1,indx_choose);
%          sirhat(1,t)=particles(:,indx_choose,t);
%          outndx(t)=indx_choose;



%----------------------------------------------------------------------------------------------------------------------------
% version 1
%       indx_old=indx(t+1,outndx(t+1)); 
%       indx_star=randsample(N,1,true,w(t,:));
%       particles_star=mu+phi*(particles(:,indx_star,t)-mu)+sqrt(tau)*randn(1,1);
% 
%     if t==T-1
%        log_numerator=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2);
%        log_denominator=-0.5*log(2*pi)-0.5.*log(exp(sirhat(1,t+1)))-0.5.*(1./exp(sirhat(1,t+1))).*((y(1,t+1)).^2);
%     else
%        log_numerator=-0.5*log(2*pi)-0.5.*log(exp(particles_star))-0.5.*(1./exp(particles_star)).*((y(1,t+1)).^2)-...
%                0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*((sirhat(:,t+2)-mu-phi.*(particles_star-mu)).^2);
%        log_denominator=-0.5*log(2*pi)-0.5.*log(exp(sirhat(1,t+1)))-0.5.*(1./exp(sirhat(1,t+1))).*((y(1,t+1)).^2)-...
%                0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*((sirhat(:,t+2)-mu-phi.*(sirhat(1,t+1)-mu)).^2);
%     end
%     A1=rand();
%     r1=exp(log_numerator-log_denominator);
%     C1=min(1,r1);
%     if A1<=C1
%        sirhat(1,t+1)=particles_star;
%        sirhat(1,t)=particles(:,indx_star,t);
%        outndx(t)=indx_star;
%     else
%        sirhat(1,t+1)=sirhat(1,t+1);
%        sirhat(1,t)=particles(:,indx_old,t);
%        outndx(t)=indx_old;
%     end























%-----------------------------------------------------------------------------------------------------------------------------------------------------------------
%outndx(t)=indx(t+1,outndx(t+1));
%sirhat(:,t)=particles(:,outndx(t),t);

%    particles_new(1,1)=sirhat(:,t+1);
%    particles_new(1,2:N)=mu+phi*(particles(:,indx_new(t,2:N),t)-mu)+sqrt(tau)*randn(1,N-1);
    
%end
% for t=T-1:-1:1
%     indx_new(t,:)=rs_multinomial_cond_refresh(w(t,:));    
%     particles_new(1,1)=sirhat(:,t+1);
%     particles_new(1,2:N)=mu+phi*(particles(:,indx_new(t,2:N),t)-mu)+sqrt(tau)*randn(1,N-1);    
%     particles_anc(1,1)=sirhat(:,t);
%     particles_anc(1,2:N)=particles(:,indx_new(t,2:N),t);    
%     if t==T-1
%        log_weight_backward=-0.5*log(2*pi)-0.5.*log(exp(particles_new(1,:)))-0.5.*(1./exp(particles_new(1,:))).*((y(1,t+1)).^2);
%     else
%        log_weight_backward=-0.5*log(2*pi)-0.5.*log(exp(particles_new(1,:)))-0.5.*(1./exp(particles_new(1,:))).*((y(1,t+1)).^2)-...
%               0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*(sirhat(:,t+2)-mu-phi.*(particles_new(:,1:N)-mu)); 
%     end    
%     w_backward=exp(log_weight_backward-max(log_weight_backward));  
%     w_backward=w_backward./sum(w_backward);
%     indx_choose=find(rand(1) < cumsum(w_backward),1,'first');
%     sirhat(:,t+1)=particles_new(:,indx_choose);
%     sirhat(:,t)=particles_anc(:,indx_choose);
% end

%     if t==T-1
%        indx_temp=randi(N);
%        indx_new(t,:)=rs_multinomial_cond_refresh(w(t,:));
%        particles_new(:,1,t+1)=particles(:,indx_temp(1,1),t+1);
%        particles_new(:,2:N,t+1)=mu+phi*(particles(:,indx_new(t,2:N),t)-mu)+sqrt(tau)*randn(1,N-1);
%        log_weight_backward=-0.5*log(2*pi)-0.5.*log(exp(particles_new(1,:,t+1)))-0.5.*(1./exp(particles_new(1,:,t+1))).*((y(1,t+1)).^2);
%        w_backward=exp(log_weight_backward-max(log_weight_backward));   
%        w_backward=w_backward./sum(w_backward);
%        indx_choose(t,1)=find(rand(1) < cumsum(w_backward),1,'first');
%        sirhat(:,t+1)=particles_new(:,indx_choose(t,1),t+1);   
%     end
%     if t==1
%        indx_temp=randi(N);
%        indx_new(t,:)=rs_multinomial_cond_refresh(w(t,:));
%        particles_new(:,1,t+1)=particles(:,indx_temp(1,1),t+1); 
%        particles_new(:,2:N,t+1)=sqrt(tau/(1-phi^2))*randn(1,N-1)+mu; 
%        log_weight_backward=-0.5*log(2*pi)-0.5.*log(exp(particles_new(1,:,t+1)))-0.5.*(1./exp(particles_new(1,:,t+1))).*((y(1,t+1)).^2)-...
%            0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*(sirhat(:,t+2)-mu-phi.*(particles_new(:,1:N,t+1)-mu));
%        w_backward=exp(log_weight_backward-max(log_weight_backward));   
%        w_backward=w_backward./sum(w_backward);
%        indx_choose(t,1)=find(rand(1) < cumsum(w_backward),1,'first'); 
%        sirhat(:,t+1)=particles_new(:,indx_choose(t,1),t+1); 
%        sirhat(:,t)=particles(:,indx_choose(t,1),t); 
%     end
%     
%     
%     if t>1 & t<T-1
%        indx_temp=randi(N); 
%        indx_new(t,:)=rs_multinomial_cond_refresh(w(t,:));
%        particles_new(:,1,t+1)=particles(:,indx_temp(1,1),t+1); 
%        particles_new(:,2:N,t+1)=mu+phi*(particles(:,indx_new(t,2:N),t)-mu)+sqrt(tau)*randn(1,N-1); 
%        log_weight_backward=-0.5*log(2*pi)-0.5.*log(exp(particles_new(1,:,t+1)))-0.5.*(1./exp(particles_new(1,:,t+1))).*((y(1,t+1)).^2)-...
%              0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*(sirhat(:,t+2)-mu-phi.*(particles_new(:,1:N,t+1)-mu));
%        w_backward=exp(log_weight_backward-max(log_weight_backward));   
%        w_backward=w_backward./sum(w_backward);
%        indx_choose(t,1)=find(rand(1) < cumsum(w_backward),1,'first'); 
%        sirhat(:,t+1)=particles_new(:,indx_choose(t,1),t+1); 
%     end
    %log_weight_backward=log(w(t,:))-0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/(tau)).*((sirhat(:,t+1)-mu-phi.*(particles(:,:,t)-mu)).^2);
    %w_backward=exp(log_weight_backward-max(log_weight_backward));
    %w_backward=w_backward./sum(w_backward);
    %indx(t,1)=find(rand(1) < cumsum(w_backward),1,'first');
    %sirhat(:,t)=particles(:,indx(t,1),t);

%     if t==1
%         indx_temp(1,1)=randi(N);
%         particles_new(:,1,t)=particles(:,indx_temp(1,1),t);
%         particles_new(:,2:N,t)=sqrt(tau/(1-phi^2))*randn(1,N-1)+mu;
%         log_weight_backward=-0.5*log(2*pi)-0.5.*log(exp(particles_new(1,:,t)))-0.5.*(1./exp(particles_new(1,:,t))).*((y(1,t)).^2)-...
%             0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*(sirhat(:,t+1)-mu-phi.*(particles_new(:,1:N,t)-mu));
%         w_backward=exp(log_weight_backward-max(log_weight_backward));
%         w_backward=w_backward./sum(w_backward);
%         indx_choose(t,1)=find(rand(1) < cumsum(w_backward),1,'first');
%         sirhat(:,t)=particles_new(:,indx_choose(t,1),t);
%     else
%         indx_temp(1,1)=randi(N);
%         indx_new(t-1,:)=rs_multinomial_cond_refresh(w(t-1,:),indx_temp(1,1));
%         particles_new(:,1,t)=particles(:,indx_temp(1,1),t);
%         particles_new(:,2:N,t)=mu+phi*(particles(:,indx_new(t-1,2:N),t-1)-mu)+sqrt(tau)*randn(1,N-1);
%         log_weight_backward=-0.5*log(2*pi)-0.5.*log(exp(particles_new(1,:,t)))-0.5.*(1./exp(particles_new(1,:,t))).*((y(1,t)).^2)-...
%             0.5*log(2*pi)-0.5*log(tau)-0.5.*(1/tau).*(sirhat(:,t+1)-mu-phi.*(particles_new(:,1:N,t)-mu));
%         w_backward=exp(log_weight_backward-max(log_weight_backward));
%         w_backward=w_backward./sum(w_backward);
%         indx_choose(t,1)=find(rand(1) < cumsum(w_backward),1,'first');
%         sirhat(:,t)=particles_new(:,indx_choose(t,1),t);
%     end
