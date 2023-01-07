function [W, H, beta_record, a, b, cost, time] = convNMF_vbem_FBI(V, W_init, H_init, V_hat, N_iter_max, a_init, b_init)
% Written by Yinan Li
% Last updated data 31/08/2020
% The algorithm is a full bayesian inference (FBI) for all the possible variables and
% beta is referred to as relevance parameters,inverse Gamma distribution is imposed on beta 

% Input:
% V : matrix to be approximated
% W_init : initial values for W
% H_init : initial values for H
% V_hat : the approximation of V
% N_iter_max : maximum number of iterations
% a_init: initial shape parameter of the inverse Gamma distribution that control beta
% b_init: initial scale parameter of the inverse Gamma distribution that control beta

% Output:
% W : the resulted W
% H : the resulted H
% beta_record: record the relevant parameters in each iteration
% a: the estimated shape parameter of the inverse Gamma distribution
% b: the estimated scale parameter of the inverse Gamma distribution
% cost: record the value of KL divergence in each iteration
% time: record the value of the CPU time in each iteration

% Initialization
cost = zeros(1,N_iter_max);
time = zeros(1,N_iter_max);
[F,N] = size(V);
[~,K,T] = size(W_init);
C = zeros(T,F,N,K);   % the incorporation of C turned V into compositional model
W = W_init;
H = H_init;
a = a_init;
b = b_init;

beta_record=zeros(K, N_iter_max);
% % shape and scale parameters for Gamma distribution

H_EL = H;  % the exp(<logH>) is initialized with H
W_EL = W;  % the exp(<logW>) is initialized with W

% record the initial value
tic
a_hat = F*T + N + a + 1;  % a number
b_hat = sum(sum(W,3),1)'+ sum(H,2) + b; % a vector of scale Kx1
beta_vec = b_hat/a_hat;
beta_record(:,1) = beta_vec;
cost(1) = eval_D_beta(V,V_hat,1);
time(1) = toc;
% iteration
for iter=2:N_iter_max
    % calculate WH_EL sum(<logW(t)><logH>)
    WH_EL=zeros(size(V));
    for t = 0:T-1
        tW_EL = W_EL(:,:,t+1);
        tH_EL = shift_t(H_EL,t);
        WH_EL = WH_EL + tW_EL*tH_EL;  
    end
    % estimate the component for each basis
    for t = 0:T-1
        tW_EL = W_EL(:,:,t+1);
        tH_EL= shift_t(H_EL,t);
        for k = 1:K
            C(t+1,:,:,k) = reshape((tW_EL(:,k)*tH_EL(k,:) + eps).*V./(WH_EL + T*K*eps),[1,F,N,1]);
        end
    end
    % calculate the sum of C_shift over t and f, which will be used to update parameters such as alpha and beta.
    % the scale of C_shift is T*F*N*K
    % C_shift is a shift version of C, considering the relationship between
    % N amd T in the activation matrix H
    C_shift = zeros(T,F,N,K);
    for t = 0:T-1
        for k = 1:K
            C_slice = reshape(C(t+1,:,:,k),[F,N]);
            C_slice_shift = shift_t(C_slice, -t);
            C_shift(t+1,:,:,k) = reshape(C_slice_shift,[1,F,N,1]);            
        end
    end
    C_shift_t = reshape(sum(C_shift,1),[F,N,K]); %
    C_shift_tf = reshape(sum(C_shift_t,1),[N,K])'; % the scale of C_shift_tf is KxN    
    % calculate the sum of C over n in each time slice;
    C_n = reshape(sum(C,3),[T,F,K]); % use to update W
    % calculate some estimation parameters in the E-step
    a_hat = F*T + N + a + 1;  % a number
    b_hat = sum(sum(W,3),1)'+ sum(H,2) + b; % a vector of scale Kx1
    beta_vec = b_hat/a_hat;
    ibeta_vec = 1./(beta_vec);
    beta_record(:, iter) = beta_vec;

    % estimate W
    A_W = zeros(F,K,T);
    iB_W = zeros(F,K,T);
    B_W = zeros(F,K,T);
    for t=0:T-1
        A_W(:,:,t+1) = 1 + reshape(C_n(t+1,:,:), [F,K]);
        tH = shift_t(H,t);
        iB_W(:,:,t+1) = repmat((ibeta_vec + sum(tH,2))',F,1);
        B_W(:,:,t+1) = 1./(iB_W(:,:,t+1) + eps);
        W(:,:,t+1) = A_W(:,:,t+1).*B_W(:,:,t+1);
        % estimate the W_EL
        W_EL(:,:,t+1) = exp(psi(A_W(:,:,t+1))).*B_W(:,:,t+1);   % calculate exp(<logW>)
    end
    
    % estimate H
    A_H = 1 + C_shift_tf;
    iB_H = zeros(K, N);
    W_sum_FT = sum(sum(W,3),1)'; % a vector of scale Kx1
    for n = 1:N
        if n<=N-T+1
            iB_H(:,n) = ibeta_vec + W_sum_FT;
        else
            W_sum_t = zeros(F,K);
            for t=0:N-n
                W_sum_t = W_sum_t + W(:,:,t+1);
            end
            iB_H(:,n) = ibeta_vec + sum(W_sum_t,1)';
        end
    end
    B_H = 1./(iB_H + eps);
    H = A_H.*B_H;
    H_EL = exp(psi(A_H)).*B_H;                                  % calculate exp(<logH>) % psi¾ÍÊÇdi-gammaº¯Êý
    % 
    % update a and b
    a = a * K * (psi(a_hat)-psi(a))/(sum(log(b_hat/b)));
    b = a / sum(ibeta_vec);
    
    
    % calculate V_hat after each iteration
    V_hat = zeros(size(V));
    for t = 0:T-1
        tW = W(:,:,t+1);
        tH = shift_t(H,t);
        V_hat = V_hat + tW*tH;
    end
    V_hat = max(V_hat,0);
    % record the result for time and cost
    time(iter)=toc;
    cost(iter) = eval_D_beta(V,V_hat,1);
    beta_record(:,iter) = beta_vec;
    disp(['VBEM_FBI, Iteration ',num2str(iter),' , cost = ',num2str(cost(iter))])
end

