% This code performs the numerical example from Section using the 
% spectrogram corresponding to the dataset `Mary has a little lamb'. 
clear all; clc; 
close all;
addpath(genpath('.\data'));  
rng(12345); 
fprintf('******************************************************************************\n')
fprintf('Running the proposed FBI method on  \n'); 
fprintf('It computes rank-6 NMFs of the audio dataset `Mary had a little lamb''.\n'); 
fprintf('The sequence of `Mary had a little lamb''composed of three notes, namely E4, D4 and C4.\n'); 
fprintf('******************************************************************************\n')

% load data matrix
load('piano_Mary.mat'); 
% initialization of data

[F, N]=size(V);
T = 1;
K = 6;
% parameters for the input signal
fs = 16000;
time_end = length(x)/fs;
t_vec=linspace(0,time_end,N);


% initialization for W and H
W_init = abs(randn(F,K,T)) + 1;
H_init = abs(randn(K,N)) + 1;
N_iter_max = 1000;


V_hat=zeros(size(V));
for t=0:T-1
    tW = W_init(:,:,t+1);
    tH = shift_t(H_init,t);
    V_hat = V_hat + tW*tH;
end
% % hyper-parameters for initialization
a = 1;
b = 1;

[W, H, b_H_record, a, b, cost, time] = convNMF_vbem_FBI(V, W_init, H_init, V_hat, N_iter_max, a, b);
[B_order, W_order]= sort( b_H_record(:,end),'descend');


V_hat = zeros(size(V));
for t=0:T-1
    tW = W(:,:,t+1);
    tH = shift_t(H,t);
    V_hat = V_hat + tW*tH;
end

figure,
plotW_color(log(W(:,W_order,:)+1));

figure,
subplot(221)
imagesc(log(V+1))
title('Observation')
set(gca,'Fontname','Times New Roman');
axis xy
subplot(223)
imagesc(log(V_hat+1))
title('Reconstruction')
set(gca,'Fontname','Times New Roman');
axis xy
subplot(122)
b_H_sort = sort(b_H_record(:,end)','descend');
stem(b_H_sort,'r','LineWidth',2)
title('Value of relevance parameters')
set(gca,'Fontname','Times New Roman');



H_max=max(max(H));
% figure,
% for ii=1:K
%     subplot(K,1,ii)
%     plot(t_vec,H(W_order(ii),:))
%     xlim([0,time_end])
%     ylim([0,1.1*H_max])
% end


H_sort=H(W_order,:);
H_sort_bias = H_sort;
for i=1:K
    H_sort_bias(i,:) = H_sort_bias(i,:) + (K-i)*H_max;
end

figure,
plot(t_vec,H_sort_bias)
xlim([0,time_end])
title('The Activation') 
set(gca,'ytick',[],'ycolor','w')
set(gca,'Fontname','Times New Roman');


figure,
semilogx(b_H_record','k','LineWidth',1)
xlabel('Number of iterations')
ylabel('Value of relevance parameters')
title('Trace of Relevance Parameters') 
set(gca,'Fontname','Times New Roman');


% ylim([0,1.1*H_max])