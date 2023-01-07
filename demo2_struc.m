clc
close all
addpath(genpath('.\data'));  
load VV_data
V=max(VV, eps);
[F,N]=size(V);
T = 20;
K = 10;

rng(888)
% initialization for W and H
W_init = abs(randn(F,K,T))+1;
H_init = abs(randn(K,N))+1;
N_iter_max = 1000;

V_hat=zeros(size(V));
for t=0:T-1
    tW = W_init(:,:,t+1);
    tH = shift_t(H_init,t);
    V_hat = V_hat + tW*tH;
end
% parameters
mean_V = sum(V(:))/(F*N); % Data sample mean per component
% % hyper-parameters for initialization
a = 1;
b = 1;


[W, H, b_H_record, a, b, cost, time] = convNMF_vbem_FBI(V, W_init, H_init, V_hat, N_iter_max, a, b);
[B_order, W_order]= sort( b_H_record(:,end),'descend');

figure,
subplot(121)
semilogy(cost,'k','LineWidth',2)
xlabel('Iteration #')
ylabel('Objective function')
set(gca,'Fontname','Times New Roman');

subplot(122)
semilogy(time,cost,'r','LineWidth',2)
xlabel('Time (s)')
ylabel('Objective function')
set(gca,'Fontname','Times New Roman');

figure,
imagesc(V)
axis xy
colormap(1-gray);
xlabel('Index for columns')
ylabel('Index for rows')
set(gca,'Fontname','Times New Roman');

figure,
plotW(log(W(:,W_order,:)+1));


b_H_record = b_H_record(W_order,:);

figure,
subplot(121)
% semilogy((b_H_record+1)','k','LineWidth',1)
semilogx(b_H_record','k','LineWidth',1)
xlabel('Number of iterations')
ylabel('Value of relevance parameters')
set(gca,'Fontname','Times New Roman');



subplot(122)
stem(b_H_record(:,end)','k','LineWidth',1)
xlabel('Index of relevance parameters')
ylabel('Value of relevance parameters')
set(gca,'Fontname','Times New Roman');


