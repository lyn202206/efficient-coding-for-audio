clc
clear all
close all
addpath(genpath('.\data'));  
addpath('.\cochleagram')



% hammering
[x1,fs1] = audioread('hammering.wav');
y1 = resample(x1,160,441);
player = audioplayer(y1,16000);
play(player)


 % woman speaking
[x2,fs2] = audioread('womam_speaking.wav');
y2 = resample(x2,160,441);
player = audioplayer(y2,16000);
play(player)

% combine the two natrual sounds
y3 = y1+y2;

channel_num = 128;
gf1 = gammatone(y1, channel_num, [80, 8000], 16000);        
V1 = cochleagram(gf1);
V1 = 1e3/max(max(V1))*V1;
[F, N]=size(V1);

channel_num = 128;
gf2 = gammatone(y2, channel_num, [80, 8000], 16000);        
V2 = cochleagram(gf2);
V2 = 1e3/max(max(V2))*V2;


channel_num = 128;
gf3 = gammatone(y3, channel_num, [80, 8000], 16000);        
V3 = cochleagram(gf3);
V3 = 1e3/max(max(V3))*V3;




% parameter settings for the algorithm
T = 16;
K = 12;
% initialization for W and H
W_init = abs(randn(F,K,T)) + 1;
H_init = abs(randn(K,N)) + 1;
N_iter_max = 1000;

V_hat=zeros(size(V3));
for t=0:T-1
    tW = W_init(:,:,t+1);
    tH = shift_t(H_init,t);
    V_hat = V_hat + tW*tH;
end
% % hyper-parameters for initialization

a = 1;
b = 1;

[W, H, b_H_record, a, b, cost, time] = convNMF_vbem_FBI(V3, W_init, H_init, V_hat, N_iter_max, a, b);
[B_order, W_order]= sort( b_H_record(:,end),'descend');


V_hat = zeros(size(V3));
for t=0:T-1
    tW = W(:,:,t+1);
    tH = shift_t(H,t);
    V_hat = V_hat + tW*tH;
end



figure,
subplot(3,2,3)
cochplot(V1,[80,8000]);
title('Cochleagram of Ground Truth "{\it{hammering}}"');
set(gca,'Fontname','Times New Roman');
subplot(3,2,5)
cochplot(V2,[80,8000]);
title('Cochleagram of Ground Truth "{\it{woman speaking}}"');
set(gca,'Fontname','Times New Roman');
subplot(3,2,1)
cochplot(V3,[80,8000]);
title('Cochleagram of Mixture');
set(gca,'Fontname','Times New Roman');


subplot(3,2,4)
V_hat1 = zeros(size(V3));
for t=0:T-1
    tW = W(:,W_order(1),t+1);
    tH = shift_t(H(W_order(1),:),t);
    V_hat1 = V_hat1 + tW*tH;
end
cochplot(V3.*V_hat1./max(V_hat,eps),[80,8000]);
title('Cochleagram Reconstructed by #1 STK');
set(gca,'Fontname','Times New Roman');

subplot(3,2,6)
V_hat2 = zeros(size(V3));
for t=0:T-1
    tW = W(:,W_order(2:end),t+1);
    tH = shift_t(H(W_order(2:end),:),t);
    V_hat2 = V_hat2 + tW*tH;
end
cochplot(V3.*V_hat2./max(V_hat,eps),[80,8000]);
title('Cochleagram Reconstructed by the rest STKs');
set(gca,'Fontname','Times New Roman');



figure,
plotW_color(log(W(:,W_order,:)+1));

figure,
subplot(121)
semilogx(b_H_record','k','LineWidth',1)
title('Trace of Relevance Parameters') 
set(gca,'Fontname','Times New Roman');
subplot(122)
b_H_sort = sort(b_H_record(:,end)','descend');
stem(b_H_sort,'k','LineWidth',1)
title('Relevance Parameters') 
set(gca,'Fontname','Times New Roman');






