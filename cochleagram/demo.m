clc
clear all
[x, fs] = audioread('sp05.wav');
% gf = gammatone(x);        
% hc = meddis(gf);
% cg = cochleagram(hc);
% cochplot(cg);

gf = gammatone(x, 64, [80, 8000], 16000);        
cg = cochleagram(gf);
cochplot(cg);