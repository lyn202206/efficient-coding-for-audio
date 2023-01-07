function plotW(W)
% 
% s=size(W);
% 
% %%%%%  方式1%%%%%%%%
% d=s(2);
% for idx = 1:d
%     subplot(1,d,idx); 
%     if size(W,3)==1
%         plot(1:size(W,1),W(:,idx));
%         axis xy
%     else
%         imagesc(squeeze(W(:,idx,:))); 
%         axis xy
%     end
%     colormap(1-gray)
% end

%%%%%  方式2%%%%%%%%
% W0=zeros(s(1),s(3));
% for i=1:s(2)
%     W0=[W0 squeeze(W(:,i,:))];
% end
% figure
% imagesc(W0(:,s(3)+1:end));
% % axis([0 s(1) 0 s(2)*s(3)])
% axis xy;
% % xlabel('time (seconds)','fontsize',16);
% % ylabel('frequency (Hz)','fontsize',16);
% % dynamicData = max(caxis,-35);
% % caxis(dynamicData)
% % colorbar

%%%%%  方式3 %%%%%%%%
d=size(W,2);
prair=0.01; % percent air between plots;
vt=(size(W,1));
ht=(d*size(W,3));
y_W = size(W,3)/ht*(1-prair);
y_W1 = size(W,1)/vt*(1-prair);
air2=prair/(d+3); % Vertical air space
air1=prair/(d+3); % Horizontal air space

h = axes('position', [0 0 1 1]);
set(h, 'Visible', 'off');
for k=1:d
    h = axes('position', [k*air1+(k-1)*y_W 2*air2 y_W y_W1]);
    imagesc(squeeze(W(:,k,:))); 
    colormap(1-gray)
    axis xy;
    set(h, 'XTick', []);
    set(h, 'yTick', []);
    set(h, 'XAxisLocation', 'top');
%     xlabel('\tau');
end

