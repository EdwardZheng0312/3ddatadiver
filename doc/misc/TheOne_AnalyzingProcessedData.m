%%
% clear all
% close all
% LoadDirectory = ('C:\Users\nako825\Desktop\Hydration Layers\MATLAB processing files and saved data\Force maps\ProcessedData');
% DataLoadName = [LoadDirectory,'\06032018_Mica_10mM_USC_FFM0013'];
% load(DataLoadName)
[zSIZE,xSIZE,ySIZE] = size(PHASEphaseD);
%%
%%%%%%%%%%% This part should eventually be moved to TheOne_ProcessingData
% Amplitude modulated force analysis using Kuehnle equations
% PARAMETERS => k: Spring constant, vEXC: Excitation frequency, vE: Resonance frequency (eigenfrequency), F0: Drive amplitude, Q: Q factor for cantilever
% k = 232.1; % nN/nm
% vEXC = 1.4215e+06; % Hz
% vE = 1.435e+06;  % Hz
% DA = 0.1; % V (drive amplitude)
% AmpInvOLS = 2.548e-08; % m/V
% F0 = DA*AmpInvOLS; 
% Q = 15.383; % 
% 
% % FORCEe = k*DEFL;
% for n = 1:zSIZE
%     for x = 1:xSIZE
%         for y = 1:ySIZE
%             kGRAD(n,x,y) = k*(1 - (vEXC/vE)^2)-F0./AMPampD(n,x,y).*cos(PHASEphaseD(n,x,y)*pi/180);
%             GAMMA(n,x,y) = -k/(2*pi*vE*Q) -F0./(2*pi*vEXC.*AMPampD(n,x,y)).*sin(PHASEphaseD(n,x,y)*pi/180);
%         end
%     end
% end
%%
figure(1)
clims = [mean2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP))-4*std2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP)) , mean2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP))+4*std2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP))];
imagesc(Dcorr, clims), set(gca,'YDir','normal'), daspect([1 1 1]), set(gca,'linewidth',2,'fontsize',14)
xlabel('x (pix)','fontsize',16), ylabel('y (pix)','fontsize',16), title(['height image (',num2str(Xnm),'\times',num2str(Ynm),' nm^2)'])

figure(2)
Znm = 1.5; SLICEpix = 55; 
[~,Dheight] = min(abs(Dlinear-Znm*10^-9)); % find index of Znm
IM1 = squeeze(PHASEphaseD(:,:,SLICEpix)); IM2 = squeeze(PHASEphaseD(:,:,SLICEpix-2)); IM3 = squeeze(PHASEphaseD(:,:,SLICEpix+2));
% IM4 = squeeze(PHASEphaseD(:,:,SLICEpix-1)); IM5 = squeeze(PHASEphaseD(:,:,SLICEpix+1));
% IM = (IM1+IM2+IM3+IM4+IM5)/5;
IMphase = (IM1+IM2+IM3)/3;
imagesc(IMphase), colorbar
set(gca,'linewidth',2,'fontsize',19), axis([-inf inf Dheight length(Dlinear)])
xticks((0:Xnm)*xSIZE/Xnm), xticklabels({'0','1','2','3'})
yticks(Dheight:(length(Dlinear)-Dheight)/4:length(Dlinear)), yticklabels({num2str(Znm), num2str(3*Znm/4), num2str(Znm/2) , num2str(Znm/4), num2str(0)})
xlabel('x (nm)','fontsize',23), ylabel('z (nm)','fontsize',23)
title('Phase shift, \phi (deg)')
stop
figure(3)
IM1 = squeeze(kGRAD(:,:,SLICEpix)); IM2 = squeeze(kGRAD(:,:,SLICEpix-2)); IM3 = squeeze(kGRAD(:,:,SLICEpix+2));
IMkGRAD = (IM1+IM2+IM3)/3;
imagesc(IMkGRAD), colorbar
set(gca,'linewidth',2,'fontsize',16), axis([-inf inf Dheight length(Dlinear)])
xticks((0:Xnm)*xSIZE/Xnm), xticklabels({'0','1','2','3'})
yticks(Dheight:(length(Dlinear)-Dheight)/4:length(Dlinear)), yticklabels({num2str(Znm), num2str(3*Znm/4), num2str(Znm/2) , num2str(Znm/4), num2str(0)})
xlabel('x (nm)','fontsize',20), ylabel('z (nm)','fontsize',20)
title('Force gradient, k (N/m)')

figure(4)
IM1 = squeeze(AMPampD(:,:,SLICEpix)); IM2 = squeeze(AMPampD(:,:,SLICEpix-2)); IM3 = squeeze(AMPampD(:,:,SLICEpix+2));
IMamp = (IM1+IM2+IM3)/3;
imagesc(IMamp), colorbar
set(gca,'linewidth',2,'fontsize',16), axis([-inf inf Dheight length(Dlinear)])
xticks((0:Xnm)*xSIZE/Xnm), xticklabels({'0','1','2','3'})
yticks(Dheight:(length(Dlinear)-Dheight)/4:length(Dlinear)), yticklabels({num2str(Znm), num2str(3*Znm/4), num2str(Znm/2) , num2str(Znm/4), num2str(0)})
xlabel('x (nm)','fontsize',20), ylabel('z (nm)','fontsize',20)
title('Amplitude, A (m)')
stop
%%
% obtain forces from selected sites from xz slice
figure(3)
[a,~] = ginput() ; a = [round(a) round(a)-1 round(a)+1]; FORCE1k = mean(IMkGRAD(:,a),2);
[b,~] = ginput() ; b = [round(b) round(b)-1 round(b)+1]; FORCE2k = mean(IMkGRAD(:,b),2);
FORCE1p = mean(IMphase(:,a),2); FORCE2p = mean(IMphase(:,b),2);
FORCE1a = mean(IMamp(:,a),2); FORCE2a = mean(IMamp(:,b),2);

figure(10)
plot(Dlinear*10^9,FORCE1k,'linewidth',2)
hold on
plot(Dlinear*10^9,FORCE2k,'linewidth',2)
axis([0 Znm -inf inf]), set(gca,'linewidth',2,'fontsize',16)
xlabel('z (nm)','fontsize',19), ylabel('k (N/m)','fontsize',19)

figure(11)
plot(Dlinear*10^9,FORCE1p,'linewidth',2)
hold on
plot(Dlinear*10^9,FORCE2p,'linewidth',2)
axis([0 Znm -inf inf]), set(gca,'linewidth',2,'fontsize',16)
xlabel('z (nm)','fontsize',19), ylabel('\phi (deg)','fontsize',19)

figure(12)
plot(Dlinear*10^9,FORCE1a*10^9,'linewidth',2)
hold on
plot(Dlinear*10^9,FORCE2a*10^9,'linewidth',2)
axis([0 Znm -inf inf]), set(gca,'linewidth',2,'fontsize',16)
xlabel('z (nm)','fontsize',19), ylabel('A (nm)','fontsize',19)
stop
%%
% obtain forces from selected sites from xy height image
figure(1)
[a,~] = ginput() ; a = [round(a) round(a)-1 round(a)+1]; FORCE1k = mean(IMkGRAD(:,a),2);
[b,~] = ginput() ; b = [round(b) round(b)-1 round(b)+1]; FORCE2k = mean(IMkGRAD(:,b),2);
FORCE1p = mean(IMphase(:,a),2); FORCE2p = mean(IMphase(:,b),2);
FORCE1a = mean(IMamp(:,a),2); FORCE2a = mean(IMamp(:,b),2);

figure(15)
plot(Dlinear*10^9,FORCE1k,'linewidth',2)
hold on
plot(Dlinear*10^9,FORCE2k,'linewidth',2)
axis([0 Znm -inf inf]), set(gca,'linewidth',2,'fontsize',16)
xlabel('z (nm)','fontsize',19), ylabel('k (N/m)','fontsize',19)

%%
%%%%%% create animation of xy slices
h1 = figure(20);
Znm_slice = 0.1:0.05:1.2;
OBJ = VideoWriter('XY_Phase_06032018_Mica_10mM_USC_FFM0013.avi'); % create file for making movie
OBJ.FrameRate=7;
open(OBJ); % open movie file
for i = 1:length(Znm_slice)
    ZnmSLICE = Znm_slice(i); [~,Dheight2] = min(abs(Dlinear-ZnmSLICE*10^-9)); % find index of Znm
    IM1 = squeeze(PHASEphaseD(Dheight2,:,:)); IM2 = squeeze(PHASEphaseD(Dheight2-1,:,:)); IM3 = squeeze(PHASEphaseD(Dheight2+1,:,:));
    IM = (IM1+IM2+IM3)/3; IM = (fliplr(IM))'; 
    imagesc(IM) 
    daspect([1 1 1])
    set(gca,'linewidth',2,'fontsize',14), set(gca,'xtick',[],'ytick',[])
    TITLE = [num2str(Xnm),'\times',num2str(Ynm),' nm^2, z = ',num2str(ZnmSLICE),' nm'];    
    title(TITLE,'fontsize',18)
    currFrame = getframe(h1); 
    writeVideo(OBJ,currFrame);  
end
close(OBJ); % close movie
%%
%%%%%% create animation of xz slices
Znm = 1.5; [~,Dheight] = min(abs(Dlinear-Znm*10^-9)); % find index of Znm
Xnm_slice = 0.2:0.1:Xnm-0.2;
h2 = figure(21);
OBJ = VideoWriter('XZ_Phase_06032018_Mica_10mM_USC_FFM0013.avi'); % create file for making movie
OBJ.FrameRate=10;
open(OBJ); % open movie file
for i = 1:length(Xnm_slice)
    XnmSLICE = round( Xnm_slice(i)*xSIZE/Xnm ); if rem(XnmSLICE,2) == 1, XnmSLICE=XnmSLICE+1; end
    IM1 = squeeze(PHASEphaseD(:,:,XnmSLICE)); IM2 = squeeze(PHASEphaseD(:,:,XnmSLICE-2)); IM3 = squeeze(PHASEphaseD(:,:,XnmSLICE+2));
    IM = (IM1+IM2+IM3)/3; 
%     clims = [-2 8];
    imagesc(IM)%, clims), 
    set(gca,'linewidth',2,'fontsize',16), axis([-inf inf Dheight length(Dlinear)])
    xticks((0:Xnm)*xSIZE/Xnm), xticklabels({'0','1','2','3'})
    yticks(Dheight:(length(Dlinear)-Dheight)/4:length(Dlinear)), yticklabels({num2str(Znm), num2str(3*Znm/4), num2str(Znm/2) , num2str(Znm/4), num2str(0)})
    xlabel('x (nm)','fontsize',20), ylabel('z (nm)','fontsize',20)
    TITLE = ['y = ',num2str(Xnm_slice(i)),' nm'];    
    title(TITLE,'fontsize',18)
    drawnow
    writeVideo(OBJ,getframe(h2));  
end
close(OBJ); % close movie
