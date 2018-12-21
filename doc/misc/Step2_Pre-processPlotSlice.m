%%%%%%%%%%%%%% This code loads the data saved in Step 1, pre-processes it,
%%%%%%%%%%%%%% and prepared for plotting slices

close all
clear all
load Image0077_PHASEtot.mat
load Image0077_AMPtot.mat
load Image0077_ZSNSRtot.mat
load Image0077_DRIVEtot.mat

%%%%%%%%% What is the height image? Is it the lowest extension of Zsnsr?
j1 = 1; j2 = 128; i1 = 1; i2 = 128; % crop to pixels of interest
for j = j1:j2
    for i = i1:i2
        [Zmax(j-j1+1,i), indZ(j-j1+1,i)] = max(ZSNSRtot(:,i,j));
        [Dmax(j-j1+1,i), indD(j-j1+1,i)] = max(DRIVEtot(:,i,j));
        [Pmin(j-j1+1,i), indP(j-j1+1,i)] = min(PHASEtot(:,i,j));   
        [Amin(j-j1+1,i), indA(j-j1+1,i)] = min(AMPtot(:,i,j));        
    end
        % correct for sample slope/tilt in vertical direction
        Zdrifty(j-j1+1) = mean(Zmax(j-j1+1,:)); Zcorr(j-j1+1,:) = Zmax(j-j1+1,:)-Zdrifty(j-j1+1);
        Ddrifty(j-j1+1) = mean(Dmax(j-j1+1,:)); Dcorr(j-j1+1,:) = Dmax(j-j1+1,:)-Ddrifty(j-j1+1);
end

% correct for sample slope/tilt in horizontal direction
[a,~] = size(Zcorr);
for i = 1:a
    Zdriftx(i) = mean(Zcorr(:,i)); Zcorr(:,i) = Zcorr(:,i)-Zdriftx(i);
    Ddriftx(i) = mean(Dcorr(:,i)); Dcorr(:,i) = Dcorr(:,i)-Ddriftx(i);
end

% The important result from this section is ZSNSRtotCORR, which is the Z data corrected for sample slope, 
% Another important piece is ZMEAN, which averages all ZSNSRtotCORR data, and helps us decide the bounds and increment of the linearized ZSNSR vector to be used later
ZMEAN = zeros(5000,1); m = 0;
for j = 1:1:128
    for i = 1:1:128
        ZSNSRtotCORR(:,i,j) = ZSNSRtot(:,i,j)-Zdriftx(i)-Zdrifty(j); % correct for sample slope in x and y directions
        DIFF(i,j) = max(ZSNSRtotCORR(:,i,j)) - min(ZSNSRtotCORR(:,i,j)); % measure amplitude of sinusoidal in every approach and retract, for checking purposes
        ZMEAN = ZMEAN+ZSNSRtotCORR(:,i,j); m = m+1; % acquire data to average Z
    end
end
ZMEAN = ZMEAN/m; % average Z

% %%%%%%%%%%%%%%%%%%%% This section is for bookkeeping and testing the quality of the data, now commented out
% %%%%% height map, can plot same based on Dcorr (Drive) instead of Zcorr (Zsnsr),
% %%%%% which is better, Zsnsr or Drive? Not sure yet...
% figure(1)
% Zcorr = max(max(Zcorr)) - Zcorr; % flip Zdata so that height map matches sample
% imagesc(Zcorr(6:end,:)), set(gca,'YDir','normal'), daspect([1 1 1])
% set(gca,'linewidth',2,'fontsize',14)
% xlabel('x (pix)','fontsize',16)
% ylabel('y (pix)','fontsize',16)
% title('height image (5\times5 nm^2)')

% % plot corrected Zsnsr data, compare it to averaged Zsnsr data ... 
% figure(1)
% for j = 6:4:128
%     for i = 1:4:128
%         plot(ZSNSRtotCORR(:,i,j))
%         axis([0 5000 -3.3*10^-9 0.3*10^-9]);
%         drawnow
%         pause(0.1)
%     end
% end
% hold on
% plot(ZMEAN,'k','linewidth',5)

% This piece bins and linearizes the ZSNSRtotCORR data 
% We also group the corresponding PHASE and AMP data
% After this piece, we are ready to slice sections from the 3D data
%ready to slice 
Zlinear = fliplr(0:-0.01*10^-9:min(ZMEAN));
for j = 1:128
    for i = 1:128
        z = ZSNSRtotCORR(1:indZ(i,j),i,j);
        Y = discretize(z,Zlinear);
        for n = 1:length(Zlinear)
            ind = find(Y==n); 
            PHASElin{n}(i,j) = mean(PHASEtot(ind,i,j));
            AMPlin{n}(i,j) = mean(AMPtot(ind,i,j));
            PHASEphase(n,i,j) = mean(PHASEtot(ind,i,j));
            AMPamp(n,i,j) = mean(AMPtot(ind,i,j));
        end
    end
    j
end
clims = [30 140];
figure(2)
imagesc(PHASEphase(180:end,:,10),clims)
set(gca,'linewidth',2,'fontsize',14)
xticks([0 1 2 3 4 5]*128/5)
xticklabels({'0','1','2','3','4','5'})
yticks([3 23 43 63 83])
yticklabels({'0.8','0.6','0.4','0.2','0'})
xlabel('x (nm)','fontsize',16)
ylabel('z (nm)','fontsize',16)
stop

% This piece is for xy slices, currently not written in the most efficient way
clims = [60 110];
for n = 200:length(PHASElin)
    clims = [mean2(PHASElin{n}(:,6:end))-2*std2(PHASElin{n}(:,6:end)) mean2(PHASElin{n}(:,6:end))+2*std2(PHASElin{n}(:,6:end))];
    imagesc(PHASElin{n}(:,6:end))
    text(-10, -10, ['z = ',num2str(Zlinear(n)),' nm'],'fontsize',20);
    set(gca,'YDir','normal'), daspect([1 1 1])
    drawnow
    pause(0.8)
end
