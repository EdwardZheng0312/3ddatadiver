close all
clear all
% load the unprocessed 3D matrices 
load FFM0012_04272018_AMPtot.mat
load FFM0012_04272018_PHASEtot.mat
load FFM0012_04272018_DRIVEtot.mat
load FFM0012_04272018_ZSNSRtot.mat

%%%%%%%%% First step, for every x,y position, find the maximum z. This will allow us to reconstruct the height image: Zmax
%%%%%%%%% Also, obtain average Zmax at a given y position, we call this Zdrifty. This will be used to correct for sample slope
%%%%%%%%% Notice that I do the same for both ZSNSR and DRIVE. In the later steps I use only the Zsnsr data. We still do not know what variable for tip position we will ultimately use. The company suggested DRIVE, but our intuition (and the company's previous suggestion) was ZSNSR
j1 = 2; j2 = 64; i1 = 1; i2 = 64; % crop to pixels of interest, note: often the first line is not good data, as the tip is still trying to feel/find the surface
for j = j1:j2
    for i = i1:i2
        [Zmax(j-j1+1,i), indZ(j-j1+1,i)] = max(ZSNSRtot(:,i,j));
        [Dmax(j-j1+1,i), indD(j-j1+1,i)] = max(DRIVEtot(:,i,j));      
    end
        % correct for sample slope/tilt in vertical direction
        Zdrifty(j-j1+1) = mean(Zmax(j-j1+1,:)); Zcorr(j-j1+1,:) = Zmax(j-j1+1,:)-Zdrifty(j-j1+1);
        Ddrifty(j-j1+1) = mean(Dmax(j-j1+1,:)); Dcorr(j-j1+1,:) = Dmax(j-j1+1,:)-Ddrifty(j-j1+1);
end

% so far, Zmax is the original height image and Zcorr is the height image corrected for the y height gradient
% now we  correct for sample slope/tilt in the x direction
[a,~] = size(Zcorr);
for i = 1:a
    Zdriftx(i) = mean(Zcorr(:,i)); Zcorr(:,i) = Zcorr(:,i)-Zdriftx(i);
    Ddriftx(i) = mean(Dcorr(:,i)); Dcorr(:,i) = Dcorr(:,i)-Ddriftx(i);
end
Zcorr = max(max(Zcorr)) - Zcorr; % flip Zdata so that height map matches sample, this makes the surface at Z = 0, instead of at some arbitrary maximum height

%%%%% visualize corrected height map, no processing done in this paragraph,
%%%%% simply imaging
figure(1)
imagesc(Zcorr), set(gca,'YDir','normal'), daspect([1 1 1])
set(gca,'linewidth',2,'fontsize',14)
xlabel('x (pix)','fontsize',16)
ylabel('y (pix)','fontsize',16)
title('height image (5\times5 nm^2)')

% now we need to adjust ALL the zdata for the sample slope/tilt, not just the Zmax for plotting the height image like we did earlier
% at this point we did the hard work and it is straigthforward, we use the gradient vectors we calculated earlier Zdrfitx and Zdrifty
% The other lines (now commented) DIFF and ZMEAN are just for checking that everything makes sense and the qualiity of the instrument operation, I wanted to see if the amplitude of the sinusoidal varies a lot, and to measure the average Z profile 
% They look good, and I am not sure we care about them anymore
% ZMEAN = zeros(5000,1); m = 0;
for j = 1:1:63
    for i = 1:1:63
        ZSNSRtotCORR(:,i,j) = ZSNSRtot(:,i,j)-Zdriftx(i)-Zdrifty(j); % correct for sample slope in x and y directions
        % DIFF(i,j) = max(ZSNSRtotCORR(:,i,j)) - min(ZSNSRtotCORR(:,i,j)); % amplitude of sinusoidal in every approach and retract, simply for checking purposes
        % ZMEAN = ZMEAN+ZSNSRtotCORR(:,i,j); m = m+1; % acquire data to average Z
    end
end
% ZMEAN = ZMEAN/m; % average Z

%%%%%%% This section is again a quality check for how the instrument is working, and is now completely commented out
%%%%%%% I wanted to compare the Z profiles from all the measurements, especially now that we have done some processing to adjust for sample tilt
%%%%%%% Looks good!
% figure(1)
% for j = 1:4:64
%     for i = 1:4:64
%         plot(ZSNSRtotCORR(:,i,j))
% %         axis([0 5000 -3.3*10^-9 0.3*10^-9]);
%         drawnow
%         pause(0.1)
%     end
% end
% hold on
% plot(ZMEAN,'k','linewidth',5)

%%%%%%% This step bins or "discretizes" the Z values used into a unified Z vector that is actually linear and not sinusoidal
%%%%%%% when you reach here let me know, this works so far, but let us test for a few more samples first before you code this component
Zlinear = fliplr(0:-0.01*10^-9:min(ZMEAN)); % we use the average Z profile to approximate the lower and upper bounds of our unified Z vector. This is the part I am not 100% sure about yet before testing a few more data sets. How do we choose z = 0?
% Zlinear has the edges of our binning intervals for z, we go from z = 0 by 0.01 nm intervals to the minimum of ZMEAN
for j = 1:63
    for i = 1:63
        % If you have the function in python, this whole binning thing I keep talking about is actually only a couple of code lines :P
        z = ZSNSRtotCORR(1:indZ(i,j),i,j); % we create a temporary variable z for the ZSNSR data of the force cruve at position i,j, we actually do not need to do this, but it is easier for me to keep track of "z" insted of something like "ZSNSRtotCORR(1:indZ(i,j),i,j)"
        Y = discretize(z,Zlinear); % now discretize z according to the intervals in Zlinear
        for n = 1:length(Zlinear)
            ind = find(Y==n); % find which entries (ind) in Y belong to the nth interval

            PHASElin{n}(i,j) = mean(PHASEtot(ind,i,j)); % the "nth" entry of our processed PHASE is the average of the "ind" entries in the original PHASE
            AMPlin{n}(i,j) = mean(AMPtot(ind,i,j));
            
            PHASEphase(n,i,j) = mean(PHASEtot(ind,i,j));
            AMPamp(n,i,j) = mean(AMPtot(ind,i,j));
            % Notice that we save the data in two different versions, not sure yet which is better, one is slightly better for xz slices, one is slightly better for xy slices. 
            % Specifically the one with {n}(i,j) is a "cell" and the one with (n,i,j) is a 3D matrix
        end
    end
    j
end
%%%%%%%%%%%%%%%% this step completes the processing


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this paragraph is for imaging, I think an xz slice, no processing done here
figure(2)
clims = [30 140];
imagesc(PHASEphase(1:end,:,40),clims)
set(gca,'linewidth',2,'fontsize',14)
% xticks([0 1 2 3 4 5]*128/5)
% xticklabels({'0','1','2','3','4','5'})
% yticks([3 23 43 63 83])
% yticklabels({'0.8','0.6','0.4','0.2','0'})
xlabel('x (nm)','fontsize',16)
ylabel('z (nm)','fontsize',16)

% this paragraph is for imaging, I think an xy slice, no processing done here
% this paragraph does the job, but is not written in the best way, I would like to change it
clims = [60 110];
for n = 200:length(PHASElin)
    clims = [mean2(PHASElin{n}(:,6:end))-2*std2(PHASElin{n}(:,6:end)) mean2(PHASElin{n}(:,6:end))+2*std2(PHASElin{n}(:,6:end))];
    imagesc(PHASElin{n}(:,6:end))
    text(-10, -10, ['z = ',num2str(Zlinear(n)),' nm'],'fontsize',20);
    set(gca,'YDir','normal'), daspect([1 1 1])
    drawnow
    pause(0.8)
end