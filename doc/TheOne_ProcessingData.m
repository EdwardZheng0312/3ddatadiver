close all
clear all
%%%%%%%%%%%%%%%%%%% load data from .h5 file
directory = ('E:\AM-AFM\Elias\06-03-2018');
filename = [directory,'\FFM0013.h5'];
fileinfo = hdf5info(filename);
AMPtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(1));
DRIVEtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(2));
PHASEtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(3));
RAWtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(4));
ZSNSRtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(5));

AMPim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(1));
HEIGHTim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(2));
PHASEim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(3));
ZSNSRim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(4));
METAdata = fileinfo.GroupHierarchy.Attributes.Value.Data;


%%%%%%%%%%%%%%%%%%% USER input
SaveDirectory = ('C:\Users\nako825\Desktop\Hydration Layers\MATLAB processing files and saved data\Force maps\ProcessedData');
DataSaveName = [SaveDirectory,'\07112018_Mica_2M_USC_FFM0015'];
Xnm = 20; Ynm = 20; % NOTE: these values will be overwritten in the next section, by searching through METAdata, What is the size of the image in nanometers?
SLICEpix = 20; % decide which slice be visualized as xz, y = SLICEPIX

Zbin = 0.02*10^-9; % Z bin width (in m) for binning/linearizing ZSNSR fata
Znm = 1.5; % decide max Z above surface to be considered for slicing 
CROP = 15; % How many lines should be neglected from the top and bottom of the data while calculating average tilts
[zSIZE, xSIZE, ySIZE] = size(ZSNSRtot); % 
correctx = 1; % do we plan to correct the drift in the fast scan direction? 0 => no, 1 => yes
ShowSamplePlots = 1; ShowSampleZdrive = 0; % Plot sample 1D force curves extracted from data? Plot sample z/drive profiles?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%% Extract useful METAdata from .h5 file
sf1 = strfind(METAdata,'ThermalQ: '); sf2 = strfind(METAdata,'ThermalFrequency: '); Qfactor = str2double(METAdata(sf1(1)+length('ThermalQ: '):sf2(1)-1));
sf1 = strfind(METAdata,'ThermalFrequency: '); sf2 = strfind(METAdata,'ThermalWhiteNoise: '); FreqRes = str2double(METAdata(sf1(1)+length('ThermalFrequency: '):sf2(1)-1));
sf1 = strfind(METAdata,'DriveAmplitude: '); sf2 = strfind(METAdata,'DriveFrequency: '); AmpDrive = str2double(METAdata(sf1(1)+length('DriveAmplitude: '):sf2(1)-1));
sf1 = strfind(METAdata,'AmpInvOLS: '); sf2 = strfind(METAdata,'UpdateCounter: '); AmpInvOLS = str2double(METAdata(sf1(1)+length('AmpInvOLS: '):sf2(1)-1));
sf1 = strfind(METAdata,'DriveFrequency: '); sf2 = strfind(METAdata,'SweepWidth: '); FreqDrive = str2double(METAdata(sf1(1)+length('DriveFrequency: '):sf2(1)-1));
sf1 = strfind(METAdata,'Initial FastScanSize: '); sf2 = strfind(METAdata,'Initial SlowScanSize: '); Xnm = str2double(METAdata(sf1(1)+length('Initial FastScanSize: '):sf2(1)-1))*10^9; % save in nm
sf1 = strfind(METAdata,'Initial SlowScanSize: '); sf2 = strfind(METAdata,'Initial ScanRate: '); Ynm = str2double(METAdata(sf1(1)+length('Initial SlowScanSize: '):sf2(1)-1))*10^9; % save in nm

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Obtain equivalent height image (for every x,y position, find the maximum z) and correct for sample slope
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
j1 = 1; j2 = ySIZE; i1 = 1; i2 = xSIZE; % crop to pixels of interest, note: often the first line is not good data, as the tip is still trying to feel/find the surface
for j = j1:j2
    for i = i1:i2
        [Zmax(j-j1+1,i-i1+1), indZ(j-j1+1,i-i1+1)] = max(ZSNSRtot(:,i-i1+1,j-j1+1));
        [Dmax(j-j1+1,i-i1+1), indD(j-j1+1,i-i1+1)] = max(DRIVEtot(:,i-i1+1,j-j1+1));      
    end
        % correct for sample slope/tilt in y direction, obtain average Zmax at a given y position, we call this Zdrifty
        Zdrifty(j-j1+1) = mean(Zmax(j-j1+1,:)); Zcorr(j-j1+1,:) = Zmax(j-j1+1,:)-Zdrifty(j-j1+1);
        Ddrifty(j-j1+1) = mean(Dmax(j-j1+1,:)); Dcorr(j-j1+1,:) = Dmax(j-j1+1,:)-Ddrifty(j-j1+1);
end
[~,a] = size(Zcorr);

if correctx == 1 % do we need to correct for x drift? user prompted in the first section
for i = 1:xSIZE
    % now we correct for sample slope/tilt in the x direction
    Zdriftx(i) = mean(Zcorr(CROP:ySIZE-CROP,i)); 
    Ddriftx(i) = mean(Dcorr(CROP:ySIZE-CROP,i)); 
end
DcorrLIN = polyfit(1:xSIZE,Ddriftx,1); DdriftxL = (1:xSIZE)*DcorrLIN(1) + DcorrLIN(2);
ZcorrLIN = polyfit(1:xSIZE,Zdriftx,1); ZdriftxL = (1:xSIZE)*ZcorrLIN(1) + ZcorrLIN(2);
Zcorr(CROP:ySIZE-CROP,:) = Zcorr(CROP:ySIZE-CROP,:)-ZdriftxL;
Dcorr(CROP:ySIZE-CROP,:) = Dcorr(CROP:ySIZE-CROP,:)-DdriftxL;
else
    Zdriftx = zeros(1,xSIZE); Ddriftx = zeros(1,xSIZE);
end

Zcorr = max(max(Zcorr(CROP:end-CROP,CROP:end-CROP))) - Zcorr; % flip Zdata so that height map matches sample, this makes the surface at Z = 0
Dcorr = max(max(Dcorr(CROP:end-CROP,CROP:end-CROP))) - Dcorr; 

%%%%% visualize corrected height map, no processing done in this paragraph, simply imaging
% Image corrected height image, in terms of drive data
figure(1)
clims = [mean2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP))-4*std2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP)) , mean2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP))+4*std2(Dcorr(CROP:ySIZE-CROP,CROP:xSIZE-CROP))];
imagesc(Dcorr, clims), set(gca,'YDir','normal'), daspect([1 1 1]), set(gca,'linewidth',2,'fontsize',14)
xlabel('x (pix)','fontsize',16), ylabel('y (pix)','fontsize',16), title(['height image (',num2str(Xnm),'\times',num2str(Ynm),' nm^2)'])

% plot average drift in y direction zsnsr, y direction drive, x direction both
figure(2)
set(gcf, 'Position', [600, 100, 400, 700])
subplot(3,1,1)
plot(Zdrifty,'linewidth',2), set(gca,'linewidth',2,'fontsize',12), axis([-inf inf -inf inf])
xlabel('y (pix)','fontsize',14), ylabel('average z','fontsize',14), title('sample slope in y direction')
subplot(3,1,2)
plot(Ddrifty,'linewidth',2), set(gca,'linewidth',2,'fontsize',12), axis([-inf inf -inf inf])
xlabel('y (pix)','fontsize',14), ylabel('average drive','fontsize',14), title('sample slope in y direction')
subplot(3,1,3)
plot(DdriftxL,'linewidth',2), set(gca,'linewidth',2,'fontsize',12), axis([-inf inf -inf inf])
hold on, plot(Ddriftx,'linewidth',2), set(gca,'linewidth',2,'fontsize',12), 
xlabel('x (pix)','fontsize',14), ylabel('average z','fontsize',14), title('sample slope in x direction')
legend({'Linear fit','Drive'})

% show original height image
figure(3)
imagesc(Dmax(4:end,:)), set(gca,'YDir','normal'), daspect([1 1 1]), set(gca,'linewidth',2,'fontsize',14)
xlabel('x (pix)','fontsize',16), ylabel('y (pix)','fontsize',16), title('original height image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Adjust all data to correct for sample slope using Zdriftx and Zdrifty
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1:ySIZE 
    for i = 1:xSIZE
        ZSNSRtotCORR(:,i,j) = ZSNSRtot(:,i,j)-Zdriftx(i)-Zdrifty(j); % correct for sample slope in x and y directions
        DRIVEtotCORR(:,i,j) = DRIVEtot(:,i,j)-Ddriftx(i)-Ddrifty(j);
        % DIFF(i,j) = max(ZSNSRtotCORR(:,i,j)) - min(ZSNSRtotCORR(:,i,j)); % amplitude of sinusoidal in every approach and retract, simply for checking purposes
    end
end
% obtain mean Z profile, do not consider edges (first and last 5 points and lines)
ZMEAN = mean(ZSNSRtotCORR(:,5:end-5,5:end-5),2); ZMEAN = mean(ZMEAN,3); % this calculates the average Z profile
DMEAN = mean(DRIVEtotCORR(:,5:end-5,5:end-5),2); DMEAN = mean(DMEAN,3); % this calculates the average Z profile
% stop

%%%%%%% QUALITY CHECK (not processing), Compare the Z profiles from 1D measurements, extracted from the 3D data set, having adjusted for sample tilt
if ShowSampleZdrive == 1 % user prompted if they want to see this or not
figure(4)
hold on
for j = 15:5:50, for i = 15:5:50, plot(ZSNSRtotCORR(:,i,j)), end, end
plot(ZMEAN,'k','linewidth',5), set(gca,'linewidth',2,'fontsize',12), axis([-inf inf -inf inf])
xlabel('t','fontsize',14), ylabel('Zsnsr','fontsize',14), title('examples of Zsnsr profiles')

figure(5)
hold on
for j = 20:10:ySIZE-20, for i = 20:20:xSIZE-20, plot(ZSNSRtotCORR(:,i,j),'b'), plot(DRIVEtotCORR(:,i,j),'r'),end, end
plot(ZMEAN,'k--','linewidth',5),plot(DMEAN,'k--','linewidth',5), set(gca,'linewidth',2,'fontsize',12), axis([-inf inf -inf inf])
xlabel('t','fontsize',14), ylabel('Zsnsr / Drive','fontsize',14), title('examples of Zsnsr profiles')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Bins or "discretize" the Z values used into a unified Z vector that is linear and not sinusoidal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
Zlinear = fliplr(0.2*10^-9:-1*Zbin:min(ZMEAN)); % we use the average Z profile (ZMEAN) to approximate the lower and upper bounds of our unified Z vector. This is the part I am not 100% sure about yet before testing a few more data sets. How do we choose z = 0?
Dlinear = fliplr(0.2*10^-9:-1*Zbin:min(DMEAN)); % we use the average Z profile (ZMEAN) to approximate the lower and upper bounds of our unified Z vector. This is the part I am not 100% sure about yet before testing a few more data sets. How do we choose z = 0?
% Zlinear has the edges of our binning intervals for z, we go down from z = 0.05 nm by 0.01 nm intervals to the minimum of ZMEAN
for j = 1:ySIZE
    for i = 1:xSIZE
        % indZ is for closest surface approach a a given x,y, here we only consider tip approach data
        z = ZSNSRtotCORR(:,i,j); % we create a temporary variable z for the ZSNSR data of the force cruve at position i,j, we actually do not need to do this, but it is easier for me to keep track of "z" insted of something like "ZSNSRtotCORR(1:indZ(i,j),i,j)"
        Y = discretize(z,Zlinear); % now discretize z according to the intervals in Zlinear
        d = DRIVEtotCORR(:,i,j); Yd = discretize(d,Dlinear); 
        for n = 1:length(Zlinear)
            ind = find(Y==n); % find which entries (ind) in Y belong to the nth interval         
            PHASEphase(n,i,j) = mean(PHASEtot(ind,i,j));
            AMPamp(n,i,j) = mean(AMPtot(ind,i,j));
        end
        for n = 1:length(Dlinear)
            indd = find(Yd==n); % find which entries (ind) in Y belong to the nth interval         
            PHASEphaseD(n,i,j) = mean(PHASEtot(indd,i,j));
            AMPampD(n,i,j) = mean(AMPtot(indd,i,j));
        end
    end
end
Zlinear = -1*Zlinear; Dlinear = -1*Dlinear; % fix orientation of Zlinear

%%
% Plot examples of phase vs zsnsr and amp vs zsnsr to check that binning and linearizing was done appropriately
if ShowSamplePlots == 1
ii = [floor(xSIZE/3) floor(xSIZE/2) floor(2*xSIZE/3)]; jj = [floor(ySIZE/3)  floor(ySIZE/2) floor(2*ySIZE/3)];

for n = 1:length(ii)
    figure(n+200)
    set(gcf, 'Position', [600, 100, 400, 700])
    subplot(2,1,1)
    plot(-1*DRIVEtot(1:indD(jj(n),ii(n)),ii(n),jj(n))+Ddriftx(ii(n))+Ddrifty(jj(n)),PHASEtot(1:indD(jj(n),ii(n)),ii(n),jj(n)),'linewidth',2)
    hold on, plot(Dlinear,PHASEphaseD(:,ii(n),jj(n)),'linewidth',2)
    axis([-inf inf -inf inf]), set(gca,'linewidth',2,'fontsize',16)
    xlabel('z (m)','fontsize',19), ylabel('\phi (deg)','fontsize',19)
    TI = ['data at x = ',num2str(ii(n)),' pix, y = ',num2str(jj(n)),' pix'];
    title(TI)
    
    subplot(2,1,2)
    plot(-1*DRIVEtot(1:indD(jj(n),ii(n)),ii(n),jj(n))+Ddriftx(ii(n))+Ddrifty(jj(n)),AMPtot(1:indD(jj(n),ii(n)),ii(n),jj(n)),'linewidth',2)
    hold on, plot(Dlinear,AMPampD(:,ii(n),jj(n)),'linewidth',2)
    axis([-inf inf -inf inf]), set(gca,'linewidth',2,'fontsize',16)
    xlabel('z (m)','fontsize',19), ylabel('A (m)','fontsize',19)
end

end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Test an xy or xz slice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% for xz slice or yz slice
% NOTE: Znm and Zbin defined in the user prompt section
figure(10)
Znm = 1.5; SLICEpix = 40;
[~,Dheight] = min(abs(Dlinear-Znm*10^-9)); % find index of Znm
IM = squeeze(PHASEphaseD(Znm:end,:,SLICEpix)); imagesc(IM), 
set(gca,'linewidth',2,'fontsize',16), axis([-inf inf Dheight length(Dlinear)])
xticks([0 1 2 3]*xSIZE/Xnm), xticklabels({'0','1','2','3'})
yticks(Dheight:(length(Dlinear)-Dheight)/4:length(Dlinear)), yticklabels({num2str(Znm), num2str(3*Znm/4), num2str(Znm/2) , num2str(Znm/4), num2str(0)})
xlabel('x (nm)','fontsize',20)
ylabel('z (nm)','fontsize',20)

figure(11)
[~,Dheight] = min(abs(Dlinear-Znm*10^-9)); % find index of Znm
IM = squeeze(AMPampD(Znm:end,:,SLICEpix)); imagesc(IM), 
set(gca,'linewidth',2,'fontsize',16), axis([-inf inf Dheight length(Dlinear)])
xticks([0 1 2 3]*xSIZE/Xnm), xticklabels({'0','1','2','3'})
yticks(Dheight:(length(Dlinear)-Dheight)/4:length(Dlinear)), yticklabels({num2str(Znm), num2str(3*Znm/4), num2str(Znm/2) , num2str(Znm/4), num2str(0)})
xlabel('x (nm)','fontsize',20)
ylabel('z (nm)','fontsize',20)
%%
STOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%% save processed data
save(DataSaveName,'PHASEphaseD', 'AMPampD', 'Dlinear', 'Dcorr', 'Ddrifty', 'Ddriftx', 'Zbin', 'CROP', 'Xnm', 'Ynm', 'METAdata', 'Qfactor', 'FreqRes','AmpDrive', 'AmpInvOLS', 'FreqDrive')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
