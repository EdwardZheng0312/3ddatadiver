%%%%%%%% This program accesses the data in the individual igor files of a
%%%%%%%% force map, and eventually saves the unprocessed data as is into 3D
%%%%%%%% matrices that could be loaded into a later program
close all
clear all
PHASE = []; ZSNSR = []; PHASE = []; AMP = []; DRIVE = []; RAW = []; % initiate variables to temporarily store observables
NUMline = 63; NUMpts = 63; % number of y pixels and x pixels
for i = 1:NUMline
    % a big part of this program is to simply find the filenames of the files we need to open
    % we need to define the i index (here y) and j index (x)
    % also we need to take care of the fact that the "counter" in the filenames always has enough left-side zeros to make it 4 digits. ex: 0233 or 0015 or 0009
    % also we need to take care of the fact that each variable is stored in a separate file
    if i < 10
      Afilename =  ['D:\AM-AFM\Elias\04-27-2018\FFM0012 04-27-2018\Line000',num2str(i)];
    elseif i < 100
      Afilename =  ['D:\AM-AFM\Elias\04-27-2018\FFM0012 04-27-2018\Line00',num2str(i)];
    else
      Afilename =  ['D:\AM-AFM\Elias\04-27-2018\FFM0012 04-27-2018\Line0',num2str(i)]; 
    end
    for j = 0:NUMpts
       if j < 10
        filenamePHASE =  [Afilename,'Point000',num2str(j),'Phase.ibw'];
        filenameZsnsr =  [Afilename,'Point000',num2str(j),'Zsnsr.ibw'];
        filenameAMP =    [Afilename,'Point000',num2str(j),'Amp.ibw'];
        filenameDrive =  [Afilename,'Point000',num2str(j),'Drive.ibw'];
        filenameRaw =    [Afilename,'Point000',num2str(j),'Raw.ibw'];

       elseif j < 100
        filenamePHASE =  [Afilename,'Point00',num2str(j),'Phase.ibw'];
        filenameZsnsr =  [Afilename,'Point00',num2str(j),'Zsnsr.ibw'];
        filenameAMP =    [Afilename,'Point00',num2str(j),'Amp.ibw'];
        filenameDrive =  [Afilename,'Point00',num2str(j),'Drive.ibw'];
        filenameRaw =    [Afilename,'Point00',num2str(j),'Raw.ibw'];

       else
        filenamePHASE =  [Afilename,'Point0',num2str(j),'Phase.ibw'];
        filenameZsnsr =  [Afilename,'Point0',num2str(j),'Zsnsr.ibw'];
        filenameAMP =    [Afilename,'Point0',num2str(j),'Amp.ibw'];
        filenameDrive =  [Afilename,'Point0',num2str(j),'Drive.ibw'];
        filenameRaw =    [Afilename,'Point0',num2str(j),'Raw.ibw'];

       end
       %%%%%%%%%% finally, we have the names of the files we need to open
       %%%%%%%%%% now let us open them using the Matlab function IBWread that allows us to open igor files
        phase = IBWread(filenamePHASE);
        zsnsr = IBWread(filenameZsnsr);
        amp = IBWread(filenameAMP);
        drive = IBWread(filenameDrive);
        raw = IBWread(filenameRaw);
        
       %%%%%%%%% within these igor files, the data of interest is generically under the variable "y" (ex: phase.y, amp.y ...)
       %%%%%%%%% Let us add the 
        PHASE = [PHASE phase.y];
        ZSNSR = [ZSNSR zsnsr.y];
        AMP = [AMP amp.y];
        DRIVE = [DRIVE drive.y];
        RAW = [RAW raw.y];  
        fclose('all');

    end
    %%%%%%%% The variables PHASE, ZSNSR ... include information for all points in a line (all "x" for a given y)
    %%%%%%%% Here we store those variables in a 3D matrix PHASEtot, ZSNSRtot, etc ... and then move to the next line in the for loop to open the files for the next line and do the same thing
    PHASEtot(:,:,i+1) = PHASE;
    ZSNSRtot(:,:,i+1) = ZSNSR;
    AMPtot(:,:,i+1) = AMP;
    DRIVEtot(:,:,i+1) = DRIVE;
    PHASE = []; ZSNSR = []; AMP = []; DRIVE = [];
    i
end 

%%%%%%%%% Finally, we save the unprocessed 3D matrices for each of our variables
save FFM0012_04272018_PHASEtot.mat PHASEtot
save FFM0012_04272018_AMPtot.mat AMPtot
save FFM0012_04272018_DRIVEtot.mat DRIVEtot
save FFM0012_04272018_ZSNSRtot.mat ZSNSRtot