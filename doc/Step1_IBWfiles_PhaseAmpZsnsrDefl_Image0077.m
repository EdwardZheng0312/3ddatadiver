%%%%%%%%%%% This code loads individual .ibw files 
%%%%%%%%%%% Requires ibw reader to import igor files into Matlab

close all
clear all

% initiate variables for each of the observables being saved
ZSNSR = []; PHASE = []; AMP = []; DRIVE = []; RAW = [];

for i = 0:127
    % define first portion of filename to be opened, including directory, measurement number, and what line is being scanned (first position index)
    if i < 10
      Afilename =  ['D:\AM-AFM\SaveData_Image0077\Line000',num2str(i)];
    elseif i < 100
      Afilename =  ['D:\AM-AFM\SaveData_Image0077\Line00',num2str(i)];
    else
      Afilename =  ['D:\AM-AFM\SaveData_Image0077\Line0',num2str(i)]; 
    end
    
    % define second portion of filename to be opened, including what point is being scanned (second position index) is being scanned, and the observable being recorded
    for j = 0:127
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
        phase = IBWread(filenamePHASE);
        zsnsr = IBWread(filenameZsnsr);
        amp = IBWread(filenameAMP);
        drive = IBWread(filenameDrive);
        raw = IBWread(filenameRaw);
        
        % add data from this Point (j) into a matrix that includes all data from Line (i)
        PHASE = [PHASE phase.y];
        ZSNSR = [ZSNSR zsnsr.y];
        AMP = [AMP amp.y];
        DRIVE = [DRIVE drive.y];
        RAW = [RAW raw.y];  
        fclose('all');

    end
    % add saved data from Line (i) which now includes data from all Points (j) into 3D matrix 
    PHASEtot(:,:,i+1) = PHASE;
    ZSNSRtot(:,:,i+1) = ZSNSR;
    AMPtot(:,:,i+1) = AMP;
    DRIVEtot(:,:,i+1) = DRIVE;
    PHASE = []; ZSNSR = []; AMP = []; DRIVE = [];
    i
end 

stop % comment out this line, if interested in saving the data for Step 2 and beyond
save Image0077_PHASEtot.mat PHASEtot
save Image0077_AMPtot.mat AMPtot
save Image0077_DRIVEtot.mat DRIVEtot
save Image0077_ZSNSRtot.mat ZSNSRtot