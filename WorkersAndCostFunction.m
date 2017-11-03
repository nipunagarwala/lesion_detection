%% Calculate the number of workers
myCluster = parcluster('local');
PoolSize=myCluster.NumWorkers;

%% New cost function

 % If you have 100 patches, 15 features - size (NormPatchFeat) will be 100*15 - 
 % the average will be done on the first dimension (size=1*15), if you want the 2nd one - just
 % change the '1' in the brackets to '2'. Same with the lesion patches
 % TestPatchFeat - the current specific test patch  - 1*15
 NormalHist=mean(NormPatchFeat,1);
 LesionHist=mean(LesPatchFeat,1);
 
 % the 1st new cost function - 
 force = -nu+lambda1*(TestPatchFeat-NormalHist).^2-lambda2*(TestPatchFeat-LesionHist).^2;

  % the 2nd new cost function - 
  % The Bhattacharyya coefficient is a measure used to compare probability density functions, 
  % and results in a scalar corresponding to the similarity of the two histograms.
  BNorm=sqrt(TestPatchFeat.*NormalHist);
  BLes=sqrt(TestPatchFeat.*LesionHist);
  force = -nu+lambda1*(BNorm.^2)-lambda2*(BLes.^2); %Yezzi
