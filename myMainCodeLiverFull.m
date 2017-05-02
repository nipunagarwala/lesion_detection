clear TempImg contrasts
RelCases=([1:4 6:7 9:14 16:21 29:39 42:43 47:50 52:64 67:77 (78) 81:85 87:90 92:98 (99) 100:107 109:114 116:122 125:132 139:142 144:146 148:154 156:158 160:165 167 170:174 176:181 183:189 191:193 196:208]);
route=RelCases;


w = warning ('off','all');
id = w.identifier;
warning('off',id)
rmpath('folderthatisnotonpath');

% Getting maximum dimensions of bounding box
maxBBx = 0;
maxBBy = 0;

% log of changes - prevented non-zero images after erosion of mask, changed
% rotation angle to 10 degrees, prevented non-zero images after imtransform
% erosion, disallowed zero images

%imRad = zeros(1, 164);
%route = [1:112];

Interval=50; % for CT
% Interval = 26; % for MRI

for i=1:length(route)
    
    % specific for CT dataset
%       data.ROI_X{route(i)} = newDB{1,i}.roi.xnew-double(newDB{1,i}.offset.x);
%       data.ROI_Y{route(i)} = newDB{1,i}.roi.ynew-double(newDB{1,i}.offset.y);
%       NewImage{route(i)} = double(newDB{i}.image);
    
    imR=sqrt(((max(data.ROI_X{route(i)})-min(data.ROI_X{route(i)}))).^2+((max(data.ROI_Y{route(i)})-min(data.ROI_Y{route(i)}))).^2)/3;
    imRad(i) = imR;
    close all
    mask=zeros(size(NewImage{1},1),size(NewImage{1},2));
    Img=adapthisteq(NewImage{route(i)});
    %     data=OriginalROIdata;
    
    %% Input - Bounding box
    XCent=mean(data.ROI_X{route(i)});
    YCent=mean(data.ROI_Y{route(i)});
    MinX= round(XCent) - Interval + 1;
    MinY= round(YCent) - Interval + 1;
    MaxX= round(XCent) + Interval;
    MaxY= round(YCent) + Interval;
    
    %% Input - Center and boundary points
    
    mask(MinY:MaxY,MinX:MaxX)=1;
    Idx=find(data.ROI_X{route(i)}==min(data.ROI_X{route(i)}));
    croppedImage{route(i)}=imcrop(Img,[MinX MinY (MaxX-MinX) (MaxY-MinY)]);
    
    maxBBx = max(maxBBx, size(croppedImage{route(i)}, 1));
    maxBBy = max(maxBBy, size(croppedImage{route(i)}, 2));
    
    markMask = zeros(512);
    markMask=roipoly(NewImage{route(i)},(data.ROI_X{route(i)}),(data.ROI_Y{route(i)}));
    markedMasks{route(i)} = imcrop(markMask,[MinX MinY (MaxX-MinX) (MaxY-MinY)]);

    % code for showing each image
    
%     ImgGray=double(croppedImage{route(i)}*255);
%     imshow(ImgGray/max(max(ImgGray)));
%     hold on;
%     croppedPhi = imcrop(PhiFull{i,tempRad}, [MinX MinY (MaxX-MinX) (MaxY-MinY)]);
%     contour(croppedPhi(:,:,1),[0 0],'y-','LineWidth',2);
%     
%     [mask,x,y] = roipoly(NewImage{route(i)},(data.ROI_X{route(i)}),(data.ROI_Y{route(i)}));
%     x = x - MinX+1;
%     y = y - MinY+1;
%     plot(x,y,'g-');
%     hold off;
    
%     %Added
%      hold on;
%      ic = abs(YCent-MinY);
%      jc = abs(XCent-MinX);
%      ImgGray=double(croppedImage{route(i)}*255);
%      [nrow,ncol] =size(croppedImage{route(i)}*255);
%      initialLSF = sdf2circle(nrow,ncol,ic,jc,3*5,1);
%      contour(initialLSF(:,:,1),[0 0],'m-','LineWidth',2);
%      
%      croppedPhi = imcrop(PhiNonAD{i,1}, [MinX MinY (MaxX-MinX) (MaxY-MinY)]);
%      contour(croppedPhi(:,:,1),[0 0],'c-','LineWidth',2);
%      title(['iteration#',num2str(i),' Rad = ', num2str(3*5)]);
end

% TrainCases=[1 3 6 10 13 15 18 21 22 24 27 30 33 35 38 41 42 45 53 55 60 64 65 67 69 71 75 83 86 94 98 99 100 105 107 111 112 116 121 123 124 126 128 132 135 136 139 145 150 151 153 158 161 163 164];
%addpath('/Users/aiflab/Desktop/Rishi/MRI_ALW_Final/CNN/matconvnet-1.0-beta13/examples');
%addpath('/Users/aiflab/Desktop/Rishi/MRI_ALW_Final/CNN/matconvnet-1.0-beta13/matlab');
%run vl_setupnn;

BBdim = max(maxBBx, maxBBy);
if(mod(BBdim, 2) ~= 0)
    BBdim = BBdim + 1;
end

% 10 - fold cross validation
folds = round(linspace(0, 164, 11));
len = length(route);

dicefold = cell(1, 10);
hausfold = cell(1, 10);

myCNNDiceLoc = zeros(164);

myRads = [5:5:30];
for myFold=1:10
    fold = (folds(myFold) + 1):folds(myFold + 1);
    TrainCases = horzcat(1:folds(myFold), folds(myFold + 1) + 1:len);
    %load(strcat('Results4', num2str(myFold)));
    
    %load(strcat('MRIChan', num2str(myFold)));
    %myRads = [5];
    
    % create the training set for the CNN
    [Dictionary,TrainLabel,myContrast]=DictionaryLiver(croppedImage,NewImage,data, route(TrainCases),Interval,BBdim);
    
    meanContrast = mean(myContrast);
    
    % train the CNN - returns the trained network and the image database
    [net, imdb] = train_cnn(Dictionary, TrainLabel, BBdim);
    
    net = vl_simplenn_tidy(net);
    
    for magRadIndex=1:length(myRads)
        magRadIndex
        for i=fold(1):fold(end)
            %                    i = 1;
            mask=zeros(size(NewImage{1},1),size(NewImage{1},2));
            Img=adapthisteq(NewImage{route(i)});
            %%% Input - Bounding box
            XCent=mean(data.ROI_X{route(i)});
            YCent=mean(data.ROI_Y{route(i)});
            MinX=round(XCent)-Interval + 1;
            MinY=round(YCent)-Interval + 1;
            MaxX=round(XCent)+Interval;
            MaxY=round(YCent)+Interval;
            MinManCorX{i}=MinX;
            MinManCorY{i}=MinY;
            mask2=zeros(size(NewImage{1},1),size(NewImage{1},2));
            mask2(MinY:MaxY,MinX:MaxX)=1;
            mask=imcrop(mask2,[MinX MinY (MaxX-MinX) (MaxY-MinY)]);
            r = myRads(magRadIndex);
            
            mu=0.15;   % 0.075 Originally 0.15
            lambda_1=2; % 2.5 - higher value - lessescapind for sides - external constraint
            lambda_2=2; % 2 - responsible to pull the curve out
            nu =sqrt(max(max(Img))*255);  % 2.5 - Smoothness - tune this parameter for different images
            NarrowBandSize=1;
            numIter = 300;
            close all
            
            ic = abs(YCent-MinY);
            jc = abs(XCent-MinX);
            ImgGray=double(croppedImage{route(i)}*255);
            [nrow,ncol] =size(croppedImage{route(i)}*255);
            initialLSF = sdf2circle(nrow,ncol,ic,jc,r,1);
            
            
            
            %Full Adaptive
             [phi,~,myLData, myLData2,ParamsVec] = active_contourCNN(ImgGray/(max(max(ImgGray)))*255,initialLSF,(nrow+ncol)/2,mu,nu,lambda_1,lambda_2, ...
                 numIter,NarrowBandSize,'on',1,2,i,route(i),ic,jc,r, meanContrast, net,BBdim, imdb);
            PhiFull{i,magRadIndex}=zeros(size(NewImage{route(i)},1),size(NewImage{route(i)},2));
            PhiFull{i,magRadIndex}(MinY:MaxY,MinX:MaxX)=phi;
            
            
            
            fullRData{i,magRadIndex} = myLData;
            
            checkPhi = PhiFull{i, magRadIndex};
            
            [myDice1] = SegmentationComparisonAd(NewImage, data, route(i), checkPhi);
            disp(strcat('Dice Score: ', num2str(myDice1)));
            %disp(strcat('Hausdorff Distance: ', num2str(myHaus1)));
            myCNNDiceLoc(i, magRadIndex) = myDice1;
            %myCNNHaus(i, magRadIndex) = myHaus1;
        end
    end
    % save the results after each fold
    %save(['/Users/arjunsubramaniam/Desktop/RubinLab/CTFinal/CTChan+3+3'  num2str(myFold)],'-v7.3');
end