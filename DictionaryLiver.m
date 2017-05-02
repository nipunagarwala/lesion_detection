function [Dictionary,Label,myData]=DictionaryLiver(croppedImage,bigImage,data,route,interval, dim)

% dim is the zero padded dimension of the bounding box for use in the
% dictionary

AllFeatures=[];
Label = [];

Interval = interval;
for i=1:length(route)
    disp(i);
    im=croppedImage{route{i}};
    markMask = zeros(512);
    markMask=roipoly(cell2mat(bigImage{route{i}}),(data.ROI_X(route{i})),(data.ROI_Y(route{i})));
    XCent=mean(data.ROI_X(route{i}));
    YCent=mean(data.ROI_Y(route{i}));
    MinX= round(XCent) - Interval + 1;
    MinY= round(YCent) - Interval + 1;
    MaxX= round(XCent) + Interval;
    MaxY= round(YCent) + Interval;
    OrigMask = imcrop(markMask,[MinX MinY (MaxX-MinX) (MaxY-MinY)]);
    

    % EDIT: Try different sigmas/alphas to reduce training set redundancy
    sigma = [3, 4, 5, 6, 7, 8];
    alpha = [12, 16, 20, 24, 28, 32];
    
    %Small mask for CNN
    for j = 0.25:0.2:0.85
        %smallMask = imerode(OrigMask, strel('square', j));
        smallMask = imresize(OrigMask, j);
        [m, n] = size(smallMask);
        smallMask = padarray(smallMask, [floor((dim-m)/2) floor((dim-n)/2)], 'replicate','post');
        smallMask = padarray(smallMask, [ceil((dim-m)/2) ceil((dim-n)/2)], 'replicate','pre');
        if(nnz(smallMask) == 0)
            continue;
        end
        newImg = im .* smallMask;
        [m, n] = size(newImg);
        newImg = padarray(newImg, [floor((dim-m)/2) floor((dim-n)/2)], 'replicate','post');
        newImg = padarray(newImg, [ceil((dim-m)/2) ceil((dim-n)/2)], 'replicate','pre');
        AllFeatures = [AllFeatures, newImg(:)];
        Label = [Label, 1];
        for k = 1:19
            imTrans = imTransform(newImg, sigma(randi(6)), alpha(randi(6)),Interval);
            if(nnz(imTrans) ~= 0)
                AllFeatures = [AllFeatures, imTrans(:)];
                Label = [Label, 1];
            end
        end
    end
    
    %smallMask = imerode(OrigMask, strel('square', round(imR/2)));
    %[Features] = haralick(im,smallMask);
    %if (sum(isnan(Features)) == 0)
    %    AllFeatures=[AllFeatures, Features];
    %    Label = [Label, 1];
    % end
    
    %Original mask for CNN
    im
    size(cell2mat(im{0}))
    newImg = cell2mat(im).* OrigMask;
    [m, n] = size(newImg);
    
    % Padding image to maximum dimension of bounding box
    newImg = padarray(newImg, [floor((dim-m)/2) floor((dim-n)/2)], 'replicate','post');
    newImg = padarray(newImg, [ceil((dim-m)/2) ceil((dim-n)/2)], 'replicate','pre');
    AllFeatures = [AllFeatures, newImg(:)];
    Label = [Label, 2];
    
    % Image distortion to enlarge training set
    for k = 1:19
        imTrans = imTransform(newImg, sigma(randi(6)), alpha(randi(6)),Interval);
        if(nnz(imTrans) ~= 0)
            AllFeatures = [AllFeatures, imTrans(:)];
            Label = [Label, 2];
        end
    end
    
    %[Features] = haralick(im,OrigMask);
    %AllFeatures=[AllFeatures, Features];
    %Label = [Label, 2];
    
    for j = 1.2:0.4:2.4
        %bigMask = imdilate(OrigMask, strel('square', j));
        bigMask = imresize(OrigMask, j);
        [m, n] = size(bigMask);
        bigMask = bigMask(m/2 - Interval+1: m/2 + Interval, n/2 - Interval+1: n/2 + Interval);
        newImg = im .* bigMask;
        [m, n] = size(newImg);
        newImg = padarray(newImg, [floor((dim-m)/2) floor((dim-n)/2)], 'replicate','post');
        newImg = padarray(newImg, [ceil((dim-m)/2) ceil((dim-n)/2)], 'replicate','pre');
        AllFeatures = [AllFeatures, newImg(:)];
        Label = [Label, 3];
        for k = 1:19
            imTrans = imTransform(newImg, sigma(randi(6)), alpha(randi(6)),Interval);
            if(nnz(imTrans) ~= 0)
                AllFeatures = [AllFeatures, imTrans(:)];
                Label = [Label, 3];
            end
        end
    end
    
    %Big mask
    % bigMask = imdilate(OrigMask, strel('square', 5));
    %[Features] = haralick(im,bigMask);
    %AllFeatures=[AllFeatures, Features];
    %Label = [Label, 3];
    
    GLCM = graycomatrix(im,'offset',[0 3; -3 3; -3 0; -3 -3],'NumLevels',8,'GrayLimits',[min(im(im>0)) max(im(:))],'Symmetric', true);
    stats = graycoprops(GLCM,{'contrast'});
    myData(i)=mean(stats.Contrast);
end

Dictionary=[AllFeatures];

% pcaData = doPca(AllFeatures,1);
% pcaCoeff = projectOntoEigenVectors(pcaData, AllFeatures, 5);
% disp('Doing kmeans . . . . . . ')
% [clusterIndices,dictionary] = kmeansandrew(50,Dictionary);




