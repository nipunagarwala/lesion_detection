% Load your dataset
load('/scratch/users/nipuna1/lesion_data/CTLiverNew.mat')

% Relevant cases
route = 1:length(NewImage);

% not displaying Matlab warnings
w = warning ('off','all');
id = w.identifier;
warning('off',id)
rmpath('folderthatisnotonpath');
% job_number_str = getenv('SLURM_ARRAY_TASK_ID');


% for i=1:length(route)
    
    % Radius of the initial contour (many radii sizes). this is for testing.
    % Exploring how good is the patch classification for differnet
    % initializations of the contour
    
    % myRads = [5:5:30];
    myRads = [5:10:25];
    
    % TODO: MAKE IT 5 after debugging
    % FoldNum=5;
    FoldNum=1;
    
    %% Running the segmentation for every fold
    
    for myFold=1:FoldNum
        
        % Load the trained model checkpoint into python for running the
        % experiments - this function should be called Dictionary
        
        % The FEATURES of the TRAIN SET patches (from last CNN layers), along with labels
        % Dictionary: mxn where m=Number of patches and n=Number of features in
        % each patch
        % TrainLabel: Inside (Label 1), outside (Label 3) or on the boundary (Label 2).
%         [Dictionary,TrainLabel]=Dictionary(croppedImage,NewImage,data, route(TrainCases),Interval,BBdim);

%         dataset_path_list = py.list({'~/Documents/NipunDocuments/CotermYear/test_data'});
%         Dictionary, TrainLabel = py.run.get_features(dataset_path_list);

        Dictionary = zeros(5,5);
        TrainLabel = zeros(5,1);
        
        % dataset_path_list = py.list({'~/Documents/NipunDocuments/CotermYear/test_data'});
        py.matlab_python_api.load_model('/scratch/users/nipuna1/lesion_patched_ct_no_normal_ckpt/');
        % py.matlab_python_api.load_model('/scratch/users/nipuna1/lesion_patched_ct_patch_10_ckpt/model.ckpt-192');
        py.matlab_python_api.collect_data_stats('/scratch/users/nipuna1/lesion_data/lesion_patched_ct_rand_100_dataset/');
        % py.matlab_python_api.collect_data_stats('/scratch/users/nipuna1/lesion_data/lesion_patched_ct_patch_10_dataset/');
%         net = vl_simplenn_tidy(net);
        
        % The chosen test radious that will be run for all test cases in
        % the fold
        for magRadIndex=1:length(myRads)
            
            % fold should include the patient numbers (TEST cases - NOT train) that are relevant to this
            % fold - a subset of numbers from the whole cohort (route)
%             RUN ONLY 1 FOLD INITIALLY!!!
            magRadIndex
            fold{1} = 1:20;
            % fold{2} = 21:40;
            % fold{3} = 41:60
            % fold{4} = 61:80
            % fold{5} = 81:112
            for i=fold{1}(1):fold{1}(end)
                % i = str2num(job_number_str);
                i
                BW=poly2mask(ROIdata{i}.ROI_X,ROIdata{i}.ROI_Y,size(NewImage{i},1),size(NewImage{i},2));
                [LesCorY{i},LesCorX{i}]=find(BW);
                % if i == 7
                %     continue
                % end
                %  Already done pre-processing, so not required. Maybe used later though
                %  Img=adapthisteq(NewImage{route(i)});
                Img = NewImage{route(i)};
                
                r = myRads(magRadIndex);
                
                % DONT CHANGE: Active contour boundaries
                mu=0.15;   % 0.075 Originally 0.15
                lambda_1=2; % 2.5 - higher value - lessescapind for sides - external constraint
                lambda_2=2; % 2 - responsible to pull the curve out
                nu =sqrt(max(max(Img))*255);  % 2.5 - Smoothness - tune this parameter for different images
                NarrowBandSize=1;
                % numIter = 300;
                numIter = 500;
                close all
                
                % PLOT THESE IMAGES AND SEE WHETHER (jc,ic) are inside (at the center) of the lesion
                jc=mean(ROIdata{route(i)}.ROI_X);
                ic=mean(ROIdata{route(i)}.ROI_Y);
                
                % Multiply by 255 for getting a range between 0-255
                ImgGray=double(Img*255);
                [nrow,ncol] =size(Img*255);
                
                
%                 Magnitude inside sdf2circle Parameter is NOT IMPORTANT
                % Generate initial countour using size of image, center coordinates, and radius
                initialLSF = sdf2circle(nrow,ncol,ic,jc,r,1);
                
                %Full Adaptive
                % Every pixel in the image is between 0 and 255. Be careful about double and 255
                % initialLSF: Initila level Set function from above
                
                [phi,~,myLData, myLData2,ParamsVec] = active_contourCNN_adaptive(ImgGray/(max(max(ImgGray)))*255,initialLSF,(nrow+ncol)/2,mu,nu,lambda_1,lambda_2, ...
                    numIter,NarrowBandSize,'on',1, LesCorX{i},LesCorY{i});
                
                PhiFull{i,magRadIndex}=zeros(size(NewImage{route(i)},1),size(NewImage{route(i)},2));
                % PhiFull{i,magRadIndex}(MinY:MaxY,MinX:MaxX)=phi;
                PhiFull{i,magRadIndex}=phi;
                ParamsTot{i,magRadIndex}=ParamsVec;

                fullRData{i,magRadIndex} = myLData;
                checkPhi = PhiFull{i, magRadIndex};
                
                %% Calculate Dice between the manual and the automated segemntation to evaluate the segmentation quality
                %% [Dice] = SegmentationEval(NewImage, ROIdata, route(i),checkPhi);
                %% disp(strcat('Dice Score: ', num2str(myDice1)));
                %disp(strcat('Hausdorff Distance: ', num2str(myHaus1)));
                %% myCNNDiceLoc(i, magRadIndex) = Dice;
                %myCNNHaus(i, magRadIndex) = myHaus1;
            end
        end
        % CODE TO SAVE THE WORKSPACE`
        save(['/scratch/users/nipuna1/saved_workspace/CTLiver/ChanVese/adaptive_win_11_iter_500_patch_10_kl-'  num2str(myFold)],'-v7.3');
        % save(['/scratch/users/nipuna1/saved_workspace/CTLiver/ChanVese/adaptive_win_11_iter_500_patch_10_kl-'  num2str(myFold) '-im_num-' job_number_str],'-v7.3');
    end
% end
