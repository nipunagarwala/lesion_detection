function [phi,Rad,Data1, Data2,ParamsVec] =active_contourCNN_adaptive(im,phi,ObjectSize,mu,nu, lambda1, lambda2,numIter,NarrowBandSize, ...
    print,normalize_force_flag, x_gnd_truth, y_gnd_truth)

% The code is chan and vese with the improvements of narrow-band and normalization

% DO NOT NEED THE FOLLOWING PARAMETERS
%  big mu - less curve
% shift operations
shiftD = @(M) M([1 1:end-1],:,:);
shiftL = @(M) M(:,[2:end end],:);
shiftR = @(M) M(:,[1 1:end-1],:);
shiftU = @(M) M([2:end end],:,:);
% derivatives
Dx = @(M) (shiftL(M) - shiftR(M))/2;
Dy = @(M) (shiftU(M) - shiftD(M))/2;
[dimy, dimx] = size(phi);
ii=1;    % init iterations
% ObjectSizeX = mean([dimy dimx]);
ObjectSizeX = 20;
% compute Global Contrast and Correlation
PxDist=1;
GLCM = graycomatrix(im,'offset',[0 PxDist; -PxDist PxDist; -PxDist 0; -PxDist -PxDist],'GrayLimits',[min(im(:)) max(im(:))], 'Symmetric', true);
stats = graycoprops(GLCM,{'contrast','correlation'});
WholeCont=mean(stats.Contrast); % vector -> one value per angle
WholeHomo=mean(stats.Correlation); % vector -> one value per angle % homogenity

% parfor init
force = 0;
if strcmp(print,'on')
    figure
end

while (ii<=numIter)
    % ii
    idx = [];
    % init narrow band size
    idx = find(phi <=NarrowBandSize & phi >= -NarrowBandSize);  % get the curve's narrow band indexes
    
    while (isempty(idx))
        NarrowBandSize=NarrowBandSize+0.05;
        idx = find(phi <=NarrowBandSize & phi >= -NarrowBandSize);
    end
    NarrowBandSize=1;
    
    [y, x] = ind2sub(size(phi),idx); % points in narrow band
    
    % for m=1:length(x)
    %     DelId=find(x>=max((x(m)-1),1) & x<=min((x(m)+1),size(im,1))& y>=max((y(m)-1),1) & y<=min((y(m)+1),size(im,2)));
    %     x(DelId(DelId~=m))=0;
    %     y(DelId(DelId~=m))=0;
    % end
    idnew=find(x~=0);
    xnew=x(idnew);
    ynew=y(idnew);
    idxnew=idx(idnew);
    
    clear x y idx
    kuku = 0;
    if (ii == 209)
        kuku = 1;
    end

    x=xnew;
    y=ynew;
    idx=idxnew;
    
    %% Calculate the adaptive window size - parallel computing so you need matlabpool function but you should have it
    
    % myRads is an array for window size per pixel in contour
    clear MinDistIdx dist rad Neigborhood TempVec StdSize convergence RadSize SmallArea ang AngClass
    
    if (ii==1)
        rad(1:length(idx))=5;
    else
        for i=1:length(x)
            mone=1;
            FstIter=1;
            flag=0;
            if FstIter==1
                HalfNeighbor=3;
            end
            while ~flag
                flag=0;
                PrevForce=force;
                convergence=PrevForce;
                NormConvergence=(abs(min(convergence))+convergence)+0.05;
                NormConvergence= NormConvergence/max(NormConvergence);
                [pixelCount grayLevels] = hist(NormConvergence);
                % Calculate the mean convergence
                meanConv = sum(pixelCount .* grayLevels) / sum(pixelCount);
                
                for j=1:length(x)
                    dist(j)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
                end
                SortDist=sort(dist);
                if (length(SortDist)>=5)
                    RelInd{i}=find(dist<=SortDist(5));
                else
                    RelInd{i}=find(dist<=SortDist(length(SortDist)));
                end
                
                SmallArea=(im(max(y(i)-HalfNeighbor,1):min(y(i)+HalfNeighbor,dimy),max(x(i)-HalfNeighbor,1):min(x(i)+HalfNeighbor,dimx)));
                NewGrayLevels=linspace(min(min(im)),max(max(im)),10);
                
                GLCM = graycomatrix(SmallArea,'offset',[0 PxDist; -PxDist PxDist; -PxDist 0; -PxDist -PxDist],'GrayLimits',[min(im(:)) max(im(:))], 'Symmetric', true);
                stats = graycoprops(GLCM,{'contrast'});
                StdSize=stats.Contrast;
                
                [pixelCount grayLevels] = hist(abs(StdSize));
                % Calculate the mean gray level
                LocalHomo(i) = sum(pixelCount .* grayLevels) / sum(pixelCount);
                RadSize=ObjectSize/(log10(ObjectSize/2)*(2^((LocalHomo(i)/WholeCont)+(sqrt(1/meanConv)))));
                rad(i)=max(2,round(RadSize));
                
                if abs(rad(i)-HalfNeighbor)<=1
                    flag=1;
                else
                    HalfNeighbor=rad(i);
                    flag=0;
                end
                mone=mone+1;
                if (mone==5)
                    flag=1;
                end
                FstIter=2;
            end
            if ii==2
                Data.Rad=rad;
            end
        end
        for i=1:length(x)
            rad(i)=round(mean(rad(RelInd{i})));
        end
    end
    xneg= x-rad'; xpos= x+rad';      %get subscripts for local regions
    yneg= y-rad'; ypos= y+rad';
    
    %%
    
    P = phi; %sub phi
    
    %% Compute the Level Set parameters (according to probabilities) every 5th iteration
    
%     UNCCOMMMMMMEEEEENNNNNNTTTTT

    if (ii==1 || (mod(ii,1)==0))
%         
%         %  Get the probababilites from Python for the trained CNN model for each patch
%         
%         % For each point on this narrow band marding around the contour, create a patch and then classify
%         % (y,x) is the center of the patch to - internal, external, boundary. USE THIS FOR CREATING PATCHES
         % [y x] = ind2sub(size(phi),idx);
%         
%         %  Divide into patches and then find probabilities of each patch
%         % Get CNN predictions before this
%         %       % predictOrig should be a R3 vector, with 3 probabilities
%         % Do 2 things: Create Patches AND get classification probabilities
        patch_radius = 8;
        py_x = py.numpy.array(x(:).');
        py_y = py.numpy.array(y(:).');
        py_im = py.numpy.array(im(:).');

        % py_x_gnd_truth = py.numpy.array(x_gnd_truth(:).');
        % py_y_gnd_truth = py.numpy.array(y_gnd_truth(:).');

        [dim_x_im,dim_y_im] = size(im);
        return_val = py.matlab_python_api.run_iteration(py_x, py_y, py_im,dim_x_im,dim_y_im,patch_radius);

        predictOrig = return_val{1};
        x_dim_pred = return_val{2};
        y_dim_pred = return_val{3};
        predictOrig = double(py.array.array('d',py.numpy.nditer(predictOrig)));
        predictOrig = reshape(predictOrig',[y_dim_pred x_dim_pred]);
        predictOrig = predictOrig';
           % lambda1 = 1;
           % lambda2 = 2;
        lambda1 = [];
        lambda2 = [];
        for k = 1:length(x)
            tempPred = predictOrig(k,:);
            lambda1(k) = exp((1+tempPred(2)+ tempPred(3))/(1+tempPred(2)+ tempPred(1)));
            lambda2(k) = exp((1+tempPred(2)+ tempPred(1))/(1+tempPred(2)+ tempPred(3)));
        end

        % 1 - Probability inside
        % 2 - Probability boundaries
        % 3 - Normal tissue

        % lambda1 = exp((1+predictOrig(2)+ predictOrig(3))/(1+predictOrig(2)+ predictOrig(1)));
        % lambda2 = exp((1+predictOrig(2)+ predictOrig(1))/(1+predictOrig(2)+ predictOrig(3)));
    end
    

    
    %% Evolve the contour based on the new cost function - for now don't change it
    
    %-- re-initialize u,v,Ain,Aout
    u=zeros(size(idx)); v=zeros(size(idx));
    Ain=zeros(size(idx)); Aout=zeros(size(idx));
    
    Data1{ii} = lambda1;
    Data2{ii} = lambda2;
    
    %-- compute local stats
    for i = 1:numel(idx)  % for every point in the narrow band
        img = im(yneg(i):ypos(i),xneg(i):xpos(i)); %sub image
        P = phi(yneg(i):ypos(i),xneg(i):xpos(i)); %sub phi
        upts = find(P<=0);            %local interior
        Ain(i) = length(upts);
        if (Ain(i)==0)
            Ain(i)=1;
        end
        u(i) = sum(img(upts))/Ain(i);
        
        vpts = find(P>0);             %local exterior
        Aout(i) = length(vpts);
        if (Aout(i)==0)
            Aout(i)=1;
        end
        v(i) = sum(img(vpts))/Aout(i);
    end
    
    [curv,phi_x2,phi_y2] = get_curvature(phi,idx,x,y);
    %     curv=(fx2.*fyy + fy2.*fxx -2.*fx.*fy.*fxy)./den;
    grad=phi_x2+phi_y2;
    grad_m=grad.^0.5;
    kappa=curv.*grad_m;
    %     kappa=kappa./(max(abs(kappa(:)))+realmin);
    
    % NEEDED FOR CONVEX OPTIMIZATION
    % u and v are the means inside
    % Each index is a 10x10 patch (POINT == PATCH)
    % Element wise squaring and summation. For u, we can use the entire mask for the internal and external averages
    % instead of dividing into masks and averging.
    force = -nu+lambda1'.*((im(idx)-u)).^2-lambda2'.*((im(idx)-v)).^2; %Yezzi
    %       force=  -nu+lambda1*(LocIntFinal-u).^2-lambda2*(LocIntFinal-v).^2;
    %       clear LocIntFinal
    % normalize the force
    if normalize_force_flag
        force = force./max(abs(force(:)));
        dphidt=force+ mu* kappa;
        dt = .45/(max(abs(dphidt(:)))+eps);
    else
        dphidt=force+ mu* kappa;
    end
    ii=ii+1;
    PrevPhi=phi;
    phi(idx)=phi(idx)+dt*dphidt;
    phi = sussman(phi, .5);
    %
    %         BW=zeros(size(phi,1),size(phi,2));
    %         BW(phi<0)=1;
    %         BWLab=bwlabel(im2bw(BW));
    %         Area=cell2mat(struct2cell(regionprops(BWLab,'Area')));
    %         if (~isempty(Area))
    %             BW(BWLab==find(Area==max(Area)))=10;
    %             phi(BW<10)=abs(phi(BW<10));
    %         end
    ParamsVec{ii,1}=lambda1;
    ParamsVec{ii,2}=lambda2;
    ParamsVec{ii,3}=mu;
end

if ii>2
    Rad=ParamsVec;
else
    Rad=ParamsVec;
    FeatDist=[];
end

%% internal functions

function D = sussman(D, dt)
% forward/backward differences
a = D - shiftR(D); % backward
b = shiftL(D) - D; % forward
c = D - shiftD(D); % backward
d = shiftU(D) - D; % forward

a_p = a;  a_n = a; % a+ and a-
b_p = b;  b_n = b;
c_p = c;  c_n = c;
d_p = d;  d_n = d;

a_p(a < 0) = 0;
a_n(a > 0) = 0;
b_p(b < 0) = 0;
b_n(b > 0) = 0;
c_p(c < 0) = 0;
c_n(c > 0) = 0;
d_p(d < 0) = 0;
d_n(d > 0) = 0;

dD = zeros(size(D));
D_neg_ind = find(D < 0);
D_pos_ind = find(D > 0);
dD(D_pos_ind) = sqrt(max(a_p(D_pos_ind).^2, b_n(D_pos_ind).^2) ...
    + max(c_p(D_pos_ind).^2, d_n(D_pos_ind).^2)) - 1;
dD(D_neg_ind) = sqrt(max(a_n(D_neg_ind).^2, b_p(D_neg_ind).^2) ...
    + max(c_n(D_neg_ind).^2, d_p(D_neg_ind).^2)) - 1;

D = D - dt .* sussman_sign(D) .* dD;

%-- whole matrix derivatives
function shift = shiftD(M)
shift = [ M(1,:,:); M(1:size(M,1)-1,:,:) ];

function shift = shiftL(M)
shift = [ M(:,2:size(M,2),:) M(:,size(M,2),:) ];

function shift = shiftR(M)
shift = [ M(:,1,:) M(:,1:size(M,2)-1,:) ];

function shift = shiftU(M)
shift = [ M(2:size(M,1),:,:); M(size(M,1),:,:) ];

function S = sussman_sign(D)
S = D ./ sqrt(D.^2 + 1);

% Calculate curvature
function [curvature,phi_x2,phi_y2] = get_curvature(phi,idx,x,y);
[dimy, dimx] = size(phi);

%-- get subscripts of neighbors
ym1 = y-1; xm1 = x-1; yp1 = y+1; xp1 = x+1;

%-- bounds checking
ym1(ym1<1) = 1; xm1(xm1<1) = 1;
yp1(yp1>dimy)=dimy; xp1(xp1>dimx) = dimx;

%-- get indexes for 8 neighbors
idup = sub2ind(size(phi),yp1,x);
iddn = sub2ind(size(phi),ym1,x);
idlt = sub2ind(size(phi),y,xm1);
idrt = sub2ind(size(phi),y,xp1);
idul = sub2ind(size(phi),yp1,xm1);
idur = sub2ind(size(phi),yp1,xp1);
iddl = sub2ind(size(phi),ym1,xm1);
iddr = sub2ind(size(phi),ym1,xp1);

%-- get central derivatives of SDF at x,y
phi_x  = -phi(idlt)+phi(idrt);
phi_y  = -phi(iddn)+phi(idup);
phi_xx = phi(idlt)-2*phi(idx)+phi(idrt);
phi_yy = phi(iddn)-2*phi(idx)+phi(idup);
phi_xy = -0.25*phi(iddl)-0.25*phi(idur)...
    +0.25*phi(iddr)+0.25*phi(idul);
phi_x2 = phi_x.^2;
phi_y2 = phi_y.^2;

%-- compute curvature (Kappa)
curvature = ((phi_x2.*phi_yy + phi_y2.*phi_xx - 2*phi_x.*phi_y.*phi_xy)./...
    (phi_x2 + phi_y2 +eps).^(3/2)).*(phi_x2 + phi_y2).^(1/2);


