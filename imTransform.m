function [newImg] = imTransform(img, sigma, alpha,Interval)
% imTransform(img, nTrans, alpha) takes in an image and returns an image 
% that is a elastic/affine deformation of the original image.
%   
% imTransform generates a transformed image. A displacement
% matrix the same dimensions as the image is generated with all numbers
% between [-1, 1]. The matrix is blurred with a gaussian filter, and then
% the original image is displaced with the new image. Then, the image is
% rotated and translated a random amount.
%
% The Gaussian filters vary with respect to 'sigma', or the standard
% deviation. 

img = im2single(img);

% Create displacement matrices with scaling factor alpha
dplacex = -1 + 2 * rand(size(img, 1), size(img, 2));
dplacey = -1 + 2 * rand(size(img, 1), size(img, 2));

% apply gaussian filter to x and y matrices
dplacex = imgaussfilt(dplacex, sigma);
dplacey = imgaussfilt(dplacey, sigma);

%normalize displacement x and y matrices

dxmin = min(dplacex(:));
dxmax = max(dplacex(:));
dplacex = (dplacex - dxmin) ./ (dxmax - dxmin);

dymin = min(dplacey(:));
dymax = max(dplacey(:));
dplacey = (dplacey - dymin) ./ (dymax - dymin);


dplacex = dplacex * alpha;
dplacey = dplacey * alpha;
dplace = dplacex;
dplace(:, :, 2) = dplacey;

% elastic deformation
newImg = imwarp(img, dplace);

% rotate image
deg = randi([-30, 30]);
newImg = imrotate(newImg, deg, 'crop');

%Center image around nnz part
newImg = centerimg(newImg);

% scale non-zero part of image

% eroding
erode = [0.75 0.8 0.9];

% dilating
dilate = [1.1 1.2 1.3];

if rand >= 0.5
    dim = Interval*2; % for CT
    newImg = imresize(newImg, erode(randi(3)));
    [m, n] = size(newImg);
    newImg = padarray(newImg, [floor((dim-m)/2) floor((dim-n)/2)], 'replicate','post');
    newImg = padarray(newImg, [ceil((dim-m)/2) ceil((dim-n)/2)], 'replicate','pre');

else
    newImg = imresize(newImg, dilate(randi(3)));
    [m, n] = size(newImg);
    newImg = newImg(m/2 - (Interval-1): m/2 + Interval, n/2 - (Interval-1): n/2 + Interval);
end

% EDIT: Dilate and erode with larger scaling values
%scale = [1, 2, 3];
% scale = [1, 2, 3];
% 
% if rand >= 0.5   
%     newImg = imdilate(newImg, strel('square', scale(randi(3))));
%     
% else
%     newImg = imerode(newImg, strel('square', scale(randi(3))));
%     
% end

%translate image
transx = randi([-1, 1]);
transy = randi([-1, 1]);
 
newImg = imtranslate(newImg, [transx transy]);

%{
newImg = zeros(size(img, 1), size(img, 2));

for row = 1:size(img, 1)
    for col = 1:size(img, 2)
        low_ii = row + floor(dplacex(row, col));
        high_ii = row + ceil(dplacex(row, col));
        low_jj = col + floor(dplacey(row, col));
        high_jj = col + ceil(dplacey(row, col));
        
        if(low_ii <= 0 || low_jj <= 0 || high_ii >= size(img, 1) || high_jj >= size(img, 2))
            continue
        end
        
        res = img(low_ii, low_jj) / 4 + img(low_ii, high_jj) / 4 + img(high_ii, low_jj) / 4 + img(high_ii, high_jj) / 4;
        
        newImg(row, col) = res;
    end
end
%}

end

