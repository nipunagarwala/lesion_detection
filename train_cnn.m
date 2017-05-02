function [net, imdb] = train_cnn(tImg, tLabel, dim)
% train_cnn(timg, tlab, dim) trains a cnn on an image database.

%   train_cnn trains a convolutional neural network using backpropagation
%   and stochastic gradient descent with momentum. imdb has two attributes:
%   the images themselves, which are arranged in a 4D array (W X H X D X
%   N), and the labels, which are a vector containing the class labels.

% Shuffle training data
randShuff = randperm(size(tLabel, 2));
tImg = tImg(:, randShuff);
tLabel = tLabel(:, randShuff);

% Defining divide of training, validation, and testing sets (1, 2, 3)
set = ones(1, size(tLabel, 2));

% Take 20% of the training set as the validation set
validSet = randperm(numel(tLabel), round(numel(tLabel) / 5));
set(validSet) = 2;

% Mean centering of data
data = single(reshape(tImg, dim, dim, 1, []));

dMean = mean(data(:, :, :, set == 1), 4);
data = bsxfun(@minus, data, dMean);

imdb.images.data = data;
imdb.images.data_mean = dMean;
imdb.images.labels = tLabel;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:3,'uniformoutput',false) ;

net = CNNArch(dim, 5, 2, 20, 100, 300);
%net = CNNArchNew(dim);

net = cnn_train(net, imdb, @getBatch, 'batchSize', 128, 'numEpochs', 10, ...
'gpus', [], 'learningRate', 0.005, 'cudnn', false);

net.layers{end} = struct('type', 'softmax');

    % getBatch returns a minibatch of the training set for SGD
    function [im, labels] = getBatch(imdb, batch)
        im = imdb.images.data(:, :, :, batch);
        labels = imdb.images.labels(batch);
    end


end

