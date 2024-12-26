function [prediction,net,info] = trainTestTransferNet(Xtrain,Ttrain,Xtest,Ttest,bedSize,regul,netType)

% Adjust training options according to the topolgy:
switch netType
    case 'openpose4'
        algorithm='sgdm';
        momentumName='momentum';
        if height(Xtrain) > 10000 % If the datased was augmented artificially
            maxEpochs=1;
        else
            maxEpochs=61;
        end
        gradientThreshold=Inf;
        LR=5e-8;
        miniBatch=1;
        valFreq=height(Xtrain);
        momentum=(1/2)^(1/height(Xtrain));
    case 'squeezenet4'
        algorithm='sgdm';
        momentumName='momentum';
        momentum=0.9;
        gradientThreshold=Inf;
        LR=2e-5;
        iterations=4e3; %  maxEpochs=ceil(iterations/numBatches)
        if height(Xtrain) > 10000 % If the datased was augmented artificially
            numBatches=4*61;
        else
            numBatches=4;
        end
        miniBatch=height(Xtrain)/numBatches;
        maxEpochs=ceil(iterations/numBatches);
        valFreq=numBatches;
    case 'googlenet4'
        algorithm='sgdm';
        momentumName='momentum';
        momentum=0.9;
        gradientThreshold=Inf;
        LR=2e-4;
        iterations=4e3; % Will be divided by the number of min batches
        if height(Xtrain) > 10000 % If the datased was augmented artificially
            numBatches=5*61;
        else
            numBatches=5;
        end
        miniBatch=floor(height(Xtrain)/numBatches);
        maxEpochs=ceil(iterations/numBatches);
        valFreq=numBatches;
    case 'resnet4'
        algorithm='sgdm';
        momentumName='momentum';
        momentum=0.9;
        gradientThreshold=Inf;
        LR=1e-3;
        iterations=2400; % Will be divided by the number of min batches, equals 24 h of training
        if height(Xtrain) > 10000 % If the datased was augmented artificially
            numBatches=3*61;
        else
            numBatches=3;
        end
        miniBatch=floor(height(Xtrain)/numBatches);
        maxEpochs=ceil(iterations/numBatches);
        valFreq=numBatches;
end
% maxEpochs = 1;
map='gray'; % Apply a colour transformation and reshape the data

% Training datastore:
da = imageDataAugmenter("RandXTranslation",[-9 9], ... [min max] in px
    "RandYTranslation",[-48 48],... % [min max] in px
    "RandRotation",[-10 10]); % [min max] in px
useRAMonly = strcmpi(netType,'openpose4');
if useRAMonly
    X4D = feature2im(Xtrain,map,bedSize,netType);
    sz=size(X4D);
    ds = augmentedImageDatastore(sz(1:2),X4D,categorical(Ttrain),"DataAugmentation",da);
else
    sz = saveImages(Xtrain,map,bedSize,netType,Ttrain);
    ds = imageDatastore("temporary","IncludeSubfolders",true,"LabelSource","foldernames");
    ds = augmentedImageDatastore(sz(1:2),ds);
end

X4Dval = feature2im(Xtest,map,bedSize,netType);

options = trainingOptions( ...
    algorithm ...
    ,'MaxEpochs',maxEpochs ...
    ,'GradientThresholdMethod','global-l2norm' ...
    ,'GradientThreshold',gradientThreshold ...
    ,'InitialLearnRate',LR ...
    ,momentumName,momentum ...
    ,'L2Regularization',regul ...
    ,'MiniBatchSize',miniBatch ...
    ,'ExecutionEnvironment','gpu' ...
    ,'verbose',false ...
    ,'ValidationData',{X4Dval,categorical(Ttest)} ...
    ,'ValidationFrequency',valFreq ...
    ...,'Plots','training-progress'...
    );
if length(unique(Ttest)) < 4 % If there is only one class
    options.ValidationData=[];
end

% Predict using pretrained OpenPose network:
% load("origOpenPose.mat")
% netInput=dlarray(X4D(:,:,:,1),"SSC");
% [heatmaps,pafs] = predict(net,netInput);
% heatmaps = extractdata(heatmaps);
% montage(rescale(heatmaps),"BackgroundColor","b","BorderSize",3)


% Train and test the network
load(netType); % Set up the network topology
switch class(lgraph_1)
    case "dlnetwork"
        [net,info] = trainnet(ds,lgraph_1,"crossentropy",options);
    otherwise
        [net,info] = trainNetwork(ds,lgraph_1,options);
end
predictedProbability = double(predict(net,X4Dval));
[~,prediction] = max(predictedProbability,[],2);
end

function X4D = feature2im(X,map,bedSize,netType)
% Colour transformation:
bitDepth=2^10;
eval(['map=' map '(bitDepth);'])
N=height(X);
X=1-(1-X)/(1-min(X,[],'all'));

% Resizing:
interimSize=bedSize(1:2)*4;
switch netType
    case 'openpose4'
        finalSize=[256 456 3];
    case 'squeezenet4'
        finalSize=[227 227 3];
    case 'googlenet4'
        finalSize=[224 224 3];
    case 'resnet4'
        finalSize=[224	224	3];
end
i=floor((finalSize(1:2)-interimSize)/2);
X4D=zeros([finalSize N],'single');
for n=1:N
    x = zeros([finalSize(1:2)],'single'); % n-th image
    x(i(1)+(1:interimSize(1)),i(2)+(1:interimSize(2))) = ...
        imresize(reshape(X(n,:),bedSize(1),bedSize(2)),interimSize);
    x = ind2rgb(round(bitDepth*x),map);
    switch netType
        case 'openpose4'
            X4D(:,:,:,n) = x(:,:,[3 2 1]) - 1/2; % OpenPose has channels in
            % reverse order and centered around zero (hence -1/2).
        case {'squeezenet4','googlenet4','resnet4'}
            X4D(:,:,:,n) = x;
    end
end
end

% Not used anymore:
% A function for storing the imges into hard drive to save RAM
function [sz]=saveImages(X,map,bedSize,netType,T)
if exist("temporary","dir")
    rmdir("temporary","s")
end
for n=1:size(X,1)
    filePath="temporary\" + T(n);
    if ~exist(filePath,"dir")
        mkdir(filePath)
    end
    X4D=feature2im(X(n,:),map,bedSize,netType);
    imwrite(X4D,filePath + "\" + n + ".png")
end
sz=[size(X4D) size(X,3)];
end