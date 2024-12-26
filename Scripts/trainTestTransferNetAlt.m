function [prediction,net,info] = trainTestTransferNetAlt(Xtrain,Ttrain,Xtest,Ttest,bedSize,regul,netType)
    % Apply image reshape
    map='gray';
    X4D = feature2im(Xtrain,map,bedSize);
    X4Dval = feature2im(Xtest,map,bedSize);
    Y1D = categorical(Ttrain);
    Y1Dval = categorical(Ttest);

    % Apply image generators:
    inputSize = [227 227 3];
    pixelRange = [-3 3];
    scaleRange = [0.9 1.1];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',false, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange, ...
        'RandXScale',scaleRange, ...
        'RandYScale',scaleRange);
    generatorTrain = augmentedImageDatastore(inputSize(1:2), X4D, Y1D, ...
        'DataAugmentation', imageAugmenter);
    generatorTest = augmentedImageDatastore(inputSize(1:2), X4Dval, Y1Dval, ...
        'DataAugmentation', imageAugmenter);
    
    load(netType); % Set up the network topology
    
    % Adjust training options according to the topolgy:
    switch netType
        case 'alexnet4'
            iterations=4e2; % Will be divided by the number of min batches
            maxEpochs=6;
            miniBatchSize=floor(iterations/maxEpochs);
            algorithm='adam';
            %momentumName='momentum';
            %momentum=0.9;
            %gradientThreshold=Inf;
            LR=1e-4;
            valFreq=maxEpochs;
        case 'resnet4'
            numMainBatches=4;
            algorithm='adam';
            %momentumName='momentum';
            %momentum=0.9;
            %gradientThreshold=Inf;
            LR=1e-4;
            iterations=4e3; % Will be divided by the number of min batches
            if size(X4D,4) > 1000 % If the datased was augmented artificially
                miniBatchSize=floor(size(X4D,4)/(numMainBatches*61));
            else
                miniBatchSize=floor(size(X4D,4)/numMainBatches);
            end
            numBatches=floor(size(X4D,4)/miniBatchSize);
            maxEpochs=ceil(iterations/numBatches);
            valFreq=numBatches;
    end
    options = trainingOptions( ...
        algorithm ...
        ,'MaxEpochs',maxEpochs ...
        ...,'GradientThresholdMethod','global-l2norm' ...
        ...,'GradientThreshold',gradientThreshold ...
        ,'InitialLearnRate',LR ...
        ,'Shuffle','every-epoch' ...
        ...,momentumName,momentum ...
        ...,'L2Regularization',regul ...
        ,'MiniBatchSize',miniBatchSize ...
        ,'ExecutionEnvironment','gpu' ...
        ,'verbose',false ...
        ,'ValidationData',generatorTest ... % Fails due to missing output classes!
        ,'ValidationFrequency',valFreq ...                % Fails due to missing output classes!
        ...,'Plots','training-progress'...
        );
    if length(unique(Ttest)) < 4 % If there is only one class
        options.ValidationData=[];
    end
    
    % Training:
    [net,info] = trainNetwork(generatorTrain,layers_1,options);
    
    % Test net's performance:
    prediction = double(classify(net,generatorTest,'ExecutionEnvironment','cpu'));
end

function X4D = feature2im(X,map,bedSize)
    % Colour transformation:
    bitDepth=2^10;
    eval(['map=' map '(bitDepth);'])
    N=height(X);
    X=1-(1-X)/(1-min(X,[],'all'));
    
    newSize=[bedSize(1) bedSize(2)];
    X4D=zeros([newSize 3 N],'single');
    for n=1:N
        X4D(:,:,:,n) = ind2rgb(round(bitDepth*reshape(X(n,:),bedSize(1),bedSize(2))),map);
    end
end

function X4D = feature2im_old(X,map,bedSize)
    % Colour transformation:
    bitDepth=2^10;
    eval(['map=' map '(bitDepth);'])
    N=height(X);
    X=1-(1-X)/(1-min(X,[],'all'));
    
    newSize=[227 227];
    X4D=zeros([newSize 3 N],'single');
    for n=1:N
        X4D(:,:,:,n) = imresize(ind2rgb(round(bitDepth*reshape(X(n,:),bedSize(1),bedSize(2))),map),newSize);
    end
end
