clf
clear all

load('newBedData20.mat')

T = Class';
bedSize=size(Bed);
K=max(subject);   % K folds

% ========================= Choose one =========================
% method = 'baggedTree';
% method = 'CART';
% method = 'LDA';
% method = 'QDA';
% method = 'SVM';
% method = 'LogReg';
% method = 'sparse'; sparseMethod = 'MSCC';
% method = 'sparse'; sparseMethod = 'MC';
% method = 'sparse'; sparseMethod = 'MCR';
% method = 'KNN';
% method = 'DKSVD';
% method = 'FcmSvdPn';
% method = 'HoG+SVM';
% method = 'HoG+PCA+SVM';
% method = 'HoG+LDA';
 method = 'HoG+KNN';
% method = 'CNN';
% method = 'LogReg';
% method = 'squeezenet4';
% method = 'openpose4';
% method = 'googlenet4';
% method = 'alexnet4';
% method = 'resnet4';
% ========================== Options ===========================
test_occlusions = true; % Test performance on occluded data. Let always true because it includes test on clean data.
occlusion_type = 0; % Type of tested occlusion
% ==============================================================

% Create the sets with all possible combinations of occluded data
BedOcc1 = repmat(Bed,1,1,bedSize(1));
BedOcc0 = BedOcc1;
for k=1:bedSize(1)
    BedOcc0(k,:,(1:bedSize(3))+(k-1)*bedSize(3))=0; % Beds occluded by zeroes
    BedOcc1(k,:,(1:bedSize(3))+(k-1)*bedSize(3))=1; % Beds occluded by ones
end
TOcc = repmat(T,bedSize(1),1); % Responses corresponding either to beds occluded
% by ones or zeroes
if test_occlusions
    P=zeros(size(TOcc)); % Crossvalidated prediction for all combinations
else
    P=zeros(size(T)); % Crossvalidated prediction for data without occl.
end

% Reshape data
X =    reshape(Bed(:),   bedSize(1)*bedSize(2),bedSize(3))';
XOcc0 = reshape(BedOcc0(:),bedSize(1)*bedSize(2),bedSize(3)*bedSize(1))';
XOcc1 = reshape(BedOcc1(:),bedSize(1)*bedSize(2),bedSize(3)*bedSize(1))';

% Hyperparameters of each method
switch method
    case 'baggedTree'
        regul = 1:50;
    case 'CNN'
        % regul = [0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005];
        regul = 0.001;
    case 'FcmSvdPn'
        select = 'nClast';         %(8) number of FCM clasters
        regul = 8;

        %         select = 'nTruncSVD';     %(23) SVD dimention reduction
        %         regul = 1:1:5;
        %         regul = 23

        %         select = 'nHydNeur';      %(20) number of hydden neuron in paternnet
        %         regul = 18:22;

        %select = 'nLearnEpoch';   %(11) nubmer of learning epoch (stoping param)
        %regul = 1:20;
        %regul = 10:12;

        %          select = 'bestParam';
        %          regul = 0;

    case 'HoG+SVM'
        % regul = [0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.7 1 1.4 2 5 10 20 50 100 200 500 1000];
        regul = 1;
    case 'HoG+PCA+SVM'
        % regul = [0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.7 1 1.4 2 5 10 20 50 100 200 500 1000];
        regul = 1;
    case 'HoG+LDA'
        regul = [0.2 0.3 0.5 0.7 0.8 0.9 0.95 1];
    case 'HoG+KNN'
        regul = 1:100;
    case 'CART'
        regul = NaN; % Only one tree, hence no regularisation.
    case 'DKSVD'
        regul = 10; % Number of atoms.
    case 'KNN'
        % regul = 1:50; % Optimum value
        regul = 3;
    case 'LDA'
        % regul = linspace(0.1,1-eps,51);
        regul = 0.55;
    case 'LogReg'
        % regul = logspace(-9,log10(1-1e-16),19);
        regul = 0.003;
    case 'QDA'
        regul = 0;
    case 'sparse'
        % regul = [0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001];
        regul=0.003;
    case 'SVM'
        % regul = logspace(-3,3,30);
        regul = 0.9;
    case {'squeezenet4','openpose4','googlenet4','alexnet4','resnet4'}
        % regul = [0.0001 0.001 0.01 0.1];
        regul=0.1;
end

% HoG feature extraction if the method needs it
if strcmpi(method(1:3),'HoG')
    cellSize = 2;
    blockSize = 2;
    X=extractHOG(X,bedSize,cellSize,blockSize);
    XOcc0=extractHOG(XOcc0,bedSize,cellSize,blockSize);
    XOcc1=extractHOG(XOcc1,bedSize,cellSize,blockSize);
end

for solve_occlusions = [false true] % Build a classifier immune to occlusions
    for r = 1:length(regul)
        rng('default')
        tic
        for k=1:K % Crossvalidation across the 'patients'
            % Prepare the trainig sets according to user-defined parameters
            if solve_occlusions && ...
                    ~sum(strcmpi(method, {'sparse','DKSVD'}))
                % Train over a large dataset containing all possible occlusions to
                % make the comparision of these methods and SRC method fair. SRC
                % does not need this extension of the training datased because the
                % occlusions are included in the dictionary.
                Xtrain=[X
                    XOcc0
                    XOcc1];
                Ttrain=[T
                    TOcc
                    TOcc];
                trainInd = repmat(subject',length(Ttrain)/length(T),1) ~= k;
                Xtrain(~trainInd,:)=[];
                Ttrain(~trainInd)=[];
            else % Train over normal data
                trainInd = subject' ~= k; % Training set
                Xtrain=X(trainInd,:);
                Ttrain=T(trainInd);
            end

            % Choose the test set according to the user-defined parameters
            if test_occlusions
                if occlusion_type == 1
                    Xtest = [X; XOcc1];
                elseif occlusion_type == 0
                    Xtest = [X; XOcc0];
                else
                    error('This type of occlusion is not supported by the script.')
                end
                Ttest = [T; TOcc];
                testInd = repmat(subject',length(Ttest)/length(T),1) == k;
                Xtest(~testInd,:)=[];
                Ttest(~testInd)=[];
            else
                testInd = subject' == k;
                Xtest = X(testInd,:);
                Ttest = T(testInd);
            end

            switch method
                case 'LDA'
                    mdl = fitcdiscr(Xtrain, Ttrain,...
                        'DiscrimType', 'linear', ...
                        'Gamma', regul(r), ...
                        'FillCoeffs', 'off');
                    P(testInd)=predict(mdl,Xtest);
                case 'QDA'
                    mdl = fitcdiscr(Xtrain, Ttrain, 'DiscrimType', 'quadratic', ...
                        'Gamma', regul(r), ...
                        'FillCoeffs', 'off');
                    P(testInd)=predict(mdl,Xtest);
                case {'SVM','HoG+SVM'}
                    template = templateSVM(...
                        'KernelFunction', 'linear', ...
                        'CacheSize',6*1024, ...
                        'KernelScale', 'auto', ...
                        'BoxConstraint', regul(r), ...
                        'Standardize', true);
                    mdl = fitcecoc(Xtrain, Ttrain, ...
                        'Learners', template, ...
                        'Coding', 'onevsone', ...
                        'ClassNames', [1; 2; 3; 4]);
                    P(testInd)=predict(mdl,Xtest);
                case {'KNN','HoG+KNN'}
                    mdl = fitcknn(Xtrain,Ttrain,'NumNeighbors',regul(r));
                    P(testInd)=predict(mdl,Xtest);
                case 'CART'
                    mdl = fitctree(Xtrain, Ttrain, ...
                        'SplitCriterion', 'gdi', ...
                        'MaxNumSplits', 100, ...
                        'Surrogate', 'off', ...
                        'ClassNames', [1; 2; 3; 4]);
                    P(testInd)=predict(mdl,Xtest);
                case 'baggedTree'
                    mdl = TreeBagger(regul(r),Xtrain,Ttrain);
                    p=predict(mdl,Xtest);
                    p=str2double(p);
                    P(testInd) = p;
                case 'sparse'
                    P(testInd) = predictSRC(Xtrain,Ttrain,Xtest,Ttest,regul(r),sparseMethod,solve_occlusions,bedSize(1:2));
                case 'DKSVD'
                    % Train a discriminative dictionary model
                    mdl=trainDKSVD(Xtrain,Ttrain,regul(r),0.001);

                    % Test the dictionary
                    P(testInd)=predictDKSVD(mdl,Xtest);
                case 'CNN'
                    % Test net's performance
                    P(testInd) = trainTestSimpleCNN(Xtrain,Ttrain,Xtest,Ttest,regul(r),bedSize(1:2));
                case 'LogReg'
                    mdl=trainLogReg(Xtrain,Ttrain,regul(r));
                    p = mdl(Xtest');
                    [~,P(testInd)]=max(p);
                case {'squeezenet4','openpose4','googlenet4','resnet4'}
                    [P(testInd),net,info{k}]=trainTestTransferNet(Xtrain,Ttrain,Xtest,Ttest,bedSize,regul(r),method);
                case 'alexnet4'
                    [P(testInd),net,info{k}]=trainTestTransferNetAlt(Xtrain,Ttrain,Xtest,Ttest,bedSize,regul(r),method);
                case {'FcmSvdPn'}
                    P(testInd) = trainTestFCMSVDPN(Xtrain,Ttrain,Xtest,regul(r),bedSize(1:2),select);
            end
        end
        t=toc;
        if test_occlusions
            Co = confusionmat(TOcc,P(length(T)+1:end));
            TPRo(r) = sum(diag(Co))/sum(sum(Co));
            C =  confusionmat(T,P(1:length(T)));
        else
            C = confusionmat(T,P);
        end
        TPR(r) = sum(diag(C))/sum(sum(C));

        % Display crossvalidation accuracy on clean data
        disp_online(['Classification method is: ' method ' with parameter ' num2str(regul(r))])
        disp_online(['Deal with occlusions = ' num2str(solve_occlusions) ...
            ', test occlusion = false, ' ...
            ' Occluded by ' num2str(occlusion_type)])
        disp_online(['TPR on test data: ' num2str(TPR(r),'%.4f') '.'])

        % Add crossvalidation accuracy on test data
        if test_occlusions
            disp_online(['Deal with occlusions = ' num2str(solve_occlusions) ...
                ', test occlusion = true ' ...
                ' Occluded by ' num2str(occlusion_type)])
            disp_online(['TPR on test data: ' num2str(TPRo(r),'%.4f') '.'])
        end
        disp_online(['Computational time ' num2str(t) ' seconds.'])
    end

    % Plot TPRs
    if test_occlusions
        loglog(regul,[TPR; TPRo]','-o')
        ylabel 'TPR, Occluded TPR'
    else
        loglog(regul,TPR','-o')
        ylabel 'TPR'
    end

    grid on
    xlabel 'Regularisation'

    hold on
    drawnow
end
disp(' ')