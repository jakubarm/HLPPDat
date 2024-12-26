function [p] = trainTestFCMSVDPN(Xtrain,Ttrain,Xtest,regularisation,bedSize,selector)
% Train model chain with Fuzzy C-mean, Singular Value Decomposition, 
% paternnet
% fully contected clasification
%   Detailed explanation goes here

nClastBest = 8; % Optimal param
truncBest = 23;
hyddenNeuronsBest = 20;
learningEpochBest = 11;

class = 4;% poloha na levo na pravo a na zádech na bříše

% parametry -> regularizace
switch selector
    case 'nClast'
        nClast = regularisation; %FCM clasters
        trunc = truncBest;%ořez SVD
        hyddenNeurons = hyddenNeuronsBest;
        learningEpoch = learningEpochBest;
    case 'nTruncSVD'
        nClast = nClastBest; %FCM clasters
        trunc = regularisation;%ořez SVD
        hyddenNeurons = hyddenNeuronsBest;
        learningEpoch = learningEpochBest;
    case 'nHydNeur'
        nClast = nClastBest; %FCM clasters
        trunc = truncBest;%ořez SVD
        hyddenNeurons = regularisation;
        learningEpoch = learningEpochBest;
    case 'nLearnEpoch'
        nClast = nClastBest; %FCM clasters
        trunc = truncBest;%ořez SVD
        hyddenNeurons = hyddenNeuronsBest;
        learningEpoch = regularisation;
    otherwise
        nClast = nClastBest; %FCM clasters
        trunc = truncBest;%ořez SVD
        hyddenNeurons = hyddenNeuronsBest;
        learningEpoch = learningEpochBest;
end

% in bed shape
BedTrain = reshape(Xtrain',bedSize(1),bedSize(2),size(Xtrain,1));
BedTest = reshape(Xtest',bedSize(1),bedSize(2),size(Xtest,1));

%% Fuzzy C-means (train data)

parfor i = 1:size(BedTrain,3)

   
    
    im = uint8(256-256*BedTrain(:,:,i));
    
    %filtr průměr
    %h = ones(2,2)/4;
    %im = imfilter(im,h,'replicate');
    
    %[C,F,LUT,H]=FastFCMeans(im,nClast,2.1);
    LUT = FastFCMeansEdit(im,nClast,2.1);
    L=LUT2label(im,LUT);

    % feature vector FCM
    Vr = reshape(L,1,[]);
    %Vc = reshape(L.',1,[]);
    Ar(:,i) = Vr;
    %Ac(:,i) = Vc;

end

%% SVD

Ar = double(Ar);
[U,S,V] = svds(Ar,trunc);
Vt = V.';

% figure
% [Uc,Sc,Vc] = svd(Ar);
% loglog(diag(Sc));
% title('(SVD) \Sigma values')
% ylabel('Eigen value of \Sigma')
% xlabel('Dimention')
% xlim([0 80])


%% NN train

% define target class
target = zeros(class,size(Ttrain,1)); % plus nulová třída
for i = 1:1:class
    target(i,Ttrain(:,1)==i) = 1;
end

net = patternnet(hyddenNeurons,'trainscg','crossentropy'); %počet neuronů ve skryté skryté vstvě
net.trainParam.showWindow = false;
net.divideParam.trainRatio = 1;
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 0;
net.trainParam.epochs = learningEpoch; %test epoch
                    
net = train(net,Vt,target,'useGPU','yes');

%% TEST

p_pred = zeros(class,size(BedTest,3));
Transform = U*diag(1./diag(S));
parfor i = 1:size(BedTest,3)

    imTest = uint8(256-256*BedTest(:,:,i));
    LUT = FastFCMeansEdit(imTest,nClast,2.1);
    L=LUT2label(imTest,LUT);

    % feature vector FCM
    V = reshape(L,1,[]);

    %přepočet SVD
    
    X=double(V)*Transform;
    p_pred(:,i) = net(X.');
    
end
[~,pT]=max(p_pred,[],1);
p=pT';


end