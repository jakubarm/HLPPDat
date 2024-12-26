function [net] = trainLogReg(Xtrain,Ttrain,regularisation)
% A standard logistic regression model with Classes-1 regressions
% and l2 regularisation of its weight

Classes = max(Ttrain);
TrainTargets = zeros(Classes,length(Ttrain));
i=4*(0:length(Ttrain)-1)';
i=i+Ttrain;
TrainTargets(i)=1;
net=feedforwardnet(Classes-1);
net.layers{2}.transferFcn='softmax';
net=configure(net,Xtrain',TrainTargets);
net.b{2}(:)=0;
net.b{2}(end)=1; % In the log. reg. model, the last class has probability
% 1 divided by sum of (previous classes + 1). This is achieved if we always
% send 1 into the last of output neuron. The 4 output neurons then enter a
% softmax function. The result will tally with (4.18) in "The elements of
% statistical learning" by T. Hastie.
net.biases{2}.learn=false;
net.layerWeights{2}.learn=false;
net.LW{2}=[eye(Classes-1); zeros(1,Classes-1)];
net.IW{1}(:)=0;
net.b{1}(:)=0;
net.performFcn='crossentropy';
net.trainFcn='trainscg';
net.outputs{2}.range(:,1)=-1; % Set default range
net.divideMode = 'none';
net.trainParam.min_grad=1e-9;
net.performParam.regularization=regularisation;
net.trainParam.epochs = 2000;

net.layers{1}.transferFcn='purelin';
net.trainParam.showWindow=1;
net=train(net,Xtrain',TrainTargets ...
    ,'usegpu','yes' ...
    );
end