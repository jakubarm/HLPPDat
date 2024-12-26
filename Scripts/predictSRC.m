function [p] = predictSRC(Xtrain,Ttrain,Xtest,Ttest,regularisation,sparseMethod,solve_occlusions,bedSize)
disable_plots = true;
[Ttrain,i] = sort(Ttrain); % Sorted trainig-group targets
D = Xtrain(i,:)'; % Sorted dictionary
if solve_occlusions % If occlusion cancelation is allowed
    % D = [D eye(size(D,1))]; % For dealing with occlusions
    for n=1:bedSize(1)
        D(:,end+1) = ...
            repmat([zeros(n-1,1); 1; zeros(bedSize(1)-n,1)],bedSize(2),1);
    end
end
D=D./vecnorm(D); % Normalise the dictionary atoms
N=height(Xtest);  % Number of tests
p=zeros(N,1);    % Predictions on test data
m=1:size(D,2);   % Indexing vector
for n=1:N
    y=Xtest(n,:)';  % Test observation
    B=lasso(D,y,... % l1 basis pursuit
        'lambda',regularisation,'Standardize',false,'Intercept',false);
    if ~disable_plots
        clf
        subplot(3,1,1)
        plot([y D*B])
    end
    C=max(Ttrain);
    CR  = zeros(C,1); % Class residue
    SCC = zeros(C,1); % Sum of class coefficients
    MC  = zeros(C,1); % Maximum coefficient for each class
    for c=1:C % Compute prediction error for each class
        % Trainig observations for class c
        if solve_occlusions
            i=[Ttrain==c; false(size(D,2)-length(Ttrain),1)];
        else
            i=Ttrain==c;
        end
        CR(c)  = norm(y-D(:,i)*B(i)); % Class residue
        SCC(c) = sum(B(i)); % Sum of class coefficients
        MC(c)  = max(B(i)); % Maximum coefficient for each class


        if c==Ttest(n) && ~disable_plots
            subplot(3,1,2)
            stem(m(~i),B(~i))
            hold on
            stem(m(i),B(i))
        end
    end
    if ~disable_plots
        subplot(3,1,3)
        bar(CR)
        drawnow
    end
    switch sparseMethod
        case 'MCR'
            [~,c]=min(CR);
        case 'MC'
            [~,c]=max(MC);
        case 'MSCC'
            [~,c]=max(SCC);
    end
    p(n)=c;
end
end