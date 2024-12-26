function [mdl] = trainDKSVD(X,T,NumAtoms,gamma)
%TRAINDKSVD trains a discriminative dictionary for classification using
%sparse representation.
%   X - input signals - column vectors stacked into a matrix
%   T - targets, column vector containing integer labels starting from 1
%   AtomsPerClass - number oa atoms for each class

% Training parameters
mdl.algorithm="MOD";
% mdl.algorithm="KSVD";
mdl.gamma=gamma;
mdl.lam=5e-4;

% Determine the number of learning iterations
if NumAtoms == 1 % Depending on the number of atoms
    Iter = 5; % Sufficient for achieving the convergence
else
    Iter = NumAtoms*50 + 5;
end

X=X.';
C=max(T);
% Create matrix containing features and targets
XT=[X; zeros(C,length(T))];
[K,N]=size(XT);
XT((0:N-1)'*K+size(X,1)+T)=sqrt(mdl.gamma); % Set targets to 1

% Random initialization
DW=randn(K,NumAtoms);
DW(1:size(X,1),1)=1; % Add one constant atom
DW=DW./vecnorm(DW);

% SVD initialization
% [DW,~,~]=svd(XT,"econ");
% DW(:,NumAtoms+1:end)=[];

diffNorm=zeros(Iter,1);
averageAtomUsage=zeros(Iter,1);
MSE=zeros(N,1);
E=zeros(Iter,1);
for i=1:Iter
    A=zeros(NumAtoms,N);
    parfor n=1:N % Compute sparse representation coefficients
        try [A(:,n),info]=lasso(DW,XT(:,n),'lambda',mdl.lam,...
                ...'DFmax',15,'NumLambda',1,...
                'Standardize',false,'Intercept',false,'CacheSize',33e3);
        catch msg
            disp(msg)
        end
        MSE(n)=info.MSE;
    end
    E(i)=norm(MSE);
    A0norm=sum(A~=0,2); % l0 norm of A's rows
    rm=A0norm==0;       % Indices of atoms which will be removed
    A(rm,:)=[];         % Remove useless atoms
    averageAtomUsage(i)=sum(A0norm)/N;

    % !!! For debugging only !!!
    if ~rem(i,1)
        clf
        subplot(2,3,2)
        plot(DW)
        subplot(2,3,1)
        plot(A')
        subplot(2,3,4)
        plot((DW(end-C+1:end,~rm)*A)')
        title("Iter. " + i)
        subplot(2,3,5)
        plot(sum(A~=0))
        ylim([0 NumAtoms])
        subplot(2,3,3)
        plot(1:i,averageAtomUsage(1:i))
        subplot(2,3,6)
        semilogy(1:i-1,[E(1:i-1) diffNorm(1:i-1)])
        drawnow
    end
    % !!! End of For debugging only !!!

    % Update the dictionary
    switch mdl.algorithm
        case "MOD"
            DWnew=lsqminnorm(A',XT')';
            DWnew=DWnew./vecnorm(DWnew);
        case "KSVD"
            DWnew=DW(:,~rm); % Keep only the atoms that are being used
            for n=1:height(A)
                k=(1:height(A))~=n;
                atomUsed = A(n,:) ~= 0; % usage of the nth atom
                En=XT(:,atomUsed) - DWnew(:,k)*A(k,atomUsed);
                [U,S,V]=svd(En,"econ");
                DWnew(:,n)=U(:,1);
                A(n,atomUsed)=S(1)*V(:,1)';
            end
    end
    if height(A) < NumAtoms % If unused atoms were removed, add atoms.
        [~,s]=sort(MSE,"descend"); % Sort signals by their last approximation error
        ind=s(1:NumAtoms-width(DWnew));
        disp(length(ind) + " atoms were reset.")
        newAtoms=X(:,ind) - DWnew(1:height(X),:)*A(:,ind);
        newAtoms=newAtoms./vecnorm(newAtoms);
        DWnew(1:height(X),width(DWnew)+1:NumAtoms)=newAtoms;
    end
    diffNorm(i)=norm(DW-DWnew,"fro");
    DW=DWnew;
    if diffNorm(i) < 1e-5
        break
    end
end
% Renormalise the dictionary and weights
DW=DW./vecnorm(DW(1:size(X,1),:));
mdl.D=DW(1:size(X,1),:);
mdl.W=DW(size(X,1)+1:end,:);
end