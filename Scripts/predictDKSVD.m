function [y] = predictDKSVD(mdl,X)
%PREDICTDKSVD trains a discriminative dictionary for classification using
%sparse representation.
%   mdl - model containing a redundant dictionary and a weight matrix
%   X - input signals - column vectors stacked into a matrix

X=X';
N=size(X,2);
for n=1:N % Compute sparse representation coefficients
    A(:,n)=lasso(mdl.D,X(:,n),'lambda',mdl.lam,...
        'Standardize',false,'Intercept',false);
    
%     plot([mdl.D*A(:,n) X(:,n)])
%     drawnow
end
Y=mdl.W*A;
[~,y]=max(Y);
end