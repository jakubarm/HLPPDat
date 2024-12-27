function [HOG] = extractHOG(X,bedSize,cellSize,blockSize)
if nargin < 3
    cellSize = 2;
end
if nargin < 4
    blockSize = 3;
end

% in bed shape
X = reshape(X',bedSize(1),bedSize(2),size(X,1));

% Add a column of ones
X(:,end+1,:)=1;
bedSize(2)=bedSize(2)+1;

% The function extracts HOG features
cellSize = cellSize*[1 1];
blockSize = blockSize*[1 1];
blockOverlap = ceil(blockSize/2);
blockOverlap(blockOverlap == blockSize)=blockSize(blockOverlap == blockSize)-1;
numBins = 9;

blocksPerImage = floor((bedSize(1:2)./cellSize - blockSize(1:2))./(blockSize - blockOverlap) + 1);
N = prod([blocksPerImage, blockSize, numBins]);
HOG = zeros(size(X,3),N);
for i = 1:size(X,3)
    HOG(i,:) = extractHOGFeatures(X(:,:,i),'CellSize',cellSize,'BlockSize',blockSize,'BlockOverlap',blockOverlap,'NumBins',numBins);
end
end