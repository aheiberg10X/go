function V = Value1(CharBoard)

% compute the value V for a given 19x19 board configuration
% using binary features
% input BitBoard = 1x361 character vector (set={'w','b','e'})


%load precomputed weight vector
%load Training0202.mat

%converting character board representation into trinary representation

%{
BitBoard=zeros(1,361);
BitBoard(CharBoard=='w')=-1;
BitBoard(CharBoard=='b')=1;
BitBoard(CharBoard=='e')=0; 
%}

%initialization variables
offset=11;
gamma=0.99;    % discount factor
lambda=0.001;  % 0 >  lambda > 1 
numFeat=31;
totalFeaturesNum=9610-2883; 

FeaturesMatrix=zeros(41,41,20); % cell(1,size(currGame,1)); 

FeaturesMaps=cell(1,10);
numMaps=size(FeaturesMaps,2)*numFeat^2; % 10x31x31 cross-features
HigherOrderFeatures=zeros(1,numMaps);
HigherOrderFeatures_xy=zeros(1,totalFeaturesNum);

a = 5+5
class(a)
b = CharBoard(8)
class(b)

V = b;


end
