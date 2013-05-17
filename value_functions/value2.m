function V = ComputeValueBoard_binaryFeat(BitBoard,Weights)

% compute the value V for a given 19x19 board configuration
% using binary features
% input BitBoard = 1x361 character vector (set={'w','b','e'})


%load precomputed weight vector
%load Training0202.mat
%global Weights
%load Weights

%converting character board representation into trinary representation
%BitBoard=zeros(1,361);
%BitBoard(CharBoard=='w')=-1;
%BitBoard(CharBoard=='b')=1;
%BitBoard(CharBoard=='e')=0; 

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


%matrix representation of the board 
Goban=zeros(19,19);
Goban(:)=BitBoard;

[BlackGroups,AllBlackStones,WhiteGroups,AllWhiteStones] = groups3(Goban); %find alive groups

psimatrix=zeros(41,41,numFeat); % embedding board into a 41x41 matrix to take into account fringe effects for the Gaussian and Gabor filters

N=size(psimatrix,1);

psimatrix(1:end,[1:11 31:end],1)=1;     % off-limit feature
psimatrix([1:11 31:end],12:30,1)=1;     % off-limit feature
psimatrix(1:end,[1:11 31:end],2:end)=0;    
psimatrix([1:11 31:end],12:30,2:end)=0; 

%compute binary features for each intersection ----------------------------
for k=1:361 

    y=ceil(k/19);
    x=k-(y-1)*19;

    color=Goban(k);

    if color==0
       psimatrix(x+offset,y+offset,2)=1;
    elseif color==1
        psimatrix(x+offset,y+offset,5)=1; % features 5 = all Black stones present on the board

        grpnum=AllBlackStones(AllBlackStones(:,1)==x & AllBlackStones(:,2)==y,3);
        if grpnum~=0
            lib=BlackGroups(grpnum).lib;
            switch lib
                case{0}                                 
                %    psimatrix(x+offset,y+offset,5)=1;
                    display('Warning: There is a Black group with 0 liberty !');
                case{1}
                    psimatrix(x+offset,y+offset,6)=1;
                case{2}
                    psimatrix(x+offset,y+offset,7)=1;
                case{3}
                    psimatrix(x+offset,y+offset,8)=1;
                case{4}
                    psimatrix(x+offset,y+offset,9)=1;
                case{5,6}
                    psimatrix(x+offset,y+offset,10)=1;
                case{7,8,9}
                    psimatrix(x+offset,y+offset,11)=1;
                otherwise
                    psimatrix(x+offset,y+offset,12)=1;
            end
        else
            psimatrix(x+offset,y+offset,3)=1;
        end
    else
        psimatrix(x+offset,y+offset,13)=1; % feature 13 = all White stones present on the board

        grpnum=AllWhiteStones(AllWhiteStones(:,1)==x & AllWhiteStones(:,2)==y,3);
        if grpnum~=0
            lib=WhiteGroups(grpnum).lib;
            switch lib
                case{0}                                 
                %    psimatrix(x+offset,y+offset,13)=1;
                    display('Warning: There is a White Group with 0 liberty !');
                case{1}
                    psimatrix(x+offset,y+offset,14)=1;
                case{2}
                    psimatrix(x+offset,y+offset,15)=1;
                case{3}
                    psimatrix(x+offset,y+offset,16)=1;
                case{4}
                    psimatrix(x+offset,y+offset,17)=1;
                case{5,6}
                    psimatrix(x+offset,y+offset,18)=1;
                case{7,8,9}
                    psimatrix(x+offset,y+offset,19)=1;
                otherwise
                    psimatrix(x+offset,y+offset,20)=1;
            end
        else
            psimatrix(x+offset,y+offset,4)=1;
        end
    end

    if color~=0

        DiffManhDist=DiffManhattanDist(x,y,AllBlackStones,AllWhiteStones,color);

        switch DiffManhDist
            case{0}
                psimatrix(x+offset,y+offset,21)=1;
            case{-1}
                psimatrix(x+offset,y+offset,22)=1; 
            case{-2,-3}
                psimatrix(x+offset,y+offset,23)=1;
            case{-4,-5}
                psimatrix(x+offset,y+offset,24)=1;
            case{-6,-7,-8}
                psimatrix(x+offset,y+offset,25)=1;
            case{1}
                psimatrix(x+offset,y+offset,27)=1;
            case{2,3}
                psimatrix(x+offset,y+offset,28)=1; 
            case{4,5}
                psimatrix(x+offset,y+offset,29)=1;
            case{6,7,8}
                psimatrix(x+offset,y+offset,30)=1;
            otherwise
                if DiffManhDist>8
                    psimatrix(x+offset,y+offset,31)=1;
                elseif DiffManhDist<-8                              
                    psimatrix(x+offset,y+offset,26)=1;
                end
        end
    end

end %binary features


FeaturesMatrix=psimatrix; %


% BASIC FEATURES - convolution w/ 3 kernels (A1x*A1y,A1x*A1y*D1x,A1x*A1y*D1y) for 3 different scales (numConv)
% A1x = 1D Gaussian kernel on x,  A1y = 1D Gaussian kernel on y
% D1x = 1D Gabor kernel on x,     D1y = 1D Gabor kernel on y 

id=0;

Features1=zeros(N,N,size(psimatrix,3));  
Features2=zeros(N,N,size(psimatrix,3)); 
Features3=zeros(N,N,size(psimatrix,3)); 

% 1st feature (off-limit) is a special case, embedded in 1s
Features1(:,:,1)=1; 
Features2(:,:,1)=1; 
Features3(:,:,1)=1;

Features1(:,:,1)=FeaturesMatrix(:,:,1); 

for numConv=[1 5 12] % # of convolution at the small, meso and large scales
    id=id+1;

    % Features1=zeros(N,N,size(psimatrix,3)); Features2=zeros(N,N,size(psimatrix,3)); Features3=zeros(N,N,size(psimatrix,3)); 

    % 1st feature (off-limit) is a special case, embedded in 1s
    %Features1(:,:,1)=1; Features2(:,:,1)=1; Features3(:,:,1)=1; 

    ind1=numConv;

    %Features1(:,:,1)=FeaturesMatrix{1,jboard}(:,:,1);

    while (ind1~=0)

        Features1(:,:,1)=0.5*Features1(:,:,1) + 0.25*[Features1(:,2:N,1) ones(N,1)] + 0.25*[ones(N,1) Features1(:,1:N-1,1)]; % A1x
        Features1(:,:,1)=0.5*Features1(:,:,1) + 0.25*[Features1(2:N,:,1); ones(1,N)] + 0.25*[ones(1,N); Features1(1:N-1,:,1)]; % A1y

        ind1=ind1-1;

    end

    if id~=3

        Features2(:,:,1)=-0.5.*[Features1(:,2:N,1) ones(N,1,1)]+0.5.*[ones(N,1,1) Features1(:,1:N-1,1)]; % D1x

        %Features3(:,:,1)=-0.5.*[Features2(2:N,:,1); ones(1,N,1)]+0.5.*[ones(1,N,1); Features2(1:N-1,:,1)]; % D1y
        Features3(:,:,1)=-0.5.*[Features1(2:N,:,1); ones(1,N,1)]+0.5.*[ones(1,N,1); Features1(1:N-1,:,1)]; % D1y 

    else
        Features2(:,:,1)=-0.5.*[FeaturesMatrix(:,2:N,1) ones(N,1,1)]+0.5.*[ones(N,1,1) FeaturesMatrix(:,1:N-1,1)]; % D1x
        Features3(:,:,1)=-0.5.*[FeaturesMatrix(2:N,:,1); ones(1,N,1)]+0.5.*[ones(1,N,1); FeaturesMatrix(1:N-1,:,1)]; % D1y 
    end

%      Features2(:,:,1)=-0.5.*[Features1(:,2:N,1) ones(N,1,1)]+0.5.*[ones(N,1,1) Features1(:,1:N-1,1)]; % D1x    
%      Features3(:,:,1)=-0.5.*[Features2(2:N,:,1); ones(1,N,1)]+0.5.*[ones(1,N,1); Features2(1:N-1,:,1)]; % D1y

    % other features are embedded in 0s
    for F=2:size(psimatrix,3)

        Features1(:,:,F)=FeaturesMatrix(:,:,F);

        if numConv==1
            ind2=numConv;
        else
            ind2=numConv+1;
        end

        while (ind2~=0)

            Features1(:,:,F)=0.5*Features1(:,:,F) + 0.25*[Features1(:,2:N,F) zeros(N,1)] + 0.25*[zeros(N,1) Features1(:,1:N-1,F)]; % A1x
            Features1(:,:,F)=0.5*Features1(:,:,F) + 0.25*[Features1(2:N,:,F); zeros(1,N)] + 0.25*[zeros(1,N); Features1(1:N-1,:,F)]; % A1y

            ind2=ind2-1;
        end

        if id~=3
            Features2(:,:,F)=-0.5.*[Features1(:,2:N,F) zeros(N,1)]+0.5.*[zeros(N,1) Features1(:,1:N-1,F)]; % D1x

            Features3(:,:,F)=-0.5.*[Features1(2:N,:,F); zeros(1,N)]+0.5.*[zeros(1,N); Features1(1:N-1,:,F)]; % D1y
        else
            Features2(:,:,F)=-0.5.*[FeaturesMatrix(:,2:N,F) zeros(N,1)]+0.5.*[zeros(N,1) FeaturesMatrix(:,1:N-1,F)]; % D1x 
            Features3(:,:,F)=-0.5.*[FeaturesMatrix(2:N,:,F); zeros(1,N)]+0.5.*[zeros(1,N); FeaturesMatrix(1:N-1,:,F)]; % D1y 
        end

    end

      FeaturesMaps{1,(id-1)*3+1}=Features1;
      FeaturesMaps{1,(id-1)*3+2}=Features2;
      FeaturesMaps{1,(id-1)*3+3}=Features3;

end

FeaturesMaps{1,10}=FeaturesMatrix;

% compute CROSS-FEATURES --> 20x20x9 cross-products among convolved features
 for cf1=1:size(psimatrix,3)
    for cf2=1:size(psimatrix,3)
        for indim=1:size(FeaturesMaps,2)

            index=(cf1-1)*310+(10*(cf2-1)+indim);
            HigherOrderFeatures(1,index)=sum(sum(FeaturesMaps{1,10}(:,:,cf1).*FeaturesMaps{1,indim}(:,:,cf2)));

        end
    end
 end

% higher-order Gabors xy --> sqrt(x^2+y^2)
xx=HigherOrderFeatures(1,[2:10:9602 5:10:9605 8:10:9608]).^2;
yy=HigherOrderFeatures(1,[3:10:9603 6:10:9606 9:10:9609]).^2;
HigherOrderFeatures_xy(1,:)=[HigherOrderFeatures(1,[1:10:9601 4:10:9604 7:10:9607 10:10:9610]) sqrt(xx+yy)];        

%size(HigherOrderFeatures_xy)
%size(Weights)
V = HigherOrderFeatures_xy*Weights; 

% scoreVal=zeros(size(Games{g,3},1),2); % V, score estimation
% 
% for kk=1:size(Games{g,3})
%     scoreVal(kk,1)=HigherOrderFeatures_xy(kk,:)*Weights;
%     scoreVal(kk,2)=Games{g,3}(kk);
% end

        
