function DiffManhDist = DiffManhattanDist(x,y,black,white,color)
% compute the difference of the Manhattan Distances between the nearest neighbor of the same color
% and the one of nearest enemy stone for a newly added stone on the board

% negative is closer to friendly stone than enemy's one
% 0 for stones touching friendly and enemy stones

% x_stone,y_stone = coordinate of the new stone
% black = all Black stones
% white = all White stones

% ---------- for all stones or only new stone ???? ------------------------

newStone=[x y];

% remove new stone from list of all stones

if isempty(black)==0
%    [pos,a]=find(newStone(1)==black(:,1) & newStone(2)==black(:,2));
%    if isempty(pos)==0
%       black(pos,:)=[]; 
%    end 
   ManhDistBlack=sum([abs(black(:,1)-newStone(1)) abs(black(:,2)-newStone(2))],2); 
   ManhDistBlack=ManhDistBlack(find(ManhDistBlack)); % remove newStone if black; ManhDist==0
   if isempty(ManhDistBlack)==1
       ManhDistBlack=0;
   end
   [minMDblack,p]=min(ManhDistBlack); 
else
   minMDblack=0; 
end


if isempty(white)==0
%    [pos,a]=find(newStone(1)==white(:,1) & newStone(2)==white(:,2));
%    if isempty(pos)==0
%       white(pos,:)=[]; 
%    end  
   ManhDistWhite=sum([abs(white(:,1)-newStone(1)) abs(white(:,2)-newStone(2))],2);
   ManhDistWhite=ManhDistWhite(find(ManhDistWhite)); % remove newStone if white; ManhDist==0
   if isempty(ManhDistWhite)==1
       ManhDistWhite=0;
   end
   [minMDwhite,p]=min(ManhDistWhite);
else
   minMDwhite=0;
end

%
if color==1
   DiffManhDist=minMDblack-minMDwhite;
else
   DiffManhDist=minMDwhite-minMDblack;
end
