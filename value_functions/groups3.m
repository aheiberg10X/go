		
function [BlackGroups,AllBlackStones,WhiteGroups,AllWhiteStones] = groups3(Go_mx)
% identify groups of connected black and white stones, report the stones in each group and the # of liberties
% also output the coordinates of all Black and White stones
% AllBlackStones & AllWhiteStones is a 3 column matrix with the number of the group to which they belong and 0 if isolated
% 
% % [xb,yb]=find(Go_mx==1);
% % B=sortrows([xb yb]);
% % Bgroups=cell(1);
% % 
% % [xw,yw]=find(Go_mx==-1);
% % W=sortrows([xw,yw]);
% % Wgroups=cell(1);

%global G 

BlackGroups=struct;
WhiteGroups=struct;

for col=1:2 % for black (1) and white (2) groups
    
    [xb,yb]=find(Go_mx==1);
    B=sortrows([xb yb]);
    Bgroups=cell(1);

    [xw,yw]=find(Go_mx==-1);
    W=sortrows([xw,yw]);
    Wgroups=cell(1);
    
    if col==1
        color=1;
        AllBlackStones=B;
        AllBlackStones(:,3)=0;
    else
        color=-1;
        B=W;
        AllWhiteStones=B;
        AllWhiteStones(:,3)=0;
    end
    
    ingroup=zeros(length(B),2); %flag connected stones. 1st column for stones that are part of a group (avoid endless loops), 2nd column for stones in an ongoing group search (inside queue)| 1 = connected 0 = non-connected
    
    g=0;

    for t=1:size(B,1)

        queue=[];
        group=[];
        
        if isempty(ingroup)==0
        
        if ingroup(t,1)==0

            queue=[queue t];

            while isempty(queue)==0


                s=queue(1);

                x=B(s,1);
                y=B(s,2);

                cand=[];
                diagonal=[];

                switch x
                    case{1}
                        switch y 
                            case{1}
                                cand=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)+1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;

                                            end
                                        end
                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)+1);
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if Go_mx(x+1,y)==0 || Go_mx(x,y+1)==0
                                                    queue=[queue diagonal(d)];
                                                    ingroup(diagonal(d),1)=1;
                                                end
                                            end
                                        end
                                    end
                                end

                            case{19}
                                cand=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)-1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end
                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)-1);
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if Go_mx(x+1,y)==0 || Go_mx(x,y-1)==0
                                                    queue=[queue diagonal(d)];
                                                    ingroup(diagonal(d),1)=1;
                                                end
                                            end
                                        end
                                    end
                                end

                            otherwise
                                cand=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1) & B(:,2)==B(s,2)+1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end
                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)+1 );
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if B(s,2)-B(diagonal(d),2)>0
                                                    if Go_mx(B(s,1)+1,B(s,2))==color || Go_mx(B(s,1)+1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)-1)==color || Go_mx(B(s,1),B(s,2)-1)==0
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                else
                                                    if Go_mx(B(s,1)+1,B(s,2))==color || Go_mx(B(s,1)+1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)+1)==color || Go_mx(B(s,1),B(s,2)+1)==0
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                end
                                            end
                                        end
                                    end

                                end 
                        end

                    case{19}
                        switch y
                            case{1}
                                cand=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)+1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end
                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)+1);
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if Go_mx(x-1,y)==0 || Go_mx(x,y+1)==0
                                                    queue=[queue diagonal(d)];
                                                    ingroup(diagonal(d),1)=1;
                                                end
                                            end
                                        end
                                    end
                                end 

                            case{19}
                                cand=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)-1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end

                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)-1);
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if Go_mx(x-1,y)==0 || Go_mx(x,y-1)==0
                                                    queue=[queue diagonal(d)];
                                                    ingroup(diagonal(d),1)=1;
                                                end
                                            end
                                        end
                                    end
                                end  

                            otherwise
                                cand=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1) & B(:,2)==B(s,2)+1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end

                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)+1);
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if B(s,2)-B(diagonal(d),2)>0
                                                    if Go_mx(B(s,1)-1,B(s,2))==color || Go_mx(B(s,1)-1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)-1)==color || Go_mx(B(s,1),B(s,2)-1)==0
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                else
                                                    if Go_mx(B(s,1)-1,B(s,2))==color || Go_mx(B(s,1)-1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)+1)==color || Go_mx(B(s,1),B(s,2)+1)==0
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end 
                        end

                    otherwise
                        switch y
                            case{1}
                                cand=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2) | B(:,1)==B(s,1)-1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)+1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end

                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)+1 | B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)+1 );
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if B(s,1)-B(diagonal(d),1)>0
                                                    if Go_mx(B(s,1)-1,B(s,2))==color || Go_mx(B(s,1)-1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)+1)==color || Go_mx(B(s,1),B(s,2)+1)==0    
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                else
                                                    if Go_mx(B(s,1)+1,B(s,2))==color || Go_mx(B(s,1)+1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)+1)==color || Go_mx(B(s,1),B(s,2)+1)==0
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end 

                            case{19}
                                cand=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2) | B(:,1)==B(s,1)-1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)-1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end

                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)-1 );
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if B(s,1)-B(diagonal(d),1)>0
                                                    if Go_mx(B(s,1)-1,B(s,2))==color || Go_mx(B(s,1)-1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)-1)==color || Go_mx(B(s,1),B(s,2)-1)==0
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                else
                                                    if Go_mx(B(s,1)+1,B(s,2))==color || Go_mx(B(s,1)+1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)-1)==color || Go_mx(B(s,1),B(s,2)-1)==0
                                                        queue=[queue diagonal(d)];
                                                        ingroup(diagonal(d),1)=1;
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end 

                            otherwise
                                cand=find(B(:,1)==B(s,1)+1 & B(:,2)==B(s,2) | B(:,1)==B(s,1)-1 & B(:,2)==B(s,2) | B(:,1)==B(s,1) & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1) & B(:,2)==B(s,2)+1);
                                if isempty(cand)==0 
                                    for c=1:length(cand)
                                        if ingroup(cand(c),2)==0
                                            if ismember(cand(c),queue)==0
                                                queue=[queue cand(c)];
                                                ingroup(cand(c),1)=1;
                                            end
                                        end

                                    end
                                end

                                diagonal=find(B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1)-1 & B(:,2)==B(s,2)+1 | B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)-1 | B(:,1)==B(s,1)+1 & B(:,2)==B(s,2)+1);
                                if isempty(diagonal)==0
                                    for d=1:length(diagonal)
                                        if ingroup(diagonal(d),2)==0
                                            if ismember(diagonal(d),queue)==0
                                                if B(s,1)-B(diagonal(d),1)>0
                                                    if B(s,2)-B(diagonal(d),2)>0
                                                        if Go_mx(B(s,1)-1,B(s,2))==color || Go_mx(B(s,1)-1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)-1)==color || Go_mx(B(s,1),B(s,2)-1)==0    
                                                            queue=[queue diagonal(d)];
                                                            ingroup(diagonal(d),1)=1;                                          
                                                        end
                                                    else
                                                        if Go_mx(B(s,1)-1,B(s,2))==color || Go_mx(B(s,1)-1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)+1)==color || Go_mx(B(s,1),B(s,2)+1)==0
                                                            queue=[queue diagonal(d)];
                                                            ingroup(diagonal(d),1)=1;   
                                                        end

                                                    end
                                                else
                                                    if B(s,2)-B(diagonal(d),2)>0
                                                        if Go_mx(B(s,1)+1,B(s,2))==color || Go_mx(B(s,1)+1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)-1)==color || Go_mx(B(s,1),B(s,2)-1)==0
                                                            queue=[queue diagonal(d)];
                                                            ingroup(diagonal(d),1)=1;
                                                        end
                                                    else

                                                        if Go_mx(B(s,1)+1,B(s,2))==color || Go_mx(B(s,1)+1,B(s,2))==0 || Go_mx(B(s,1),B(s,2)+1)==color || Go_mx(B(s,1),B(s,2)+1)==0
                                                            queue=[queue diagonal(d)];
                                                            ingroup(diagonal(d),1)=1;
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end 
                        end

                end

            %cand=find(B(:,1)==B(s,1)+1 | B(:,2)==B(s,2)+1);  
    %         dif=[];
    %         dif=bsxfun(@minus, B(cand,:), B(s,:));
    %         noncand=find(dif(:,1)>1 | dif(:,2)>1)    
    %         dif(find(dif(:,1)>1 | dif(:,2)>1),:)=[];
            
            if length(queue)>1
                ingrp(s)=1;
            end
            
            %queue=unique(queue);
            ingroup(queue,1)=1; 
            group=[group queue];
            group=unique(group);
            ingroup(group,2)=1;
            queue(1)=[];

            %q=ismember(queue,group);
            %ingroup(queue(q==1),2)=1;
            %queue(q==1)=[];

            end

        end
        
        end
        
        if length(group)>1
            g=g+1;            
            Bgroups{g}=B(group(:),:)';   
        end

    end
    
    % store the groups in a structure with:
    % - the coordinates of the stones in the .stones field
    % - the number of stones in the group in the .num field
    % - the number of the group's liberties in the .lib field
    
%     BlackGroups=struct;
%     WhiteGroups=struct;
    
    if isempty(Bgroups{1})==0 %length(Bgroups)>1
        for z=1:length(Bgroups)
            if col==1
                BlackGroups(z).stones=Bgroups{z};
                BlackGroups(z).num=length(Bgroups{z});
                AllBlackStones=B;
            else
                WhiteGroups(z).stones=Bgroups{z};
                WhiteGroups(z).num=length(Bgroups{z});
                AllWhiteStones=B;
            end
        end
        
    % computing the groups' liberties

    % boardLib=zeros(19,19);
    % boardLib=Go_mx;
    
        for numg=1:length(Bgroups)

            li=0;
            boardLib=zeros(19,19);
            boardLib=Go_mx;
            
            liblib=[];

            for numsto=1:length(Bgroups{numg})

                xst=Bgroups{numg}(1,numsto);
                yst=Bgroups{numg}(2,numsto);

                if col==1
                   %ingrp=find(AllBlackStones(:,1)==xst & AllBlackStones(:,2)==yst);
                   AllBlackStones(AllBlackStones(:,1)==xst & AllBlackStones(:,2)==yst,3)=numg;
                else
                    %ingrp=find(AllWhiteStones(:,1)==xst & AllWhiteStones(:,2)==yst);
                    AllWhiteStones(AllWhiteStones(:,1)==xst & AllWhiteStones(:,2)==yst,3)=numg;
                end


                switch xst
                    case{1}

                        switch yst
                            case{1}
                                if boardLib(xst+1,yst)==0
                                   li=li+1;
                                   lico=[xst+1;yst];
                                   liblib=[liblib lico]; 
                                   boardLib(xst+1,yst)=color;
                                end
                                if boardLib(xst,yst+1)==0
                                    li=li+1;
                                    lico=[xst;yst+1];
                                    liblib=[liblib lico];
                                    boardLib(xst,yst+1)=color;
                                end
                            case{19}
                                if boardLib(xst+1,yst)==0
                                   li=li+1;
                                   lico=[xst+1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst+1,yst)=color;
                                end
                                if boardLib(xst,yst-1)==0
                                   li=li+1;
                                   lico=[xst;yst-1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst-1)=color;
                                end
                            otherwise
                                if boardLib(xst+1,yst)==0
                                   li=li+1;
                                   lico=[xst+1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst+1,yst)=color;
                                end
                                if boardLib(xst,yst-1)==0
                                   li=li+1;
                                   lico=[xst;yst-1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst-1)=color;
                                end
                                if boardLib(xst,yst+1)==0
                                   li=li+1;
                                   lico=[xst;yst+1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst+1)=color;
                                end
                        end

                    case{19}

                        switch yst
                            case{1}
                                if boardLib(xst-1,yst)==0
                                   li=li+1;
                                   lico=[xst-1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst-1,yst)=color;
                                end
                                if boardLib(xst,yst+1)==0
                                   li=li+1;
                                   lico=[xst;yst+1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst+1)=color;
                                end
                            case{19}
                                if boardLib(xst-1,yst)==0
                                   li=li+1;
                                   lico=[xst-1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst-1,yst)=color;
                                end
                                if boardLib(xst,yst-1)==0
                                   li=li+1;
                                   lico=[xst;yst-1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst-1)=color;
                                end
                            otherwise
                                if boardLib(xst-1,yst)==0
                                   li=li+1;
                                   lico=[xst-1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst-1,yst)=color;
                                end
                                if boardLib(xst,yst-1)==0
                                   li=li+1;
                                   lico=[xst;yst-1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst-1)=color;
                                end
                                if boardLib(xst,yst+1)==0
                                   li=li+1;
                                   lico=[xst;yst+1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst+1)=color;
                                end
                        end

                    otherwise

                        switch yst
                            case{1}
                                if boardLib(xst,yst+1)==0
                                   li=li+1;
                                   lico=[xst;yst+1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst+1)=color;
                                end
                                if boardLib(xst-1,yst)==0
                                   li=li+1;
                                   lico=[xst-1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst-1,yst)=color;
                                end
                                if boardLib(xst+1,yst)==0
                                   li=li+1;
                                   lico=[xst+1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst+1,yst)=color;
                                end
                            case{19}
                                if boardLib(xst,yst-1)==0
                                   li=li+1;
                                   lico=[xst;yst-1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst-1)=color;
                                end
                                if boardLib(xst-1,yst)==0
                                   li=li+1;
                                   lico=[xst-1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst-1,yst)=color;
                                end
                                if boardLib(xst+1,yst)==0
                                   li=li+1;
                                   lico=[xst+1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst+1,yst)=color;
                                end
                            otherwise
                                if boardLib(xst-1,yst)==0
                                   li=li+1;
                                   lico=[xst-1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst-1,yst)=color;
                                end
                                if boardLib(xst+1,yst)==0
                                   li=li+1;
                                   lico=[xst+1;yst];
                                   liblib=[liblib lico];
                                   boardLib(xst+1,yst)=color;
                                end
                                if boardLib(xst,yst-1)==0
                                   li=li+1;
                                   lico=[xst;yst-1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst-1)=color;
                                end
                                if boardLib(xst,yst+1)==0
                                   li=li+1;
                                   lico=[xst;yst+1];
                                   liblib=[liblib lico];
                                   boardLib(xst,yst+1)=color;
                                end

                        end
                end

            end

            if col==1
                BlackGroups(numg).lib=li;
                BlackGroups(numg).libcoor=liblib;
            else
                WhiteGroups(numg).lib=li;
                WhiteGroups(numg).libcoor=liblib;
            end

        end
    
    else
        if col==1
%             BlackGroups(z).stones=[];
%             BlackGroups(z).num=0;
            AllBlackStones=B;
            AllBlackStones(:,3)=0;
        else
%             WhiteGroups(z).stones=[];
%             WhiteGroups(z).num=0;
            AllWhiteStones=B;
            AllWhiteStones(:,3)=0;
        end
    end
    
    %WhiteGroups=BlackGroups;
   
end







