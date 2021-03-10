function [hidro1] = interpolate(hidro)
minimo=min(hidro);
hidro= hidro+abs(minimo);
row = find(isnan(hidro));
for i=1:length(hidro)
    if ismember(i,row)
        L=i+1;
        while ismember(L,row)
            L=L+1;
        end
        l=i-1;
        while ismember(l,row)
            l=l-1;
        end
        y1=hidro(l);
        y2=hidro(L);
        
     hidro1(i) = y1+((y2-y1)*(i-l)/(L-l))- abs(minimo);
    else
        hidro1(i)=hidro(i)- abs(minimo);
    end
    i=i+1;
end
end

