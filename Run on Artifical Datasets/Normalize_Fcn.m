function [xN, MaxX, MinX] = Normalize_Fcn(x)

 MaxX = max(x,[],1);  
 MinX = min(x,[],1);
 
 xN= (x-MinX)./ (MaxX-MinX);
 end
