
function EVAL = Evaluate(Yts, y_hat, data)
Conf = confusionmat(Yts,y_hat);
Precision = diag(Conf)./sum(Conf,2);
Sensisivity = diag(Conf)./sum(Conf,1)';
Precision1 = max(mean(Precision),mean(1-Precision));
Sensisivity1 = max(mean(Sensisivity),mean(1-Sensisivity));
F_score = 2*(Precision1.*Sensisivity1)./(Precision1+Sensisivity1);
errorRate = (sum(Conf,2)-diag(Conf))./sum(Conf,2);

clusters_number = unique(Yts);
distM = squareform(pdist(data));
ind = y_hat;
i=clusters_number;

denominator=[];
for i2=1:i
    indi=find(ind==i2);
    indj=find(ind~=i2);
    x=indi;
    y=indj;
    temp=distM(x,y);
    denominator=cat(1, denominator, temp(:));
end

num=min(min(denominator)); 
neg_obs=zeros(size(distM,1),size(distM,2));

for ix=1:i
    indxs=find(ind==ix);
    neg_obs(indxs,indxs)=1;
end

dem=neg_obs.*distM;
dem=max(max(dem));

DI=num/dem;

a = Yts;
b = y_hat;
n = numel(a);
I = max(a);
J = max(b);
C = zeros(I, J);
for i = 1:I
    tmp = a==i;
    for j = 1:J
        C(i,j) = sum(tmp & b==j);
    end
end


CC = C.^2;
CC = bsxfun(@rdivide, CC, sum(CC, 1));
CC = bsxfun(@rdivide, CC, sum(CC, 2));
chi = ( sum(sum(CC, 1), 2) - 1 ) * n;

C = C / n;

if numel(C) == 1
    MOC = 1;
else
    MOC = chi / n / ( sqrt(I*J) - 1 );
end
        
Pxy = nonzeros(C);
Px = mean(C, 1);
Py = mean(C, 2);
Hxy = -dot(Pxy, log2(Pxy+eps));
Hx = -dot(Px, log2(Px+eps));
Hy = -dot(Py, log2(Py+eps));
MI = Hx + Hy - Hxy;
NMI = sqrt((MI/Hx)*(MI/Hy));
NormalizedVariationofInformation = 2-(Hx+Hy)/Hxy;
NormalizedVariationofInformation = max(0,NormalizedVariationofInformation);

idx = (Yts==1);
p = length(Yts(idx));
n = length(Yts(~idx));
N = p+n;
tp = sum(Yts(idx)==y_hat(idx));
tn = sum(Yts(~idx)==y_hat(~idx));
fp = n-tn;
fn = p-tp;
temp = (tp+tn)/(tp+tn+fp+fn);
RandIndex = min([temp, 1-temp]);

temp1 = (tp)/(tp+fp+fn);
JaccardIndicator = max([temp1, 1-temp1]);

        
f_measure = F_score;  %% Max for which maybe algorithm dont can detect class labels similar, but its distinction was good
ErrorRate = min(mean(errorRate),mean(1-errorRate));
EVAL = [f_measure, ErrorRate, DI, RandIndex, JaccardIndicator,...
    NMI, NormalizedVariationofInformation, MOC, Precision1, Sensisivity1];
end