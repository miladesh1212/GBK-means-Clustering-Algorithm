
%% Main code

clc;
clear;
close all;
warning off;
addpath('Data')

%% Data

Data = Synthetic2();

%% Normalization

Data(:,1:end-1) = Normalize_Fcn(Data(:,1:end-1));

%% Data Detection

X = Data(:,1:end-1);    %% Input data. observations must be in rows
T = Data(:,end);        %% Target labels
Ncluster = 2;           %% number of clusters, because of our exprimental setup = 2


%%

for itt=1:30  %% run 30 iteration and calculate their means for return stable results

%% GBK-means Clustering Plots

Ouput = GBKmeans(X, Ncluster);
idx = Ouput.Idx;

%% K-means Clustering Plots

idx1 = kmeans(X, Ncluster);


%% Fuzzy C-means Clustering Plots

options = [3.0 100 NaN 0];
[centers,U] = fcm(X, Ncluster, options);
[~, idx2] = max(U);
idx2 = idx2';


%% Clustering Plots
% % oncomment this section for displaying quality of GBK-means
% % clustering, k-means clustering and fuzzy c-means on current dataset

% % Target Clustering Plot
% figure;
% plot(X(T==1,1), ...
%     X(T==1,2),'r.','MarkerSize',12)
% hold on
% plot(X(T==2,1),...
%     X(T==2,2),'b.','MarkerSize',12)
% legend('Cluster 1','Cluster 2',...
%        'Location','N', 'TextColor', [0.4 0.6 0.4])
% title('Correct Clustering', 'Color', [0.4 0.6 0.4]) 
% grid on;

% % GBK-means Clustering Plot
% figure;
% plot(X(idx==1,1), ...
%     X(idx==1,2), 'r.','MarkerSize',12)
% hold on
% plot(X(idx==2,1),...
%     X(idx==2,2), 'b.','MarkerSize',12)
% legend('Cluster 1','Cluster 2',...
%        'Location','N', 'TextColor', [0.4 0.6 0.4])
% title('GBK-means Clustering Approach', 'Color', [0.4 0.6 0.4])
% grid on;

% % K-means Clustering Plot
% figure;
% plot(X(idx1==1,1), ...
%     X(idx1==1,2), 'r.','MarkerSize',12)
% hold on
% plot(X(idx1==2,1),...
%     X(idx1==2,2),'b.','MarkerSize',12)
% legend('Cluster 1','Cluster 2',...
%        'Location','N', 'TextColor', [0.4 0.6 0.4])
% title('K-means Clustering', 'Color', [0.4 0.6 0.4])
% grid on;


% % Fuzzy c-means Clustering Plot
% figure;
% plot(X(idx2==1,1), ...
%     X(idx2==1,2),'r.','MarkerSize',12)
% hold on
% plot(X(idx2==2,1),...
%     X(idx2==2,2),'b.','MarkerSize',12)
% legend('Cluster 1','Cluster 2',...
%        'Location','N',  'TextColor', [0.4 0.6 0.4])
% title('Fuzzy C-means Clustering', 'Color', [0.4 0.6 0.4])
% grid on;


%% Evaluation and Save Results

if itt==1
EVAL1 = Evaluate(T, idx', X);
else
temp = Evaluate(T, idx', X);   
EVAL1 = cat(1, EVAL1, temp);
end

if itt==1
EVAL2 = Evaluate(T, idx1, X);
else
temp = Evaluate(T, idx1, X);   
EVAL2 = cat(1, EVAL2, temp);
end

if itt==1
EVAL3 = Evaluate(T, idx2, X);
else
temp = Evaluate(T, idx2, X);  
EVAL3 = cat(1, EVAL3, temp);
end
end

% % Max of 30 iteration results
[~, temp1] = max(EVAL1(:,1));
[~, temp2] = max(EVAL2(:,1));
[~, temp3] = max(EVAL3(:,1));
EVAL1 = EVAL1(temp1,:);
EVAL2 = EVAL2(temp2,:);
EVAL3 = EVAL3(temp3,:);

Table = table([EVAL1(1); EVAL1(2);EVAL1(3);EVAL1(4);EVAL1(5);EVAL1(6);EVAL1(7);...
    EVAL1(8);EVAL1(9);EVAL1(10)],...
    [EVAL2(1); EVAL2(2);EVAL2(3);EVAL2(4);EVAL2(5);EVAL2(6);EVAL2(7);...
    EVAL2(8);EVAL2(9);EVAL2(10)],...
    [EVAL3(1); EVAL3(2);EVAL3(3);EVAL3(4);EVAL3(5);EVAL3(6);EVAL3(7);
    EVAL3(8);EVAL3(9);EVAL3(10)],...
          'VariableNames',{'GBC','Kmeans','FCM'},...
          'RowNames',{'F';'ER';'DI';'RI';'JI';'NMI';'NVI';...
          'MOC';'Precision'; 'Recall'});

disp(Table)  
    
% writetable(Table, 'Result1.xls', 'Sheet', 'Artificialdataset2')  
