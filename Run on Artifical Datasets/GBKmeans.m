function Output = GBKmeans(X, Ncluster, PSOparams)
% This function is the implementation of the GBK-means Clustering
% Algorithm.
% Inputs:
 %  X: Input data as Ninstances*Nattributes. Observations must be in rows
 %  Ncluster:  Desired number of clusters
 %  PSOparams: A structure consist of:
 % .MaxIt: Maximum Number of Iterations in PSO
 % .nPop: Population Size (Swarm Size) in PSO
 % .w: Inertia Weigh
 % .c1: Personal Learning Coefficient in PSO
 % .c2: Global Learning Coefficient in PSO
 % .WorkSpaceDisplyIteration: iterations between display of optimization in workspace, a integer between 1 and MaxIt+1
 % .PlotConvergence: Plot of convergence in PSO, 0 or 1
% Output: A structure cosist of:
 % .Idx: Cluster index of each instance
 % .BestCost: Convergence steps of PSO

%% Parameters

if nargin < 2
    
error('Number of inputs must be more than two')    

elseif nargin==3

model.M = X;
model.Ncluster = Ncluster;                                     %% 
model.Ninstance = size(X,1);                                   %% Number of instances
model.Nattribute = size(X,2);                                  %% Number of attributes
nVar = model.Ncluster*model.Nattribute;                        %% number of decision variables
VarSize = [1, nVar];                                           %% size of decision variables matrix in PSO
model.LB = min(X);                                             %% lower bound data
model.UB = max(X);                                             %% upper bound data
VarMin=0*ones(1,nVar);                                         %% Lower Bound of the PSO swarms
VarMax=1*ones(1,nVar);                                         %% Upper Bound of the PSO swarms
MaxIt = PSOparams.MaxIt;                                       %% Maximum Number of Iterations in PSO
nPop = PSOparams.nPop;                                         %% Population Size (Swarm Size) in PSO
w = PSOparams.w;                                               %% Inertia Weigh
wdamp = 0.99;                                                  %% Inertia Weight Damping Ratio in PSO
c1 = PSOparams.c1;                                             %% Personal Learning Coefficient in PSO
c2 = PSOparams.c2;                                             %% Global Learning Coefficient in PSO
WorkSpaceDisplyIteration = PSOparams.WorkSpaceDisplyIteration; %%
PlotConvergence = PSOparams.PlotConvergence;                   %%

elseif nargin<3

model.M = X;    
model.Ncluster = Ncluster;                                     %% 
model.Ninstance = size(X,1);                                   %% Number of instances
model.Nattribute = size(X,2);                                  %% Number of attributes
nVar = model.Ncluster*model.Nattribute;                        %% number of decision variables
VarSize = [1, nVar];                                           %% size of decision variables matrix in PSO
model.LB = min(X);                                             %% lower bound of data
model.UB = max(X);                                             %% upper bound of data
VarMin=0*ones(1,nVar);                                         %% Lower Bound of the PSO swarms
VarMax=1*ones(1,nVar);                                         %% Upper Bound of the PSO swarms
MaxIt = 100;                                                   %% Maximum Number of Iterations in PSO
nPop = 50;                                                     %% Population Size (Swarm Size) in PSO
w = 1;                                                         %% Inertia Weigh
wdamp = 0.99;                                                  %% Inertia Weight Damping Ratio in PSO
c1 = 2;                                                        %% Personal Learning Coefficient in PSO
c2 = 2.5;                                                        %% Global Learning Coefficient in PSO  
WorkSpaceDisplyIteration = MaxIt+1;                              %%
PlotConvergence = 0;                                           %%

end

%% Run PSO

% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;
CostFunction=@(x, model) fitness(x, model);        % Cost Function


%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.sol=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.Best.sol=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=-inf;

for i=1:nPop
    
    % Initialize Position
    particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    [particle(i).Cost,particle(i).sol]=CostFunction(particle(i).Position, model);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    particle(i).Best.sol=particle(i).sol;
    % Update Global Best
    if particle(i).Best.Cost>GlobalBest.Cost      
        GlobalBest=particle(i).Best;       
    end   
end

%% PSO Main Loop

BestCost=zeros(MaxIt,1);
for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity,VelMin);
        particle(i).Velocity = min(particle(i).Velocity,VelMax);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
        
        % Evaluation
        [particle(i).Cost,particle(i).sol] = CostFunction(particle(i).Position, model);
        
        % Update Personal Best
        if particle(i).Cost>particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            particle(i).Best.sol=particle(i).sol;
            % Update Global Best
            if particle(i).Best.Cost>GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
        if GlobalBest.sol.info.isfeasible
            Flag=' * ';
        else
            Flag=' ';
        end
    if mod(it,WorkSpaceDisplyIteration)==0
     disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it)) Flag]);  
    end
    w=w*wdamp;   
end
if PlotConvergence
    figure;
    semilogy(BestCost,'LineWidth',2);
    xlabel('Iteration');
    ylabel('Best Cost'); 
end
Output.Idx = GlobalBest.sol.info.clu;
Output.BestCost = BestCost;







function [fit,Sol]=fitness(x, model)

%% Check boundaries
x=CB(x,0,1);

%% Determine Center
x=reshape(x,model.Ncluster, model.Nattribute);
x=x.*(model.UB-model.LB);
center=x+model.LB;

%% calculating norm 2 
tdis=pdist2(center,model.M);
[Dis,clu]=min(tdis);

%% Objective function
try
Dist1 = 0.7*max(Dis(clu==1))+0.3*mean(Dis(clu==1));
Dist2 = 0.7*max(Dis(clu==2))+0.3*mean(Dis(clu==2)); 
minDist1 = min(Dis(clu==1));
minDist2 = min(Dis(clu==2));

u0 = 0.9*mean([1/Dist1, 1/Dist2]);
v0 = 0.9*mean([1/Dist1, 1/Dist2]);
V1 = max(0,(minDist1/Dist1)-1);
V2 = max(0,(minDist2/Dist2)-1);
V3 = max(0,(u0/(1/Dist1))-1);
V4 = max(0,(v0/(1/Dist2))-1);

fit = (1/Dist1)*(1/Dist2)-10000*V3-10000*V4;

Sol.info.dis=Dis;
Sol.info.clu=clu;
Sol.info.center=center;
Sol.info.isfeasible=((V1+V2+V3+V4)==0);

catch
fit=-10e20; 
Sol.info.dis=nan;
Sol.info.clu=nan;
Sol.info.center=nan;
Sol.info.isfeasible=nan;
end

function x=CB(x,lb,ub)
     x=max(x,lb);
     x=min(x,ub);
end
end
end