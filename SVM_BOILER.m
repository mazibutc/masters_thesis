load FirstPart.txt
load SecondPart.txt
Training_x=FirstPart(:,[14:16,19:end]);
%mdl_Temp_1=fitrsvm(Training_x,FirstPart(:,1),'KernelFunction','rbf',...
  %  'KernelScale','auto','Standardize',true);
%conv=mdl_Temp_1.ConvergenceInfo.Converged;
%iter=mdl_Temp_1.NumIterations;
%disp(conv);
%disp(iter);
Testing_x=SecondPart(:,[14:16,19:end]);
%Temp_1_pred=predict(mdl_Temp_1,Testing_x);
%TError=SecondPart(:,1)-Temp_1_pred;
%MSE=mse(TError);
%disp(MSE);
%disp(max(TError));
%disp(min(TError));
%var=TError==max(TError);
%var2=TError==min(TError);
%SecondPart(var,1);
%SecondPart(var2,1);
%std(TError);
%% Test other functtionalities like 
mdl_2_Temp_1=fitrsvm(Training_x,FirstPart(:,1),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Temp_1_pred_2=predict(mdl_2_Temp_1,Testing_x);
TError=SecondPart(:,1)-Temp_1_pred_2;
MSE=mse(TError);
mape_1=mean(abs(TError)./SecondPart(:,1))*100;
%% 
std(TError)
boxplot(TError)
%% 
mdl_Temp_2=fitrsvm(Training_x,FirstPart(:,2),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Temp_2_pred=predict(mdl_Temp_2,Testing_x);
TError_2=SecondPart(:,2)-Temp_2_pred;
MSE_2=mse(TError_2);
mape_2=mean(abs(TError_2)./SecondPart(:,2))*100;
%% 
mdl_Temp_3=fitrsvm(Training_x,FirstPart(:,3),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Temp_3_pred=predict(mdl_Temp_3,Testing_x);
TError_3=SecondPart(:,3)-Temp_3_pred;
MSE_3=mse(TError_3);
mape_3=mean(abs(TError_3)./SecondPart(:,3))*100;
%% 
mdl_Temp_4=fitrsvm(Training_x,FirstPart(:,4),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Temp_4_pred=predict(mdl_Temp_4,Testing_x);
TError_4=SecondPart(:,4)-Temp_4_pred;
MSE_4=mse(TError_4);
mape_4=mean(abs(TError_4)./SecondPart(:,4))*100;
%% 
mdl_Temp_5=fitrsvm(Training_x,FirstPart(:,5),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Temp_5_pred=predict(mdl_Temp_5,Testing_x);
TError_5=SecondPart(:,5)-Temp_5_pred;
MSE_5=mse(TError_5);
mape_5=mean(abs(TError_5)./SecondPart(:,5))*100;
%% 
mdl_Temp_6=fitrsvm(Training_x,FirstPart(:,6),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Temp_6_pred=predict(mdl_Temp_6,Testing_x);
TError_6=SecondPart(:,6)-Temp_6_pred;
MSE_6=mse(TError_6);
mape_6=mean(abs(TError_6)./SecondPart(:,6))*100;
%% 
save mdl_Temp_3
save mdl_Temp_4
save mdl_Temp_5
save mdl_Temp_6

%% Oxygen training and prediction
mdl_Oxy_1=fitrsvm(Training_x,FirstPart(:,17),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Oxy_1_pred=predict(mdl_Oxy_1,Testing_x);
OError= SecondPart(:,17)-Oxy_1_pred;
mse_O1=mse(OError);
mape_O1=mean(abs(OError)./SecondPart(:,17));
%% 
save mdl_Oxy_1
%% %% Oxygen training and prediction
mdl_Oxy_2=fitrsvm(Training_x,FirstPart(:,18),'KernelFunction','rbf','Standardize',true,'OptimizeHyperparameters',{'BoxConstraint','Epsilon', 'KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
Oxy_2_pred=predict(mdl_Oxy_2,Testing_x);
OError_2= SecondPart(:,18)-Oxy_2_pred;
mse_O_2=mse(OError_2);
mape_O2=mean(abs(OError_2)./SecondPart(:,18));
%% 
save mdl_Oxy_2
plot(SecondPart(:,18),Oxy_2_pred,'*');
%% 
plot(SecondPart(:,17),Oxy_1_pred,'*');
pd_O1=fitdist(Oxy_1_pred,'Normal');
pd_actual_O1=fitdist(SecondPart(:,17),'Normal');
histfit(SecondPart(:,17))
hold on
histfit(Oxy_1_pred)
hold off
legend('actual oxygen content','Predicted oxygen content');

%% 
plot(1:3807,SecondPart(:,17),'r-')
hold on
plot(1:3807,Oxy_1_pred,'b-')
hold off
%% 
plot(1:3807,SecondPart(:,18),'r-')
hold on
plot(1:3807,Oxy_2_pred,'b-')
hold off
%% 
pd_O1=fitdist(Oxy_2_pred,'Normal');
pd_actual_O1=fitdist(SecondPart(:,18),'Normal');
histfit(Oxy_2_pred);
hold on
histfit(SecondPart(:,18));
hold off
legend('Predicted oxygen content','actual oxygen content');
%% 
LNSVR_Temp_1=fitrsvm(Training_x,FirstPart(:,1),'KernelFunction','linear','Standardize',true,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','KFold',10,'ShowPlots',0));
LNTemp_1_pred=predict(LNSVR_Temp_1,Testing_x);
LN_TError=SecondPart(:,1)-LNTemp_1_pred;
MSE_LN=mse(LN_TError);
%% 














