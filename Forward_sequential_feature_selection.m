rng(45)
FP=load("FirstPart.txt");
SP=load("SecondPart.txt");
DATA=[FP;SP];
n=size(DATA,1);
rdp=randperm(n);
Data=DATA(rdp,:);
tr=round(0.7*n);
ts=round(0.15*n);
cv=round(0.15*n);
X_tr=Data(1:tr,[14:16,19:end]);
X_cv=Data(tr+1:tr+cv,[14:16,19:end]);
X_ts=Data(tr+cv+1:end,[14:16,19:end]);
X_trn=X_tr(:,[1:7,10:end]);
X_cvn=X_cv(:,[1:7,10:end]);
X_tst=X_ts(:,[1:7,10:end]);
Oxy_trn=zeros(tr,2);
Oxy_cvn=zeros(cv,2);
Oxy_tst=zeros(ts,2);
for i=1:2
    Oxy_trn(:,i)=Data(1:tr,16+i);
    Oxy_cvn(:,i)=Data(tr+1:tr+cv,16+i);
    Oxy_tst(:,i)=Data(tr+cv+1:end,16+i);
end
Temp_trn=zeros(tr,6);
Temp_cvn=zeros(cv,6);
Temp_tst=zeros(ts,6);
for i=1:6
    Temp_trn(:,i)=Data(1:tr,i);
    Temp_cvn(:,i)=Data(tr+1:tr+cv,i);
    Temp_tst(:,i)=Data(tr+cv+1:end,i);
end
%% %% Feature selection for Oxygen left
Y_trn=Oxy_trn(:,1);
Y_cvn=Oxy_cvn(:,1);
opt = statset('display','iter','UseParallel',true);

inmodelO_1 = sequentialfs(@SVRFunc_Oxy,X_trn,Y_trn,...
                       'options',opt,...
                       'direction','forward');
 %% 
X_train_fsO1=X_trn(:,[1 2 3 4 5 7 8 9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
Y_train=Oxy_trn(:,1);
X_test_fsO1=X_tst(:,[1 2 3 4 5 7 8 9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
Y_test=Oxy_tst(:,1);
mdl_rbf=fitrsvm(X_train_fsO1,Y_train,'KernelFunction','rbf',...
        'KernelScale',14.54,'BoxConstraint',153,'Epsilon',0.0144,'Standardize',true); 
Oxy_pred=predict(mdl_rbf,X_test_fsO1);
OError=abs(Oxy_pred-Y_test);
MAPEO1=mean(OError./Y_test);
MSEO1=mse(OError);
save MSEO1
save MAPEO1
%%  Feature selection for Oxygen right
Y_trn=Oxy_trn(:,2);
Y_cvn=Oxy_cvn(:,2);
opt = statset('display','iter','UseParallel',true);

inmodelO_2 = sequentialfs(@SVRFunc_Oxy,X_trn,Y_trn,...
                       'options',opt,...
                       'direction','forward');
save inmodelO_2
%% 
X_train_fsO2=X_trn;
Y_train=Oxy_trn(:,2);
X_test_fsO2=X_tst;
Y_test=Oxy_tst(:,2);
mdl_rbf=fitrsvm(X_train_fsO2,Y_train,'KernelFunction','rbf',...
        'KernelScale',7.6125,'BoxConstraint',10.5,'Epsilon',0.0019,'Standardize',true); 
Oxy_pred=predict(mdl_rbf,X_test_fsO2);
OError=abs(Oxy_pred-Y_test);
MAPEO2=mean(OError./Y_test);
MSEO2=mse(OError);
save MSEO2
save MAPEO2
 %% Feature selection for zone 5 temperature
Y_train=Temp_trn(:,5);
Y_test=Temp_cvn(:,5);
X_train=X_trn;
X_test=X_cvn;
opt = statset('display','iter','UseParallel',true);

inmodel5 = sequentialfs(@SVRFunc,X_train,Y_train,...
                       'options',opt,...
                       'direction','forward');
 
  %%  %% Feature selection for zone 1 temperature
Y_train=Temp_trn(:,1);
Y_test=Temp_cvn(:,1);
X_train=X_trn;
X_test=X_cvn;
opt = statset('display','iter','UseParallel',true);

inmodel1 = sequentialfs(@SVRFunc,X_train,Y_train,...
                       'options',opt,...
                       'direction','forward');     
                   
                   %%  zone 5 prediction
X_train_fs5=X_trn(:,[1 4 17 18 19 20 22 24 25 26 27 28 29 30]);
Y_train=Temp_trn(:,5);
X_test_fs5=X_tst(:,[1 4 17 18 19 20 22 24 25 26 27 28 29 30]);
Y_test=Temp_tst(:,5);
[MSE,MAPE,Temp_pred]=FuncT(X_train_fs5,Y_train,X_test_fs5,Y_test);
MSE5=MSE; 
MAPE5=MAPE;
Temp_pred5=Temp_pred;
plot(1:1697,Temp_pred5);
hold on
plot(1:1697,Y_test)
hold off
xlabel('Time Stamp')
ylabel('Temperature')
legend('Predicted','Actual')
title('Zone 5 temperature profile')
grid on
figure;
scatter(Temp_pred5,Y_test)
hold on
plot(Temp_pred5,Temp_pred5)
hold off
xlabel('Predicted Temperature (^{\circ}C)')
ylabel('Actual Temperature (^{\circ}C)')
  %% zone 1 prediction
X_train_fs1=X_trn(:,[1 2 3 4 5 14 15 16 17 18 20 22 25 26 27 28 29 30 31]);
Y_train=Temp_trn(:,1);
X_test_fs1=X_tst(:,[1 2 3 4 5 14 15 16 17 18 20 22 25 26 27 28 29 30 31]);
Y_test=Temp_tst(:,1);
[MSE,MAPE,Temp_pred]=FuncT(X_train_fs1,Y_train,X_test_fs1,Y_test);
MSE1=MSE; 
MAPE1=MAPE;
Temp_pred1=Temp_pred;
plot(1:1697,Temp_pred1);
hold on
plot(1:1697,Y_test)
hold off
xlabel('Time Stamp')
ylabel('Temperature')
legend('Predicted','Actual')
title('Zone 1 temperature profile')
grid on
figure;
scatter(Temp_pred1,Y_test)
hold on
plot(Temp_pred1,Temp_pred1)
hold off
xlabel('Predicted Temperature (^{\circ}C)')
ylabel('Actual Temperature (^{\circ}C)')      
%% Feature selection for zone 2 temperature
Y_train=Temp_trn(:,2);
Y_test=Temp_cvn(:,2);
X_train=X_trn;
X_test=X_cvn;
opt = statset('display','iter','UseParallel',true);
inmodel2 = sequentialfs(@SVRFunc,X_train,Y_train,...
                       'options',opt,...
                       'direction','forward');   
                   %%  %% zone 2 prediction
X_train_fs2=X_trn(:,[1 3 4 14 16 18 20 21 22 25 26 28 29 30 31]);
Y_train=Temp_trn(:,2);
X_test_fs2=X_tst(:,[1 3 4 14 16 18 20 21 22 25 26 28 29 30 31]);
Y_test=Temp_tst(:,2);
[MSE,MAPE,Temp_pred]=FuncT(X_train_fs2,Y_train,X_test_fs2,Y_test);
MSE2=MSE; 
MAPE2=MAPE;
Temp_pred2=Temp_pred;
plot(1:1697,Temp_pred);
hold on
plot(1:1697,Y_test)
hold off
xlabel('Time Stamp')
ylabel('Temperature')
legend('Predicted','Actual')
title('Zone 2 temperature profile')
grid on
figure;
scatter(Temp_pred2,Y_test)
hold on
plot(Temp_pred2,Temp_pred2)
hold off
xlabel('Predicted Temperature (^{\circ}C)')
ylabel('Actual Temperature (^{\circ}C)')
%% %% Feature selection for zone 3 temperature
Y_train=Temp_trn(:,3);
Y_test=Temp_cvn(:,3);
X_train=X_trn;
X_test=X_cvn;
opt = statset('display','iter','UseParallel',true);

inmodel3 = sequentialfs(@SVRFunc,X_train,Y_train,...
                       'options',opt,...
                       'direction','forward');

                   %%  %%  %% zone 3 prediction
X_train_fs3=X_trn(:,[1 2 3 4 5 7 8 9 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
Y_train=Temp_trn(:,3);
X_test_fs3=X_tst(:,[1 2 3 4 5 7 8 9 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
Y_test=Temp_tst(:,3);
[MSE,MAPE,Temp_pred]=FuncT(X_train_fs3,Y_train,X_test_fs3,Y_test);
MSE3=MSE; 
MAPE3=MAPE;
Temp_pred3=Temp_pred;
plot(1:1697,Temp_pred3);
hold on
plot(1:1697,Y_test)
hold off
xlabel('Time Stamp')
ylabel('Temperature')
legend('Predicted','Actual')
title('Zone 3 temperature profile')
grid on
figure;
scatter(Temp_pred3,Y_test)
hold on
plot(Temp_pred3,Temp_pred3)
hold off
xlabel('Predicted Temperature (^{\circ}C)')
ylabel('Actual Temperature (^{\circ}C)')
%% Feature selection for zone 4 temperature
Y_train=Temp_trn(:,4);
Y_test=Temp_cvn(:,4);
X_train=X_trn;
X_test=X_cvn;
opt = statset('display','iter','UseParallel',true);

inmodel4 = sequentialfs(@SVRFunc,X_train,Y_train,...
                       'options',opt,...
                       'direction','forward');

%%  %% zone 4 prediction
X_train_fs4=X_trn(:,[1 2 3 4 5 6 8 9 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
Y_train=Temp_trn(:,4);
X_test_fs4=X_tst(:,[1 2 3 4 5 6 8 9 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]);
Y_test=Temp_tst(:,4);
[MSE,MAPE,Temp_pred]=FuncT(X_train_fs4,Y_train,X_test_fs4,Y_test);
MSE4=MSE; 
MAPE4=MAPE;
Temp_pred4=Temp_pred;
plot(1:1697,Temp_pred4);
hold on
plot(1:1697,Y_test)
hold off
xlabel('Time Stamp')
ylabel('Temperature')
legend('Predicted','Actual')
title('Zone 4 temperature profile')
grid on
figure;
scatter(Temp_pred4,Y_test)
hold on
plot(Temp_pred4,Temp_pred4)
hold off
xlabel('Predicted Temperature (^{\circ}C)')
ylabel('Actual Temperature (^{\circ}C)')
%% Feature selection for zone 6 temperature
Y_train=Temp_trn(:,6);
Y_test=Temp_cvn(:,6);
X_train=X_trn;
X_test=X_cvn;
opt = statset('display','iter','UseParallel',true);

inmodel6 = sequentialfs(@SVRFunc,X_train,Y_train,...
                       'options',opt,...
                       'direction','forward');
                   %%  %% zone 6 prediction
X_train_fs6=X_trn(:,[1 2 4 14 16 17 19 20 21 22 23 25 26 27 29 30 31]);
Y_train=Temp_trn(:,6);
X_test_fs6=X_tst(:,[1 2 4 14 16 17 19 20 21 22 23 25 26 27 29 30 31]);
Y_test=Temp_tst(:,6);
[MSE,MAPE,Temp_pred]=FuncT(X_train_fs6,Y_train,X_test_fs6,Y_test);
MSE6=MSE; 
MAPE6=MAPE;
Temp_pred6=Temp_pred;
plot(1:1697,Temp_pred6);
hold on
plot(1:1697,Y_test)
hold off
xlabel('Time Stamp')
ylabel('Temperature')
legend('Predicted','Actual')
title('Zone 6 temperature profile')
grid on
figure;
scatter(Temp_pred6,Y_test)
hold on
plot(Temp_pred6,Temp_pred6)
hold off
xlabel('Predicted Temperature (^{\circ}C)')
ylabel('Actual Temperature (^{\circ}C)')        
%% 
save MSE1
save MSE2
save MSE3
save MSE4
save MSE5
save MSE6
save MAPE1
save MAPE2
save MAPE3
save MAPE4
save MAPE5
save MAPE6
                   
                 


