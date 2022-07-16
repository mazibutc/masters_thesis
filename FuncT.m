function [MSE,MAPE,Temp_pred] = FuncT(X_train,Y_train,X_test,Y_test)

    %mdl_lin=fitrsvm(X_train,Y_train,'Standardize',true);
    mdl_rbf=fitrsvm(X_train,Y_train,'KernelFunction','rbf',...
        'KernelScale',5.092,'BoxConstraint',232.69,'Epsilon',0.948,'Standardize',true); % 296.84 for zone two,689.72 zone 3, 99.86 zn4, 952.4 zn5,25.78&eps=21.38
    Temp_pred=predict(mdl_rbf,X_test);
    TError=Temp_pred-Y_test;
    MSE=mse(TError); 
    MAPE=mean(abs(TError)./Y_test)*100;
end