function [mse_o] = SVRFunc_Oxy(X_trn,Y_trn,X_cvn,Y_cvn)

    %mdl_lin=fitrsvm(X_train,Y_train,'Standardize',true);
    mdl_rbf=fitrsvm(X_trn,Y_trn,'KernelFunction','rbf',...
        'KernelScale',7.6125,'BoxConstraint',10.5,'Epsilon',0.0019,'Standardize',true); % 296.84 for zone two,689.72 zone 3, 99.86 zn4, 952.4 zn5,25.78&eps=21.38
    Oxy_pred=predict(mdl_rbf,X_cvn);
    O_Error=Oxy_pred-Y_cvn;
    mse_o=mse(O_Error);
end