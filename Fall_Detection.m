clc;
clear;
close all;
load('FallAllD.mat');

%Waist
WaistData=strcmp({FallAllD.Device},'Waist');
WaistData=WaistData';
WaistIndex=find(WaistData==1);

for i=1:length(WaistIndex)
    eval(['WaistDataAcc_',num2str(i),'=FallAllD(WaistIndex(i,1)).Acc;'])
    eval(['WaistDataGyr_',num2str(i),'=FallAllD(WaistIndex(i,1)).Gyr;'])
    WaistDataActivityGT(i,1)=FallAllD(WaistIndex(i,1)).AtivityID;
    if WaistDataActivityGT(i,1)<100
        WaistDataActivityGT(i,2)=0;
    else WaistDataActivityGT(i,2)=1;
    end
    WaistDataActivityGT=double(WaistDataActivityGT);
    eval(['WaistData_',num2str(i),'=[WaistDataAcc_',num2str(i),',WaistDataGyr_',num2str(i),'];'])
end

%Neck
NeckData=strcmp({FallAllD.Device},'Neck');
NeckData=NeckData';
NeckIndex=find(NeckData==1);

for i=1:length(NeckIndex)
    eval(['NeckDataAcc_',num2str(i),'=FallAllD(NeckIndex(i,1)).Acc;'])
    eval(['NeckDataGyr_',num2str(i),'=FallAllD(NeckIndex(i,1)).Gyr;'])
    NeckDataActivityGT(i,1)=FallAllD(NeckIndex(i,1)).AtivityID;
    if NeckDataActivityGT(i,1)<100
        NeckDataActivityGT(i,2)=0;
    else NeckDataActivityGT(i,2)=1;
    end
    NeckDataActivityGT=double(NeckDataActivityGT);
    eval(['NeckData_',num2str(i),'=[NeckDataAcc_',num2str(i),',NeckDataGyr_',num2str(i),'];'])
end

%Wrist
WristData=strcmp({FallAllD.Device},'Wrist');
WristData=WristData';
WristIndex=find(WristData==1);

for i=1:length(WristIndex)
    eval(['WristDataAcc_',num2str(i),'=FallAllD(WristIndex(i,1)).Acc;'])
    eval(['WristDataGyr_',num2str(i),'=FallAllD(WristIndex(i,1)).Gyr;'])
    WristDataActivityGT(i,1)=FallAllD(WristIndex(i,1)).AtivityID;
    if WristDataActivityGT(i,1)<100
        WristDataActivityGT(i,2)=0;
    else WristDataActivityGT(i,2)=1;
    end
    WristDataActivityGT=double(WristDataActivityGT);
    eval(['WristData_',num2str(i),'=[WristDataAcc_',num2str(i),',WristDataGyr_',num2str(i),'];'])
end

%feature table
for s=1:length(WaistIndex)
    for i=0:5
        eval(['FT_Waist(s,1+i*8)=mean(WaistData_',num2str(s),'(:,i+1));'])
        eval(['FT_Waist(s,2+i*8)=std(WaistData_',num2str(s),'(:,i+1));'])
        eval(['FT_Waist(s,3+i*8)=var(WaistData_',num2str(s),'(:,i+1));'])
        eval(['FT_Waist(s,4+i*8)=max(WaistData_',num2str(s),'(:,i+1));'])
        eval(['FT_Waist(s,5+i*8)=min(WaistData_',num2str(s),'(:,i+1));'])
        eval(['FT_Waist(s,6+i*8)=range(WaistData_',num2str(s),'(:,i+1));'])
        eval(['FT_Waist(s,7+i*8)=kurtosis(WaistData_',num2str(s),'(:,i+1));'])
        eval(['FT_Waist(s,8+i*8)=skewness(WaistData_',num2str(s),'(:,i+1));'])
    end
end
FT_GT_Waist=[FT_Waist,WaistDataActivityGT(:,2)];

for s=1:length(NeckIndex)
    for i=0:5
        eval(['FT_Neck(s,1+i*8)=mean(NeckData_',num2str(s),'(:,i+1));'])
        eval(['FT_Neck(s,2+i*8)=std(NeckData_',num2str(s),'(:,i+1));'])
        eval(['FT_Neck(s,3+i*8)=var(NeckData_',num2str(s),'(:,i+1));'])
        eval(['FT_Neck(s,4+i*8)=max(NeckData_',num2str(s),'(:,i+1));'])
        eval(['FT_Neck(s,5+i*8)=min(NeckData_',num2str(s),'(:,i+1));'])
        eval(['FT_Neck(s,6+i*8)=range(NeckData_',num2str(s),'(:,i+1));'])
        eval(['FT_Neck(s,7+i*8)=kurtosis(NeckData_',num2str(s),'(:,i+1));'])
        eval(['FT_Neck(s,8+i*8)=skewness(NeckData_',num2str(s),'(:,i+1));'])
    end
end
FT_GT_Neck=[FT_Neck,NeckDataActivityGT(:,2)];

for s=1:length(WristIndex)
    for i=0:5
        eval(['FT_Wrist(s,1+i*8)=mean(WristData_',num2str(s),'(:,i+1));'])
        eval(['FT_Wrist(s,2+i*8)=std(WristData_',num2str(s),'(:,i+1));'])
        eval(['FT_Wrist(s,3+i*8)=var(WristData_',num2str(s),'(:,i+1));'])
        eval(['FT_Wrist(s,4+i*8)=max(WristData_',num2str(s),'(:,i+1));'])
        eval(['FT_Wrist(s,5+i*8)=min(WristData_',num2str(s),'(:,i+1));'])
        eval(['FT_Wrist(s,6+i*8)=range(WristData_',num2str(s),'(:,i+1));'])
        eval(['FT_Wrist(s,7+i*8)=kurtosis(WristData_',num2str(s),'(:,i+1));'])
        eval(['FT_Wrist(s,8+i*8)=skewness(WristData_',num2str(s),'(:,i+1));'])
    end
end
FT_GT_Wrist=[FT_Wrist,WristDataActivityGT(:,2)];

%Cross Validation & Training Model
foldindex_Waist=randi([1 10],length(WaistIndex),1);
for Waistkfold=1:10
    test=(foldindex_Waist==Waistkfold);
    train=~test;
    FT_GT_train_waist=FT_GT_Waist(train,:);
    FT_GT_test_waist=FT_GT_Waist(test,:);    
    modelKNN=fitcknn(FT_GT_train_waist(:,1:48),FT_GT_train_waist(:,49),"NumNeighbors",3);
%     t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3);
    modelSVM = fitcsvm(FT_GT_train_waist(:, 1:48), FT_GT_train_waist(:,49));
    modelNB = fitcnb(FT_GT_train_waist(:, 1:48), FT_GT_train_waist(:,49));
    modelDT = fitctree(FT_GT_train_waist(:, 1:48), FT_GT_train_waist(:,49));
    predictmodelSVM=predict(modelSVM, FT_GT_test_waist(:, 1:48));
    predictmodelKNN=predict(modelKNN,FT_GT_test_waist(:,1:48));
    predictmodelNB=predict(modelNB, FT_GT_test_waist(:, 1:48));
    predictmodelDT=predict(modelDT, FT_GT_test_waist(:, 1:48));
    eval(['Waist_confusionmatrix_SVM_',num2str(Waistkfold),'=confusionmat(FT_GT_test_waist(:,49),predictmodelSVM);'])
    eval(['Waist_Matrix_1(Waistkfold,1)=sum(diag(Waist_confusionmatrix_SVM_',num2str(Waistkfold),'))/sum(sum(Waist_confusionmatrix_SVM_',num2str(Waistkfold),'));'])
    eval(['Waist_Matrix_1(Waistkfold,2)=Waist_confusionmatrix_SVM_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_SVM_', num2str(Waistkfold),'(2, :));'])
    eval(['Waist_Matrix_1(Waistkfold,3)=Waist_confusionmatrix_SVM_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_SVM_', num2str(Waistkfold),'(:, 2));'])
    eval(['Waist_confusionmatrix_KNN_',num2str(Waistkfold),'=confusionmat(FT_GT_test_waist(:,49),predictmodelKNN);'])
    eval(['Waist_Matrix_2(Waistkfold,1)=sum(diag(Waist_confusionmatrix_KNN_',num2str(Waistkfold),'))/sum(sum(Waist_confusionmatrix_KNN_',num2str(Waistkfold),'));'])
    eval(['Waist_Matrix_2(Waistkfold,2)=Waist_confusionmatrix_KNN_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_KNN_', num2str(Waistkfold),'(2, :));'])
    eval(['Waist_Matrix_2(Waistkfold,3)=Waist_confusionmatrix_KNN_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_KNN_', num2str(Waistkfold),'(:, 2));'])
    eval(['Waist_confusionmatrix_NB_',num2str(Waistkfold),'=confusionmat(FT_GT_test_waist(:,49),predictmodelNB);'])
    eval(['Waist_Matrix_3(Waistkfold,1)=sum(diag(Waist_confusionmatrix_NB_',num2str(Waistkfold),'))/sum(sum(Waist_confusionmatrix_NB_',num2str(Waistkfold),'));'])
    eval(['Waist_Matrix_3(Waistkfold,2)=Waist_confusionmatrix_NB_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_NB_', num2str(Waistkfold),'(2, :));'])
    eval(['Waist_Matrix_3(Waistkfold,3)=Waist_confusionmatrix_NB_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_NB_', num2str(Waistkfold),'(:, 2));'])
    eval(['Waist_confusionmatrix_DT_',num2str(Waistkfold),'=confusionmat(FT_GT_test_waist(:,49),predictmodelDT);'])
    eval(['Waist_Matrix_4(Waistkfold,1)=sum(diag(Waist_confusionmatrix_DT_',num2str(Waistkfold),'))/sum(sum(Waist_confusionmatrix_DT_',num2str(Waistkfold),'));'])
    eval(['Waist_Matrix_4(Waistkfold,2)=Waist_confusionmatrix_DT_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_DT_', num2str(Waistkfold),'(2, :));'])
    eval(['Waist_Matrix_4(Waistkfold,3)=Waist_confusionmatrix_DT_', num2str(Waistkfold), '(2,2)/sum(Waist_confusionmatrix_DT_', num2str(Waistkfold),'(:, 2));'])
end

foldindex_Neck=randi([1 10],length(NeckIndex),1);
for Neckkfold=1:10
    test=(foldindex_Neck==Neckkfold);
    train=~test;
    FT_GT_train_neck=FT_GT_Neck(train,:);
    FT_GT_test_neck=FT_GT_Neck(test,:);    
    modelKNN=fitcknn(FT_GT_train_neck(:,1:48),FT_GT_train_neck(:,49),"NumNeighbors",3);
%     t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3);
    modelSVM = fitcsvm(FT_GT_train_neck(:, 1:48), FT_GT_train_neck(:,49));
    modelNB = fitcnb(FT_GT_train_neck(:, 1:48), FT_GT_train_neck(:,49));
    modelDT = fitctree(FT_GT_train_neck(:, 1:48), FT_GT_train_neck(:,49));
    predictmodelSVM=predict(modelSVM, FT_GT_test_neck(:, 1:48));
    predictmodelKNN=predict(modelKNN,FT_GT_test_neck(:,1:48));
    predictmodelNB=predict(modelNB, FT_GT_test_neck(:, 1:48));
    predictmodelDT=predict(modelDT, FT_GT_test_neck(:, 1:48));
    eval(['Neck_confusionmatrix_SVM_',num2str(Neckkfold),'=confusionmat(FT_GT_test_neck(:,49),predictmodelSVM);'])
    eval(['Neck_Matrix_1(Neckkfold,1)=sum(diag(Neck_confusionmatrix_SVM_',num2str(Neckkfold),'))/sum(sum(Neck_confusionmatrix_SVM_',num2str(Neckkfold),'));'])
    eval(['Neck_Matrix_1(Neckkfold,2)=Neck_confusionmatrix_SVM_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_SVM_', num2str(Neckkfold),'(2, :));'])
    eval(['Neck_Matrix_1(Neckkfold,3)=Neck_confusionmatrix_SVM_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_SVM_', num2str(Neckkfold),'(:, 2));'])
    eval(['Neck_confusionmatrix_KNN_',num2str(Neckkfold),'=confusionmat(FT_GT_test_neck(:,49),predictmodelKNN);'])
    eval(['Neck_Matrix_2(Neckkfold,1)=sum(diag(Neck_confusionmatrix_KNN_',num2str(Neckkfold),'))/sum(sum(Neck_confusionmatrix_KNN_',num2str(Neckkfold),'));'])
    eval(['Neck_Matrix_2(Neckkfold,2)=Neck_confusionmatrix_KNN_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_KNN_', num2str(Neckkfold),'(2, :));'])
    eval(['Neck_Matrix_2(Neckkfold,3)=Neck_confusionmatrix_KNN_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_KNN_', num2str(Neckkfold),'(:, 2));'])
    eval(['Neck_confusionmatrix_NB_',num2str(Neckkfold),'=confusionmat(FT_GT_test_neck(:,49),predictmodelNB);'])
    eval(['Neck_Matrix_3(Neckkfold,1)=sum(diag(Neck_confusionmatrix_NB_',num2str(Neckkfold),'))/sum(sum(Neck_confusionmatrix_NB_',num2str(Neckkfold),'));'])
    eval(['Neck_Matrix_3(Neckkfold,2)=Neck_confusionmatrix_NB_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_NB_', num2str(Neckkfold),'(2, :));'])
    eval(['Neck_Matrix_3(Neckkfold,3)=Neck_confusionmatrix_NB_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_NB_', num2str(Neckkfold),'(:, 2));'])
    eval(['Neck_confusionmatrix_DT_',num2str(Neckkfold),'=confusionmat(FT_GT_test_neck(:,49),predictmodelDT);'])
    eval(['Neck_Matrix_4(Neckkfold,1)=sum(diag(Neck_confusionmatrix_DT_',num2str(Neckkfold),'))/sum(sum(Neck_confusionmatrix_DT_',num2str(Neckkfold),'));'])
    eval(['Neck_Matrix_4(Neckkfold,2)=Neck_confusionmatrix_DT_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_DT_', num2str(Neckkfold),'(2, :));'])
    eval(['Neck_Matrix_4(Neckkfold,3)=Neck_confusionmatrix_DT_', num2str(Neckkfold), '(2,2)/sum(Neck_confusionmatrix_DT_', num2str(Neckkfold),'(:, 2));'])
end

foldindex_Wrist=randi([1 10],length(WristIndex),1);
for Wristkfold=1:10
    test=(foldindex_Wrist==Wristkfold);
    train=~test;
    FT_GT_train_wrist=FT_GT_Wrist(train,:);
    FT_GT_test_wrist=FT_GT_Wrist(test,:);    
    modelKNN=fitcknn(FT_GT_train_wrist(:,1:48),FT_GT_train_wrist(:,49),"NumNeighbors",3);
%     t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3);
    modelSVM = fitcsvm(FT_GT_train_wrist(:, 1:48), FT_GT_train_wrist(:,49));
    modelNB = fitcnb(FT_GT_train_wrist(:, 1:48), FT_GT_train_wrist(:,49));
    modelDT = fitctree(FT_GT_train_wrist(:, 1:48), FT_GT_train_wrist(:,49));
    predictmodelSVM=predict(modelSVM, FT_GT_test_wrist(:, 1:48));
    predictmodelKNN=predict(modelKNN,FT_GT_test_wrist(:,1:48));
    predictmodelNB=predict(modelNB, FT_GT_test_wrist(:, 1:48));
    predictmodelDT=predict(modelDT, FT_GT_test_wrist(:, 1:48));
    eval(['Wrist_confusionmatrix_SVM_',num2str(Wristkfold),'=confusionmat(FT_GT_test_wrist(:,49),predictmodelSVM);'])
    eval(['Wrist_Matrix_1(Wristkfold,1)=sum(diag(Wrist_confusionmatrix_SVM_',num2str(Wristkfold),'))/sum(sum(Wrist_confusionmatrix_SVM_',num2str(Wristkfold),'));'])
    eval(['Wrist_Matrix_1(Wristkfold,2)=Wrist_confusionmatrix_SVM_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_SVM_', num2str(Wristkfold),'(2, :));'])
    eval(['Wrist_Matrix_1(Wristkfold,3)=Wrist_confusionmatrix_SVM_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_SVM_', num2str(Wristkfold),'(:, 2));'])
    eval(['Wrist_confusionmatrix_KNN_',num2str(Wristkfold),'=confusionmat(FT_GT_test_wrist(:,49),predictmodelKNN);'])
    eval(['Wrist_Matrix_2(Wristkfold,1)=sum(diag(Wrist_confusionmatrix_KNN_',num2str(Wristkfold),'))/sum(sum(Wrist_confusionmatrix_KNN_',num2str(Wristkfold),'));'])
    eval(['Wrist_Matrix_2(Wristkfold,2)=Wrist_confusionmatrix_KNN_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_KNN_', num2str(Wristkfold),'(2, :));'])
    eval(['Wrist_Matrix_2(Wristkfold,3)=Wrist_confusionmatrix_KNN_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_KNN_', num2str(Wristkfold),'(:, 2));'])
    eval(['Wrist_confusionmatrix_NB_',num2str(Wristkfold),'=confusionmat(FT_GT_test_wrist(:,49),predictmodelNB);'])
    eval(['Wrist_Matrix_3(Wristkfold,1)=sum(diag(Wrist_confusionmatrix_NB_',num2str(Wristkfold),'))/sum(sum(Wrist_confusionmatrix_NB_',num2str(Wristkfold),'));'])
    eval(['Wrist_Matrix_3(Wristkfold,2)=Wrist_confusionmatrix_NB_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_NB_', num2str(Wristkfold),'(2, :));'])
    eval(['Wrist_Matrix_3(Wristkfold,3)=Wrist_confusionmatrix_NB_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_NB_', num2str(Wristkfold),'(:, 2));'])
    eval(['Wrist_confusionmatrix_DT_',num2str(Wristkfold),'=confusionmat(FT_GT_test_wrist(:,49),predictmodelDT);'])
    eval(['Wrist_Matrix_4(Wristkfold,1)=sum(diag(Wrist_confusionmatrix_DT_',num2str(Wristkfold),'))/sum(sum(Wrist_confusionmatrix_DT_',num2str(Wristkfold),'));'])
    eval(['Wrist_Matrix_4(Wristkfold,2)=Wrist_confusionmatrix_DT_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_DT_', num2str(Wristkfold),'(2, :));'])
    eval(['Wrist_Matrix_4(Wristkfold,3)=Wrist_confusionmatrix_DT_', num2str(Wristkfold), '(2,2)/sum(Wrist_confusionmatrix_DT_', num2str(Wristkfold),'(:, 2));'])
end
for i=1:4
    for s=1:3
        eval(['testAveragePerformance_1(i,s)=mean(Waist_Matrix_',num2str(i),'(:,s));'])
        eval(['testAveragePerformance_2(i,s)=mean(Neck_Matrix_',num2str(i),'(:,s));'])
        eval(['testAveragePerformance_3(i,s)=mean(Wrist_Matrix_',num2str(i),'(:,s));'])
    end
end

