%regress out task from ERG task data

%% regress out individual regressors from atlased tcs
cd('/Users/gpreti/Documents/Data/HCP_ERG');
load('/Users/gpreti/Documents/Data/HCP_ERG/HCP_Paradigms_updated');


%%
dataset=100;
parc=[{'Glasser360'}; {'Schaefer100'}; {'Schaefer200'}; {'Schaefer400'}; {'Schaefer800'}];
parcellation=1;%choose between 1 and 5
task=[{'rfMRI_REST1'}; {'rfMRI_REST2'}; {'tfMRI_EMOTION'}; {'tfMRI_GAMBLING'}; {'tfMRI_LANGUAGE'}; {'tfMRI_MOTOR'}; {'tfMRI_RELATIONAL'}; {'tfMRI_SOCIAL'}; {'tfMRI_WM'}];
subcortical=1;
excludeDC=0;


%% read tcs files and build regressors
enc=['LR';'RL'];
data_folder='/Volumes/Data2/Enrico_Projects/HCP_MIPLAB/HCP_data/';
d=dir(data_folder);
for s=3:102 % subjects
    for t=3:9 %tasks
        for e=1:2 %enc direction
            PathToPar=strcat('/Volumes/Data2/Enrico_Projects/HCP_MIPLAB/HCP_data/',d(s).name,'/MNINonLinear/Results/',task{t},'_',enc(e,:),'/EVs');
            [time,Regressor] = GSP_Paradigm(PathToPar,0,task{t});
            save(strcat('/Users/gpreti/Documents/Data/GSP/results_ERG/TaskParadigms/',d(s).name,'_Regressor_',task{t},'_',enc(e,:),'.mat'),'Regressor');
        end
    end
end


%% build glm to regress out task and save residual tcs
hrf=spm_hrf(0.72);
data_folder='/Volumes/Data2/Enrico_Projects/HCP_MIPLAB/HCP_data/';
d=dir(data_folder);
clear Beta res paradigms

for t=3:9 %tasks
    for e=1:2 %enc direction
        for s=1:100 % subjects
            
            %load fMRI tcs
            X=load(strcat('/Users/gpreti/Documents/Data/HCP_ERG/X_',task{t},'_',enc(e,:),'_',parc{parcellation},'.mat'));
            X=X.X;
            sizetask=size(X,2);
            paradigm=load(strcat('/Users/gpreti/Documents/Data/GSP/results_ERG/TaskParadigms/',d(s+2).name,'_Regressor_',task{t},'_',enc(e,:),'.mat'));
            paradigm=paradigm.Regressor;
            
            %option 1: separate regressor for each condition
            num_cond=max(paradigm);
            clear paradigms
            for i=1:num_cond
                condition=paradigm==i;
                condition=conv(condition,hrf');
                condition=condition(1:sizetask);
                paradigms(:,i)=condition';
            end
            
            %%option 2:to use binarized task regressor (1 for all
            %             %conditions)
            %             paradigm=conv((paradigm>0)',hrf');%load paradigm, binarize it and convolve it with hrf
            %             paradigms=paradigm(1:sizetask);
            
            
            %%option 3
            %separate regressor for each trial
%             TF=diff(paradigm>0);%ischange(paradigm);
%             Trials=find(TF==1)+1;
%             End_Trials=find(TF==-1)+1;
%             if paradigm(1)>0
%                 Trials=[1,Trials];
%             end
%             if paradigm(end)>0
%             End_Trials=[End_Trials,size(paradigm,2)];
%             end
%             for i=1:size(Trials,2)
%                 trial=zeros(size(paradigm,2),1);
%                 trial(Trials(i):End_Trials(i),1)=1;
%                 trial=conv(trial,hrf');
%                 trial=trial(1:sizetask);
%                 paradigms{t,e,s}(:,i)=trial;  
%             end
%             
           
            for v1=1:size(X,1)% for each region
                [Beta{s}(:,v1),res(v1,:,s),SSE,SSR,T] = y_regress_ss(X(v1,:,s)',paradigms{t,e,s});
            end
        end
        %save(strcat('/Users/gpreti/Documents/Data/GSP/results_ERG/regressout_task/separate_trials/Xres_',task{t},'_',enc(e,:),'_',parc{parcellation},'.mat'),'res');
        %save(strcat('/Users/gpreti/Documents/Data/GSP/results_ERG/regressout_task/separate_trials/Beta_',task{t},'_',enc(e,:),'_',parc{parcellation},'.mat'),'Beta');
        clear Beta res
    end
end

%% load task paradigms to visually check variability across subjects
length_task=[1200;1200;176;253;316;284;232;274;405];


for t=3:9 %tasks
    clear Pars
    for e=1 %enc direction
        for s=1:100 % subjects
            
            
            paradigm=load(strcat('/Users/gpreti/Documents/Data/GSP/results_ERG/TaskParadigms/',d(s+2).name,'_Regressor_',task{t},'_',enc(e,:),'.mat'));
            paradigm=paradigm.Regressor;
            paradigm=conv((paradigm>0)',hrf');%load paradigm, binarize it and convolve it with hrf
            paradigm=paradigm(1:length_task(t));
            Pars(:,s)=paradigm;
            
        end
    end
    figure;plot(Pars)
    title(task{t})
end


%% BETAS
%% check betas of the regression
cd('/Users/gpreti/Documents/Data/GSP/results_ERG/regressout_task/separate_trials');
betas=dir('Beta*');
clear mean_beta betamaps
for i=1:size(betas,1)
    beta=load(betas(i).name);
    beta=beta.Beta;
    for j=1:100
        sizes(j)=size(beta{j},1);
    end
    num_trials=min(sizes);
    for j=1:100
        betamaps{i}(:,:,j)=beta{j}(1:num_trials,:);
    end
    mean_beta{i}=mean(betamaps{i},3);
end

for i=1:size(mean_beta,2)
    for j=1:size(mean_beta{i},1)
        CC2=mean_beta{i}(j,:);saturate=0;PlotGraph;
        %CC2=betamaps{i}(j,:,1);saturate=0;PlotGraph;%one subject, for
        %Language
    end
end

%% for separate conditions
cd('/Users/gpreti/Documents/Data/GSP/results_ERG/regressout_task/separate_conditions');
betas=dir('Beta*');
for i=1:size(betas,1)
    beta=load(betas(i).name);
    beta=beta.Beta;
    num_cond=size(beta,1);
    for j=1:num_cond
        betamaps{i}(:,:,j)=squeeze(beta(j,:,:));
       
    end
end

for i=1:14
    means=squeeze(mean(betamaps{i},2));
    mean_betas{i}=squeeze(mean(betamaps{i},2));
    %visualize as graphs
    for j=1:size(mean_betas{i},2)
        CC2=mean_betas{i}(:,j);saturate=0;PlotGraph;
      
    end
end

%% try LDA classification of tasks using beta maps as features


%% classify task

Betas=[];
for i=1:14
    betamap=betamaps{i};
    for j=1:size(betamap,3)
        Betas=[Betas,betamaps{i}(:,:,j)];
    end
end

GROUP_BETA=zeros(size(Betas,2),1);
count=1;
for t=1:7%task
    if t==5%motor - 5 conditions
        GROUP_BETA(count:count+999,1)=t;
        count=count+1000;
    elseif t==7 %wm - 8 cconditions
        GROUP_BETA(count:count+1599,1)=t;
    else
        GROUP_BETA(count:count+399,1)=t;
        count=count+400;
    end
end

% 1) LOOCV
Mdl=fitcdiscr(Betas',GROUP_BETA);
cvmodel=crossval(Mdl,'leaveout','on'); %default is 10fold. leaveout option is just kfold with 1600 folds
L = kfoldLoss(cvmodel);
Acc1=1-L %%

% 2) LOSOCV
clear pred_class num_errors
for i=1:100 % for each subject
    test_index=[];
    for n=1:46 %test all sequences for that subject (23 conditions x2)
        test_index=[test_index,i+((n-1)*100)];
    end
    train_index=setdiff(1:4600,test_index);
    train=Betas(:,train_index)';
    group_train=GROUP_BETA(train_index);
    test=Betas(:,test_index)';
    Mdlcv=fitcdiscr(train,group_train);
    pred_class(:,i)=predict(Mdlcv,test);
    num_errors(i)=size(nonzeros(GROUP_BETA(test_index)-pred_class(:,i)),1);
end

Acc2=1-(sum(num_errors)./4600) %9164

%% classify condition
GROUP_COND=zeros(size(Betas,2),1);
count=1;
cond=1;
for t=1:7
    if t==5%motor - 5 conditions
        num_c=5;
        for j=1:num_c
            GROUP_COND(count+100*(j-1):count+99+100*(j-1),1)=cond;
            GROUP_COND(count+100*(j-1)+(num_c*100):count+100*(j-1)+(num_c*100+99),1)=cond;
            cond=cond+1;
        end
        count=count+1000;
    elseif t==7 %wm - 8 conditions
        num_c=8;
        for j=1:num_c
            GROUP_COND(count+100*(j-1):count+99+100*(j-1),1)=cond;
            GROUP_COND(count+100*(j-1)+(num_c*100):count+100*(j-1)+(num_c*100+99),1)=cond;
            cond=cond+1;
        end
    else
        GROUP_COND(count:count+99,1)=cond;
        GROUP_COND(count+200:count+299,1)=cond;
        cond=cond+1;
        GROUP_COND(count+100:count+199,1)=cond;
        GROUP_COND(count+300:count+399,1)=cond;
        cond=cond+1;
        count=count+400;
    end
end

% 2) LOSOCV
clear pred_class num_errors
for i=1:100 % for each subject
    test_index=[];
    for n=1:46 %test all sequences for that subject (23 conditions x2)
        test_index=[test_index,i+((n-1)*100)];
    end
    train_index=setdiff(1:4600,test_index);
    train=Betas(:,train_index)';
    group_train=GROUP_COND(train_index);
    test=Betas(:,test_index)';
    Mdlcv=fitcdiscr(train,group_train);
    pred_class(:,i)=predict(Mdlcv,test);
    num_errors(i)=size(nonzeros(GROUP_COND(test_index)-pred_class(:,i)),1);
end

Acc2=1-(sum(num_errors)./4600) %