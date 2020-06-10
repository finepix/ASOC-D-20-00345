% ELM: session 1 (labeled) --> session 2 (unlabeled)

clear; clc; close all;

% prepare the labeled and unlabeled data
load('fea_session2_subject2.mat');
Xl = fea'; %Xl = NormalizeFea(Xl); 
Xl = Xl./max(Xl(:));
clear fea

load('gnd_session2.mat');
Yl = gnd;
clear gnd

load('fea_session3_subject2.mat');
Xu = fea'; %Xu = NormalizeFea(Xu);
Xu = Xu./max(Xu(:));
clear fea


load('gnd_session3.mat')
Yu = gnd;
clear gnd

[n_labeled, nDim] = size(Xl);
[n_unlabeled,~] = size(Xu);
nClass = length(unique(Yl));

% construct the graph on 'L' and 'U'
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'Binary';
options.t = 1;
S = constructW([Xl; Xu],options);
S = full(S);
D = diag(sum(S));
LapMatrix = D - S;
S_u = S(n_labeled+1:n_labeled+n_unlabeled,n_labeled+1:n_labeled+n_unlabeled);
D_u = diag(sum(S_u));

% generate the input weight and hidden bias
L = 3*nDim;
if ~exist('random_W_b.mat','file')
    input_weight = rand(L, nDim) * 2 - 1;
    hbias = rand(L, 1)*2-1;
    save('random_W_b.mat','input_weight','hbias');
else
    load('random_W_b.mat');
end

H_labeled = 1./(1+exp(-Xl*input_weight' + repmat(hbias',[n_labeled,1])));
H_unlabeled = 1./(1+exp(-Xu*input_weight' + repmat(hbias',[n_unlabeled,1])));

H_labeled = H_labeled - repmat(mean(H_labeled),[size(H_labeled,1),1]);
H_unlabeled = H_unlabeled - repmat(mean(H_unlabeled),[size(H_unlabeled,1),1]);

H = [H_labeled; H_unlabeled];
HTH = H'*H;
HTLH = H'*LapMatrix*H;

Y_labeled = zeros(n_labeled,nClass);
for i = 1:n_labeled
    Y_labeled(i,Yl(i)) = 1;
end

Y_unlabeled = ones(n_unlabeled,nClass)./nClass;
Y = [Y_labeled; Y_unlabeled];
HTY = H'*Y;

lambda1Lib = 10.^(-3:3);
lambda2Lib = 10.^(-3:3);
acc_ourSELM = zeros(length(lambda1Lib),length(lambda1Lib));

NITER = 30;
for i = 1:length(lambda1Lib)
    lambda1 = lambda1Lib(i);
    for j = 1:length(lambda2Lib)
        lambda2 = lambda2Lib(j);
        
        iter = 1;
        accuracy_unlabeled = zeros(NITER,1);
        while iter <= NITER
            % update 'beta'
            beta = (HTH + lambda1*eye(L) + lambda2*HTLH) \ HTY;
            
            % update 'Y_unlabeled'
            for n = 1:n_unlabeled
                temp = H_unlabeled(n,:)*beta;
                Y_unlabeled(n,:) = EProjSimplex_new(temp);
                clear temp
            end
            
            % update 'Y' and 'HTY'
            Y = [Y_labeled; Y_unlabeled];
            HTY = H'*Y;
            
            % objective value
            Hbeta = H*beta;
%             obj(iter) = norm(Hbeta-Y,'fro')^2 + lambda1*norm(beta,'fro')^2 + lambda2* trace(Hbeta'*LapMatrix*Hbeta);
            
            % update 'iter'
            iter = iter + 1;
        end
        
        % performance on unlabeled data
        [~,predict_label] = max(Y_unlabeled,[],2);
        
        predict_label_all(:,i,j) = predict_label;       
        acc_ourSELM(i,j) = length(find(predict_label == Yu)) ./  length(Yu);
        
        fprintf('lambda1=%f,lambda2=%f,accuracy on unlabeled samples is %.2f \n',lambda1,lambda2,acc_ourSELM(i,j)*100);
        clear predict_label
    end
end
max(acc_ourSELM(:))
[index_a,index_b] = find(acc_ourSELM==max(acc_ourSELM(:)));
predict_label = predict_label_all(:,index_a,index_b);
save SELM_s2_s3_subject2 predict_label