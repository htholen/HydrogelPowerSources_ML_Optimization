%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Machine learning & Inverse Optimization Demo Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% File Information
%{
Title: Design optimization of electric-fish-inspired power sources using statistical and machine learning approaches
Journal: Journal of Power Sources
Authors: Haley M. Tholen, Ahmed S. Mohamed, Valentino I. Wilson, Derek M. Hall, and Joseph S. Najem
Last Updated: October 2025

Summary:
========
This script solves an inverse problem by finding a normalized input that`
(via a trained Gaussian Process Regression model) recovers a target metric.
It first loads CSV data, performs standard training (including GPR model training
and evaluation for all metrics), and then it runs an inverse
optimization on a portion of the training data points (as targets) for each metric. The inverse 
optimization consists of multiple fmincon trials for initialization followed by 
a simulated annealing (SA) refinement that utilizes a Metropolis acceptance criterion. 
Finally, the recovered versus target metric values for all metrics are plotted in two 
figures (9 subplots in one and 8 in the other).

NOTE: The inverse objective function is defined as the squared error between
      the recovered raw output (obtained via back-transformation of a log‑prediction)
      and the chosen target metric value.
%}

%% Machine Learning Hyperparameter Tuning

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each performance metric:
%   - Iterates over two GPR kernels
%   - Iterates over 50 SigmaLowerBound values between 0.01 and 1
%   - For the optimal sigma, iterates over three BasisFunction choices
%   - Within each kernel+basis+sigma, iterates over two RNG seeds to maximize test accuracy
%   - Prints the best seed & accuracies for each combination
%   - Selects the best kernel+basis+sigma+seed per metric
%   - Saves all best settings into a MAT file for future use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% USER SETTINGS
clear; clc; close all;
dataFolder = 'Data Files';
csvFile = fullfile(dataFolder, 'SampleDataset_MLOptimizationDemo.csv');
trainRatio = 0.80; % fraction for training
kernels = {'ardsquaredexponential','ardmatern32'};
basisFuncs = {'linear'};
sigmaLBs = (0.01:0.02:0.49)';
seeds = 1; % RNG seeds to try
epsX = 1e-6; % offset for log

% LOAD & CLEAN DATA
T = readtable(csvFile,'PreserveVariableNames',true);
vars = T.Properties.VariableNames;
splitIdx = find(all(ismissing(T),1),1);
paramHdr = vars(1:splitIdx-1);
metricHdr = vars(splitIdx+1:end);

Xraw = T{:,paramHdr};
Yraw = T{:,metricHdr};
mask = any(isnan(Yraw),2);
Xraw(mask,:) = [];
Yraw(mask,:) = [];

% TRANSFORM: LOG + Z‑SCORE
Xlog = log(Xraw + epsX);
muX = mean(Xlog,1);
sigmaX = std(Xlog,[],1) + epsX;
Xnorm = (Xlog - muX) ./ sigmaX;

Ylog = log(Yraw + epsX);
muY = mean(Ylog,1);
sigmaY = std(Ylog,[],1) + epsX;
Ynorm = (Ylog - muY) ./ sigmaY;

N = size(Xnorm,1);
J = numel(metricHdr);

% PREALLOCATE SUMMARY
BestKernel = strings(J,1);
BestBasis = strings(J,1);
BestSigma = zeros(J,1);
BestSeed = zeros(J,1);
TrainAcc = zeros(J,1);
TestAcc = zeros(J,1);
NMSE = nan(J,1); % best NMSE per metric
yPredCell = cell(J,1); % temp hold-out predictions
yActCell = cell(J,1); % temp hold-out targets
yPred = cell(J,1); % final best-model predictions
yAct = cell(J,1); % final best-model targets
NMSE_hold = nan(J,1); % optional

fprintf('=== Per-Metric Kernel+Basis+Sigma+Seed Search ===\n\n');

for j = 1:J
    fprintf('Metric %d/%d: %s\n', j, J, metricHdr{j});
    overallBestTest = -inf;
    % loop kernels
    for ci = 1:numel(kernels)
        kern = kernels{ci};
        % loop sigma lower bounds
        for svi = 1:numel(sigmaLBs)
            sLB = sigmaLBs(svi);
            % loop basis functions
            for bf_i = 1:numel(basisFuncs)
                bf = basisFuncs{bf_i};
                bestTest = -inf;
                bestTrain= NaN;
                bestSeed = NaN;
                % try each seed
                for s = seeds
                    rng(s);
                    if j==1 && ci==1 && svi==1 && bf_i==1 && s==seeds(1)
                        idx = randperm(N); % shuffle rows once
                        nTr = round(trainRatio*N);
                        te = idx(nTr+1:end); % 20%
                        TrainIdxGlobal = true(N,1);     
                        TrainIdxGlobal(te) = false; % 80%
                        TestIdxGlobal = te;
                        Xtest_raw = Xraw(te,:); % store raw inputs
                    end
                    
                    % Fixed indices for all subsequent passes
                    tr = find(TrainIdxGlobal);
                    te = TestIdxGlobal;
                    
                    mdl = fitrgp( ...
                        Xnorm(tr,:), Ynorm(tr,j), ...
                        'KernelFunction', kern, ...
                        'BasisFunction', bf, ...
                        'SigmaLowerBound', sLB, ...
                        'Standardize', false );

                    % predict & back-transform
                    yTrPred = exp(predict(mdl,Xnorm(tr,:)).*sigmaY(j) + muY(j)) - epsX;
                    yTePred = exp(predict(mdl,Xnorm(te ,:)).*sigmaY(j) + muY(j)) - epsX;
                    yTrTrue = Yraw(tr ,j);
                    yTeTrue = Yraw(te ,j);

                    nmseTr = mean((yTrPred - yTrTrue).^2)/var(yTrTrue);
                    nmseTe = mean((yTePred - yTeTrue).^2)/var(yTeTrue);
                    accTr = 100*(1 - nmseTr);
                    accTe = 100*(1 - nmseTe);

                    % stash hold-out predictions
                    if s==seeds(1) % only need one seed’s predictions per combo
                        yPredCell{j} = yTePred; % 1×K cell, each a column vector
                        yActCell {j} = yTeTrue; % "
                    end

                    % collect NMSE on the selected best model
                    NMSE_hold(j) = nmseTe;

                    trAccThreshold = 80;
                    if (accTe > bestTest) && (accTr >= trAccThreshold)
                        bestTest  = accTe;
                        bestTrain = accTr;
                        bestSeed  = s;
                    end
                end

                % report best for this combination
                fprintf('  Kern %-20s | Basis=%-8s | SigmaLB=%5.3f | TrAcc=%5.2f%% | TeAcc=%5.2f%% (seed=%d)\n', ...
                    kern, bf, sLB, bestTrain, bestTest, bestSeed);

                % update overall best for this metric
                if bestTest > overallBestTest
                    overallBestTest = bestTest;
                    BestKernel(j) = kern;
                    BestBasis(j) = bf;
                    BestSigma(j) = sLB;
                    BestSeed(j) = bestSeed;
                    TrainAcc(j) = bestTrain;
                    TestAcc(j) = bestTest;
                    NMSE(j) = 1 - bestTest/100; % convert accuracy to NMSE for plotting
                    yPred{j} = yPredCell{j};
                    yAct {j} = yActCell {j};
                end
            end
        end
    end
    fprintf('  >>> Selected %s + %s + SigmaLB=%.3f (Tr=%.2f%%, Te=%.2f%%, seed=%d)\n\n', ...
        BestKernel(j), BestBasis(j), BestSigma(j), TrainAcc(j), TestAcc(j), BestSeed(j));
end

% FINAL SUMMARY TABLE
Tsummary = table(metricHdr(:), BestKernel, BestBasis, BestSigma, BestSeed, TrainAcc, TestAcc, ...
    'VariableNames',{'Metric','BestKernel','BestBasis','BestSigmaLB','BestSeed','TrainAcc','TestAcc'});
fprintf('=== Summary of Best Settings per Metric ===\n');
disp(Tsummary);

% SAVE BEST SETTINGS FOR FUTURE USE
save('bestGPRSettingsARD.mat', 'Tsummary', 'BestKernel', 'BestBasis', ...
     'BestSigma', 'BestSeed', 'TrainAcc', 'TestAcc', 'metricHdr',       ...
     'TestIdxGlobal','TrainIdxGlobal','Xtest_raw','yPred','yAct','NMSE');

fprintf('Best GPR settings saved to bestGPRSettingsARD.mat\n');

%% Machine Learning GPR Model

% Uses previously saved best GPR settings and re-runs the prediction 
% step to verify that the summary table matches the one saved in 
% bestGPRSettingsARD.mat.

% SETUP
clear;

% LOAD BEST SETTINGS
S = load('bestGPRSettingsARD.mat', ...
         'metricHdr','BestKernel','BestBasis','BestSigma','BestSeed', ...
         'TestIdxGlobal','TrainIdxGlobal','Xtest_raw', ...
         'TrainAcc','TestAcc','yPred','yAct');
metricHdr = S.metricHdr;
BestKernel = S.BestKernel;
BestBasis = S.BestBasis;
BestSigma = S.BestSigma;
BestSeed = S.BestSeed;
TestIdxGlobal = S.TestIdxGlobal;
TrainIdxGlobal = S.TrainIdxGlobal;
TestAcc = S.TestAcc; 
yPred = S.yPred; 
yAct = S.yAct; 

% DATA FILE & TRANSFORM PARAMETERS
dataFolder = 'Data Files';
csvFile = fullfile(dataFolder, 'SampleDataset_MLOptimizationDemo.csv');

trainRatio = 0.80;
epsX = 1e-6;

% LOAD & CLEAN DATA
T = readtable(csvFile,'PreserveVariableNames',true);
vars = T.Properties.VariableNames;
splitIdx = find(all(ismissing(T),1),1);
paramHdr = vars(1:splitIdx-1);
metricHdrFull = vars(splitIdx+1:end);

Xraw = T{:,paramHdr};
Yraw = T{:,metricHdrFull};
mask = any(isnan(Yraw),2);
Xraw(mask,:) = [];
Yraw(mask,:) = [];

% TRANSFORM: LOG + Z‑SCORE
Xlog = log(Xraw + epsX);
muX = mean(Xlog,1);
sigmaX = std(Xlog,[],1) + epsX;
Xnorm = (Xlog - muX) ./ sigmaX;

Ylog = log(Yraw + epsX);
muY = mean(Ylog,1);
sigmaY = std(Ylog,[],1) + epsX;
Ynorm = (Ylog - muY) ./ sigmaY;

N = size(Xnorm,1);
J = numel(metricHdr);

% RE‑RUN WITH SAVED SETTINGS
recomputedTrainAcc = zeros(J,1);
recomputedTestAcc  = zeros(J,1);
rawImportances = zeros(J, size(Xnorm,2));
for j = 1:J
    seed = BestSeed(j);
    kern = BestKernel{j};
    bf = BestBasis{j};
    sLB = BestSigma(j);
    
    tr = find(TrainIdxGlobal); % fixed 80%
    te = find(TestIdxGlobal); % fixed 20%

    % Train with loaded settings
    mdl = fitrgp( ...
        Xnorm(tr,:), Ynorm(tr,j), ...
        'KernelFunction',  kern, ...
        'BasisFunction',   bf, ...
        'SigmaLowerBound', sLB, ...
        'Standardize',     false );

    % Extract length scales (1./lengthScale = importance)
    kernelParams = mdl.KernelInformation.KernelParameters;

    % ARD kernels have one length scale per feature
    lengthScales = kernelParams(1:size(Xnorm,2));

    rawImportances(j, :) = 1 ./ lengthScales;

    % Predict & back‑transform
    yTrPred = exp(predict(mdl,Xnorm(tr,:)).*sigmaY(j) + muY(j)) - epsX;
    yTePred = exp(predict(mdl,Xnorm(te,:)).*sigmaY(j) + muY(j)) - epsX;
    yTrTrue = Yraw(tr ,j);
    yTeTrue = Yraw(te ,j);
    
    nmseTr = mean((yTrPred - yTrTrue).^2) / var(yTrTrue);
    nmseTe = mean((yTePred - yTeTrue).^2) / var(yTeTrue);

    recomputedTrainAcc(j) = 100*(1 - nmseTr);
    recomputedTestAcc(j) = 100*(1 - nmseTe);
end

normImportances = rawImportances ./ sum(rawImportances, 2);

% DISPLAY AND COMPARE
T_saved = table(metricHdr(:), S.TrainAcc, S.TestAcc, ...
    'VariableNames',{'Metric','SavedTrainAcc','SavedTestAcc'} );

T_recomputed = table(metricHdr(:), recomputedTrainAcc, recomputedTestAcc, ...
    'VariableNames',{'Metric','RecompTrainAcc','RecompTestAcc'} );

fprintf('--- Saved vs Recomputed Accuracy ---\n');

disp([T_saved T_recomputed(:,2:3)]);

save('lengthScaleData.mat')

%% Inverse Optimization Problem

% Loads saved GPR-ARD models and performs inverse design to recover inputs
% that reproduce desired performance metrics, using:
%   - full-range bounds derived from the training data
%   - objective & targets in z-scored output space
%   - k-NN + random multi-start seeds
%   - Metropolis SA with fixed step σ

% SETUP
clear; close all; masterTic = tic;

% LOAD BEST GPR SETTINGS 
S = load('bestGPRSettingsARD.mat', ...
         'metricHdr','BestKernel','BestBasis','BestSigma','BestSeed', ...
         'TestIdxGlobal','TrainIdxGlobal','Xtest_raw');
metricHdr = S.metricHdr;
BestKernel = S.BestKernel;
BestBasis = S.BestBasis;
BestSigma = S.BestSigma;
BestSeed = S.BestSeed;
TestIdxGlobal = S.TestIdxGlobal; % 20% hold-out indices
Xtest_raw = S.Xtest_raw; % true inputs (raw units)

% DATA SETTINGS
dataFolder = 'Data Files';
csvFile = fullfile(dataFolder, 'SampleDataset_MLOptimizationDemo.csv');
trainRatio = 0.80; % train/hold-out split for k-NN seeding
epsX = 1e-6;

% LOAD & TRANSFORM DATA
T = readtable(csvFile,'PreserveVariableNames',true);
vars = T.Properties.VariableNames;
split = find(all(ismissing(T),1),1);
paramHdr = vars(1:split-1);
metricHdrFull = vars(split+1:end);

Xraw = T{:,paramHdr}; % inputs
Yraw = T{:,metricHdrFull}; % outputs
valid = ~any(isnan(Yraw),2);
Xraw = Xraw(valid,:);  Yraw = Yraw(valid,:);

% LOG-Z TRANSFORM
Xlog = log(Xraw + epsX);
muX = mean(Xlog,1);       
sigmaX = std(Xlog,[],1) + epsX;
Xnorm = (Xlog - muX) ./ sigmaX; % optimizer works in this space

Ylog = log(Yraw + epsX);
muY = mean(Ylog,1);       sigmaY = std(Ylog,[],1) + epsX;
Ynorm = (Ylog - muY) ./ sigmaY; % objective lives in this space

% NORMALIZATION RANGE INFO 
minX = min(Xlog,[],1);
rangeX = max(Xlog,[],1) - minX + epsX;

% bounds in normalized z-space for inverse optimization
lb = min(Xnorm,[],1) - 0.25;
ub = max(Xnorm,[],1) + 0.25;

% DROP UNWANTED PARAMETERS FOR OPTIMIZATION
dropList = {
  'Total Thickness (mm)'
  'Combined Parameter B (mol m-3)'
  'Combined Parameter A (mol m-3)' 
};
keepIdx = ~ismember(paramHdr, dropList);
Xraw = Xraw(:, keepIdx);
Xtest_raw = Xtest_raw(:, keepIdx);
trueX_metric = Xtest_raw; % for later error calculation

% shrink everything to only the kept inputs
paramHdr = paramHdr(keepIdx);
muX = muX(keepIdx);
sigmaX = sigmaX(keepIdx);
Xnorm = Xnorm(:, keepIdx);
Xlog = Xlog(:,keepIdx);
lb = lb(keepIdx);
ub = ub(keepIdx);
d  = numel(lb);

% SPLIT FOR KNN SEEDING
N = size(Xnorm,1);
trainMask = S.TrainIdxGlobal;   % same 80%
Xtr = Xnorm(trainMask, :);  % only the kept dimensions
Ytr = Yraw(trainMask, :);
MdlInputKDT = KDTreeSearcher(Xtr,'Distance','euclidean'); % k-d tree in INPUT (z-scored) space

% RETRAIN SURROGATE MODELS WITH STORED HYPER PARAMETERS
J = numel(metricHdr);
models = cell(J,1);
for j = 1:J
    rng(BestSeed(j));
    models{j}.gpr = fitrgp(Xtr, ...
                           (log(Ytr(:,j)+epsX)-muY(j))./sigmaY(j), ... % z-space
                           'KernelFunction',BestKernel{j}, ...
                           'BasisFunction', BestBasis{j}, ...
                           'SigmaLowerBound',BestSigma(j), ...
                           'Standardize', false);          % inputs already z-scored
end
 
% INVERSE DESIGN SETTINGS
% optimization knobs
invOpt.nRestarts = 10; % additional purely random starts
invOpt.kNNseeds = 0; % nearest-neighbour seeds
invOpt.fminOpts = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'Display', 'iter-detailed', ...
    'Diagnostics', 'on', ...
    'FiniteDifferenceType', 'central', ...
    'FiniteDifferenceStepSize', 1e-6, ...
    'MaxFunctionEvaluations', 1e5, ...
    'StepTolerance',1e-3, ...
    'MaxIterations', 1000, ...
    'OptimalityTolerance', 1e-3, ...
    'SpecifyObjectiveGradient',false, ...
    'FunValCheck', 'on');
                   
invOpt.SA.T0 = 0.30; % initial temperature (0.5 alternative)
invOpt.SA.cool = 0.80; % cooling factor (0.85 alternative)
invOpt.SA.iters = 30; % steps / target (20 alternative)
invOpt.SA.stepSig = 0.25; % fixed proposal σ (0.2 alternative)
invOpt.SA.k = 1; % Boltzmann constant

% user selectable secondary / tertiary recipe goals (can just blank out names if not wanted)
secondaryInput = ''; % input to MINIMISE ('' = ignore)
tertiaryInput = ''; % input to MINIMISE ('' = ignore)
wSecondary = 0.05; % weight of secondary input penalty
wTertiary = 0.02; % weight of tertiary  input penalty

% Save the trained GPR models so we can reload them for single‐query
save('bestGPRSettingsARD.mat', 'models', 'muX','sigmaX','muY','sigmaY', ...
     'paramHdr','lb','ub','invOpt','Xtr','MdlInputKDT', 'Yraw', ...
     'secondaryInput','tertiaryInput','wSecondary','wTertiary', '-append');
 
% map names to column indices and z-scored target
iSecond = find(strcmp(paramHdr , secondaryInput));
iThird = find(strcmp(paramHdr , tertiaryInput ));

% containers
target_all = cell(J,1);
recovered_all = cell(J,1);
recovered_X_all = cell(J,1);
NRMSE = cell(J,1);
InputNRMSE = cell(J,1);   

fprintf('\n================ INVERSE DESIGN (v2) ================\n');

% MAIN OPTIMIZATION LOOP 
for j = 1:J
    fprintf('\n===== Metric %2d / %2d : %s =====\n', j,J,metricHdr{j});
    mdl = models{j}.gpr;

    % Choose the 20% holdout for targets
    targets = Yraw(TestIdxGlobal, j); % 20% test outputs
    trueX_metric = Xtest_raw; % true inputs for those rows

    target_all{j} = targets;
    recov = zeros(size(targets));
    recovX = cell(numel(targets),1);

    % k-NN indices for seeding
    for t = 1:numel(targets)
        tgtRaw = targets(t);
        % z-scored target for objective
        tgtNorm = (log(tgtRaw+epsX) - muY(j)) / sigmaY(j);

        % k-NN in INPUT (z-space) around a random anchor (secondary goals agnostic)
        if invOpt.kNNseeds > 0
            % k-NN seeds in INPUT (z-space)
            anchor = rand(1,d).*(ub-lb) + lb; % random anchor
            idxNN = knnsearch(MdlInputKDT, anchor, 'K', invOpt.kNNseeds);
            starts = [Xtr(idxNN,:); % k-NN seeds
                rand(invOpt.nRestarts,d).*(ub-lb) + lb];  % extra random
        else
            % purely random starts
            starts  = rand(invOpt.nRestarts,d).*(ub-lb) + lb;
        end

        prefs.primaryRaw = tgtRaw;
        prefs.primaryNorm = (log(tgtRaw+epsX)-muY(j))/sigmaY(j);
        prefs.muY = muY(j);
        prefs.sigmaY = sigmaY(j);
        prefs.epsX = epsX;
        prefs.iSecond = iSecond;
        prefs.iThird = iThird;
        prefs.w1 = 1;  
        prefs.w2 = wSecondary;  
        prefs.w3 = wTertiary;
        
        scalers.minX = lb; % both already in z-space
        scalers.rangeX = ub - lb;

        % multi-start fmincon
        bestC = inf;  
        bestX = [];
        for s = 1:size(starts,1)
            [xOpt,c] = fmincon(@(x)objFun(x,mdl,prefs,scalers), ...
                               starts(s,:),[],[],[],[],lb,ub,[],invOpt.fminOpts);
            if c < bestC, bestC = c; bestX = xOpt; end
        end

        % SA refinement (Metropolis)
        currX = bestX;  
        currC = bestC;
        bestSAx = currX; 
        bestSAc = currC;  
        Tsa = invOpt.SA.T0;

        for it = 1:invOpt.SA.iters
            propX = currX + invOpt.SA.stepSig*randn(1,d);
            propX = min(ub,max(lb,propX));
            propC = objFun(propX, mdl, prefs, scalers);
            dC = propC - currC;
            if dC < 0 || rand < exp(-dC/(invOpt.SA.k*Tsa))
                currX = propX;  
                currC = propC;
                if currC < bestSAc
                    bestSAc = currC;
                    bestSAx = currX;
                end
            end
            Tsa = Tsa*invOpt.SA.cool;
            if bestSAc < (0.05)^2, break; end  % early exit
        end

        % recover RAW prediction
        yPredNorm = predict(mdl,bestSAx);
        yPredRaw = exp(yPredNorm*sigmaY(j) + muY(j)) - epsX;
        recov(t) = yPredRaw;
        rawX = exp(bestSAx .* sigmaX + muX) - epsX;   
        recovX{t} = rawX;        

        rmsSoFar = sqrt(mean((recov(1:t)-targets(1:t)).^2)) / ...
                   (max(targets)-min(targets)+eps);
        fprintf('  target %3d/%3d  |  y* = %-10.4g  |  best NRMSE_so_far = %.4f\n', ...
                t,numel(targets), tgtRaw, rmsSoFar);
    end

    recovered_all{j} = recov;
    recovered_X_all{j} = vertcat(recovX{:});
    NRMSE{j} = sqrt((recov-targets).^2) ./ ...
               (max(targets)-min(targets)+eps);
    
    % Input-space error vs truth (RMSE per parameter)
    Xtrue = trueX_metric; % nTest x nParams  (raw units)
    Xpred = recovered_X_all{j}; % same dimensions
    rangeInp = max(Xraw,[],1) - min(Xraw,[],1) + epsX;
    relErr = (Xpred - Xtrue) ./ rangeInp; % NxD (relative per parameter)
    inputErr = sqrt( mean( relErr.^2 , 2 ) ); % Nx1  (RMS to “NRMSE” row)
    InputNRMSE{j} = inputErr; % cell, one column per metric

    fprintf('>> FINAL NRMSE (metric %s) = %.4f\n', ...
             metricHdr{j}, mean(NRMSE{j}));
end
NRMSE_vec = cellfun(@mean, NRMSE); % 1xK numeric: mean per metric
save('InverseResults.mat','metricHdr','target_all','recovered_all', ...
     'recovered_X_all','NRMSE','NRMSE_vec', 'InputNRMSE');
fprintf('\nALL DONE  |  total run-time %.1f s\n', toc(masterTic));

%%  Sanity check: Check if any recovered input exactly matches a training row

tol = 1e-6; % tolerance in RAW units
hits = false;

for j = 1:J
    recX = recovered_X_all{j};
    for t = 1:size(recX,1)
        dMin = min( vecnorm( Xraw - recX(t,:), 2, 2 ) );
        if dMin < tol
            fprintf('Metric %-25s | target %3d matched a training point (d = %.2e)\n', ...
                    metricHdr{j}, t, dMin);
            hits = true;
        end
    end
end
if ~hits
    fprintf('No recovered input is closer than %.1e to any training sample\n', tol);
end

%% PREPARE VARIABLES FOR PLOTTING & STORING

save('InverseResults.mat','metricHdr','target_all','recovered_all', ...
     'recovered_X_all','NRMSE','NRMSE_vec','InputNRMSE');

%% Machine Learning Figures (can be re-run standalone)

% Reload ML-evaluation data that were saved earlier
S = load('bestGPRSettingsARD.mat', ...
    'TestAcc','yPred','yAct','metricHdr');
TestAcc = S.TestAcc;
yPred = S.yPred;
yAct = S.yAct;
metricHdr = S.metricHdr;

% Call plotting helper function
plotSurrogateResults(metricHdr, TestAcc, yPred, yAct);

%% Optimization Figures (can be re-run standalone)

% Load mat files for section
load('bestGPRSettingsARD.mat');
load('InverseResults.mat');
load('lengthScaleData.mat');

% Call plotting helper function
plotInverseResults(metricHdr, target_all, recovered_all, NRMSE, NRMSE_vec, InputNRMSE);

%% MULTI-TARGET QUERY  (handles 1, 2 ... N metrics)
clear models muX sigmaX muY sigmaY paramHdr lb ub invOpt Xtr MdlInputKDT Yraw

% Reload everything we need to run this section
S = load('bestGPRSettingsARD.mat', ...
         'models','muX','sigmaX','muY','sigmaY','paramHdr', ...
         'lb','ub','invOpt','Xtr','MdlInputKDT', ...
         'metricHdr','Yraw', ...
         'secondaryInput','tertiaryInput','wSecondary','wTertiary');
models = S.models;
muX = S.muX;
sigmaX = S.sigmaX;
muY = S.muY;
sigmaY = S.sigmaY;
paramHdr = S.paramHdr;
lb = S.lb;
ub = S.ub;
invOpt = S.invOpt;
Xtr = S.Xtr;
MdlInputKDT = S.MdlInputKDT;
metricHdr = S.metricHdr;
Yraw = S.Yraw;
secondaryInput= S.secondaryInput;
tertiaryInput = S.tertiaryInput;
wSecondary = S.wSecondary;
wTertiary = S.wTertiary;

% Drop unwanted inputs
dropList = {'Total Thickness (mm)', ...
            'Combined Parameter B (mol m-3)', ...
            'Combined Parameter A (mol m-3)'};
keepIdx = ~ismember(paramHdr,dropList);
paramHdr = paramHdr(keepIdx);
muX = muX(keepIdx);
sigmaX = sigmaX(keepIdx);
lb = lb(keepIdx);
ub = ub(keepIdx);
Xtr = Xtr(:,keepIdx);
d = numel(lb);


% Indices for optional secondary / tertiary penalties
iSecond = find(strcmp(paramHdr,secondaryInput));
iThird = find(strcmp(paramHdr,tertiaryInput));

% Show metric names and bounds
minY = min(Yraw,[],1); % 1×K
maxY = max(Yraw,[],1);
metricNames = string(metricHdr(:));
metricInfo = compose("  %-35s  [%.4g  –  %.4g]", ...
                       metricNames, minY(:), maxY(:));
metricListStr = strjoin(metricInfo, newline);
fprintf('\nAvailable performance metrics and their bounds in the dataset:\n%s\n\n', ...
        metricListStr);

% Exclude energy efficiency metrics from display table
excludeMetrics = {'Energy Efficiency C1D2','Energy Efficiency C5D6','Energy Efficiency C9D10'};
excludeIdx = ismember(metricHdr, excludeMetrics);

% Filter out excluded metrics from the display
filteredMetricHdr = metricHdr(~excludeIdx);
minYfiltered = minY(~excludeIdx);
maxYfiltered = maxY(~excludeIdx);

% Build display string
metricNames = string(filteredMetricHdr(:));
metricInfo = compose("  %-35s  [%.4g  –  %.4g]", ...
                       metricNames, minYfiltered(:), maxYfiltered(:));
metricListStr = strjoin(metricInfo, newline);

fprintf('\nAvailable performance metrics and their bounds in the dataset:\n%s\n\n', ...
        metricListStr);

% Multi-target
% Define 1, 2 ... N metrics. All are included here for naming purposes
defMetrics = {'Capacity Retention D1D2','Capacity Retention D1D5','Capacity Retention D2D10',...
   'Coulombic Efficiency C1D2','Coulombic Efficiency C5D6','Coulombic Efficiency C9D10',...
   'Voltage Efficiency C1D2','Voltage Efficiency C5D6','Voltage Efficiency C9D10',...
   'Maximum Potential D1 (V)','Discharge Duration D1 (h)','Average Power Density D1 (W m-3)',...
   'Maximum Power Density D1 (W m-3)','Energy Density D1 (Wh m-3)'};

% Define target values for each metric (include/exclude any metrics/values here)
defVals    = [0.453,0.362,0.302,...
    0.981,0.992,0.892,...
    0.2563,0.2347,0.2246,...
    0.1918,38.7391,9.1,...
    13.4563,19];



mStr = sprintf(['Enter metric names (comma-separated) - press Enter for defaults:\n' ...
                '  %s\n> '], strjoin(defMetrics,', '));
tmp  = strtrim(input(mStr,'s'));
if isempty(tmp)
    userMetrics = defMetrics;
else
    userMetrics = strtrim(strsplit(tmp,','));
end

vStr = sprintf(['Enter target values (comma-separated, same order) - press Enter for defaults:\n' ...
                '  %s\n> '], strjoin(string(defVals),', '));
tmp = strtrim(input(vStr,'s'));
if isempty(tmp)
    targetVals = defVals;
else
    targetVals = str2double(strsplit(tmp,','));
end

% Sanity checks
if numel(userMetrics)~=numel(targetVals)
    error('Number of metrics and number of target values must match.');
end

% Remove unwanted energy efficiency metrics from user input
excludeMetrics = {'Energy Efficiency C1D2','Energy Efficiency C5D6','Energy Efficiency C9D10'};
isExcluded = ismember(userMetrics, excludeMetrics);

% Apply exclusion
userMetrics(isExcluded) = [];
targetVals(isExcluded)  = [];

% Locate surrogate models & stats
jIdx = cellfun(@(nm)find(strcmp(metricHdr,nm),1), userMetrics);
if any(isempty(jIdx))
    error('At least one metric name was not recognised.');
end
mdlList = cellfun(@(j)models{j}.gpr , num2cell(jIdx) , 'uni',0);
muYvec = muY(jIdx);
sigmaYvec = sigmaY(jIdx);

% Build prefs & scalers
epsX = 1e-6;
prefs.targetRaw = targetVals(:); % n x 1
prefs.muYvec = muYvec(:);
prefs.sigmaYvec = sigmaYvec(:);
prefs.epsX = epsX;
prefs.iSecond = iSecond;
prefs.iThird = iThird;
prefs.wPrimary = 1; % weight on output error term
prefs.wSecondary = wSecondary;
prefs.wTertiary = wTertiary;

scalers.minX = lb;
scalers.rangeX = ub - lb;

% Multi-start seeds
if invOpt.kNNseeds>0
    anchor = rand(1,d).*(ub-lb) + lb;
    idxNN  = knnsearch(MdlInputKDT,anchor,'K',invOpt.kNNseeds);
    starts = [Xtr(idxNN,:); rand(invOpt.nRestarts,d).*(ub-lb)+lb];
else
    starts = rand(invOpt.nRestarts,d).*(ub-lb)+lb;
end

% Optimization 
bestC = inf; bestX = [];
for s = 1:size(starts,1)
    [xOpt,c] = fmincon(@(x)objFunMulti(x,mdlList,prefs,scalers), ...
                       starts(s,:),[],[],[],[],lb,ub,[],invOpt.fminOpts);
    if c<bestC, bestC=c; bestX=xOpt; end
end


% SA refinement
currX = bestX; currC = bestC; bestSAx = currX; bestSAc = currC; Tsa = invOpt.SA.T0;
for it=1:invOpt.SA.iters
    propX = min(ub,max(lb,currX + invOpt.SA.stepSig*randn(1,d)));
    propC = objFunMulti(propX,mdlList,prefs,scalers);
    dC    = propC-currC;
    if dC<0 || rand<exp(-dC/(invOpt.SA.k*Tsa))
        currX=propX; currC=propC;
        if currC<bestSAc, bestSAc=currC; bestSAx=currX; end
    end
    Tsa = Tsa*invOpt.SA.cool;
    if bestSAc < (0.05)^2, break; end
end
bestX = bestSAx;

% Show results
xRaw = exp(bestX .* sigmaX + muX) - epsX;
fprintf('\n========== MULTI-TARGET RESULT ==========\n');
for k = 1:numel(userMetrics)
    yNorm = predict(mdlList{k},bestX);
    yRaw  = exp(yNorm*sigmaYvec(k) + muYvec(k)) - epsX;
    rngY  = max(Yraw(:,jIdx(k))) - min(Yraw(:,jIdx(k))) + epsX;
    fprintf('Requested  %s = %.4g\n',userMetrics{k}, targetVals(k));
    fprintf('Recovered  %s = %.4g  (%.2f %% of range error)\n', ...
            userMetrics{k}, yRaw, 100*abs(yRaw-targetVals(k))/rngY);
end
fprintf('\nOptimized recipe (raw units):\n');
for p = 1:numel(paramHdr)
    fprintf('  %-25s : %.4g\n',paramHdr{p},xRaw(p));
end
fprintf('========================================\n\n');

%% HELPER FUNCTIONS

function c = objFun(xZ, mdl, prefs, scalers)
    % predict in normalized‐log space
    yPredLog = predict(mdl, xZ);

    % back-transform into RAW units
    yPredRaw = exp(yPredLog * prefs.sigmaY + prefs.muY) - prefs.epsX;

    % squared-error on the RAW target
    cPrimary = (prefs.primaryRaw - yPredRaw).^2;

    % secondary / tertiary penalties
    cSecond = 0;
    if ~isempty(prefs.iSecond)
        z = xZ(prefs.iSecond);
        cSecond = (z - scalers.minX(prefs.iSecond)) ...
                / scalers.rangeX(prefs.iSecond);
    end
    cThird = 0;
    if ~isempty(prefs.iThird)
        z = xZ(prefs.iThird);
        cThird = (z - scalers.minX(prefs.iThird)) ...
               / scalers.rangeX(prefs.iThird);
    end

    % weighted sum
    c = prefs.w1*cPrimary + prefs.w2*cSecond + prefs.w3*cThird;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function c = objFunMulti(xZ, mdlList, prefs, scalers)
% Objective for an arbitrary number (n >= 1) of output metrics.
% - primary term = mean squared error across all requested outputs (raw units)
% - optional secondary / tertiary input penalties

nM = numel(mdlList);
err2 = zeros(nM,1); % squared errors for each metric

for k = 1:nM
    yPredLog = predict(mdlList{k}, xZ);
    yPredRaw = exp(yPredLog * prefs.sigmaYvec(k) + prefs.muYvec(k)) - prefs.epsX;
    err2(k) = (prefs.targetRaw(k) - yPredRaw).^2;
end
cPrimary = mean(err2); % average over all metrics

% Secondary / tertiary penalties
cSecond = 0;
if ~isempty(prefs.iSecond)
    z       = xZ(prefs.iSecond);
    cSecond = (z - scalers.minX(prefs.iSecond)) /  scalers.rangeX(prefs.iSecond);
end
cThird = 0;
if ~isempty(prefs.iThird)
    z      = xZ(prefs.iThird);
    cThird = (z - scalers.minX(prefs.iThird)) /  scalers.rangeX(prefs.iThird);
end

% Weighted sum 
c = prefs.wPrimary   * cPrimary + ...
    prefs.wSecondary * cSecond  + ...
    prefs.wTertiary  * cThird;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotSurrogateResults(metricHdr, TestAcc, yPred, yAct)

K  = numel(metricHdr);
accPct = TestAcc(:); % in %

% Re-order for presentation
desiredOrder = { ...
   'Maximum Potential D1 (V)'
   'Discharge Duration D1 (h)'
   'Average Power Density D1 (W m-3)'
   'Maximum Power Density D1 (W m-3)'
   'Energy Density D1 (Wh m-3)'
   'Capacity Retention D1D2'
   'Capacity Retention D1D5'
   'Capacity Retention D2D10'
   'Coulombic Efficiency C1D2'
   'Coulombic Efficiency C5C6'
   'Coulombic Efficiency C9D10'
   'Voltage Efficiency C1D2'
   'Voltage Efficiency C5D6'
   'Voltage Efficiency C9D10'
   'Energy Efficiency C1D2'
   'Energy Efficiency C5D6'
   'Energy Efficiency C9D10' };

[~,loc] = ismember(desiredOrder, metricHdr);
loc = loc(loc>0); % drop missing
accShown = accPct(loc);
lblShown = metricHdr(loc);

% (1) Horizontal bar plot
figure('Name','ML Accuracy','Color','w','Units','normalized',...
       'Position',[0.05 0.15 0.55 0.65]);
barh(accShown,'FaceColor',[.2 .4 .8],'EdgeColor','none');
set(gca,'YDir','reverse','YTick',1:numel(lblShown),'YTickLabel','',...
    'FontSize',14); grid on;
xlabel('Prediction Accuracy (%)');
for i = 1:numel(lblShown)
    text(accShown(i)+2,i,sprintf('%s  (%.1f%%)',lblShown{i},accShown(i)),...
        'FontWeight','bold','VerticalAlignment','middle');
end
xlim([0 max(accShown)*1.15]);

% (2) Predicted vs actual scatter for the first <= 3 metrics
ns = min(3,K);
figure('Name','Predicted vs Actual','Color','w','Units','normalized',...
       'Position',[0.05 0.1 0.8 0.2]);

for i = 1:ns
    subplot(1,ns,i);
    scatter(yAct{i}, yPred{i},60,'filled'); hold on; grid on;
    mn = min([yAct{i}; yPred{i}]); mx = max([yAct{i}; yPred{i}]);
    plot([mn mx],[mn mx],'k--','LineWidth',1.5);
    xlabel('Actual'); ylabel('Predicted');
    title(metricHdr{i});
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotInverseResults(metricHdr, target_all, recovered_all, NRMSEcell, NRMSEmean, InputNRMSE)

% Reorder metrics to your custom display order
 desiredOrder = { ...
     'Maximum Potential D1 (V)'
     'Discharge Duration D1 (h)'
     'Average Power Density D1 (W m-3)'
     'Maximum Power Density D1 (W m-3)'
     'Energy Density D1 (Wh m-3)'
     'Capacity Retention D1D2'
     'Capacity Retention D1D5'
     'Capacity Retention D2D10'
     'Coulombic Efficiency C1D2'
     'Coulombic Efficiency C5C6'
     'Coulombic Efficiency C9D10'
     'Voltage Efficiency C1D2'
     'Voltage Efficiency C5D6'
     'Voltage Efficiency C9D10'
     'Energy Efficiency C1D2'
     'Energy Efficiency C5D6'
     'Energy Efficiency C9D10' };
 
[~,loc] = ismember(desiredOrder, metricHdr);
loc(loc==0) = [];
metricHdr    = metricHdr(loc);
target_all   = target_all(loc);
recovered_all= recovered_all(loc);
NRMSEcell    = NRMSEcell(loc);
NRMSEmean    = NRMSEmean(loc);
InputNRMSE   = InputNRMSE(loc);

% Creates all optimization figures
K = numel(metricHdr);
set(groot,'DefaultAxesFontName','Arial');
set(groot,'DefaultTextFontName','Arial');

% (1) Recovered vs target scatter panels
cols = 2; rows = ceil(K/cols);
figure('Name','Recovered vs Target','Color','w','Units','normalized', ...
       'Position',[0.45 0.05 0.5 0.85]);
for k = 1:K
    subplot(rows,cols,k); box on; hold on;
    scatter(target_all{k}, recovered_all{k}, 50,'filled');
    mn = min([target_all{k}; recovered_all{k}]);
    mx = max([target_all{k}; recovered_all{k}]);
    plot([mn mx],[mn mx],'k--','LineWidth',1.5);
    grid on;
    title(sprintf('%s  (NRMSE = %.4f)', metricHdr{k}, NRMSEmean(k)));
    xlabel('Target'); ylabel('Recovered');
end

% (2) Ridge-style density plot and error coloring
figure('Name','Ridge Density','Color','w','Units','normalized', ...
       'Position',[0.05 0.50 0.6 0.25]);
colormap turbo; hold on;
spacing = 1.2;

allErr = []; % collect errors for caxis scaling
for k = 1:K
    t = target_all{k}; p = recovered_all{k};
    rngT = max(t)-min(t);
    err  = abs(p-t)./rngT;
    allErr = [allErr; err];

    [f,xi] = ksdensity(log10(t),'NumPoints',200);
    f = f./max(f);
    dx = (xi(2)-xi(1))/2;
    c = nan(size(xi));
    for m = 1:numel(xi)
        sel = abs(log10(t)-xi(m))<=dx;
        if any(sel), c(m)=mean(err(sel)); end
    end
    c = fillmissing(c,'nearest');

    y0 = k*spacing;
    patch('XData',[xi fliplr(xi)], ...
          'YData',[y0+f y0*ones(size(f))], ...
          'CData',[c fliplr(c)], ...
          'FaceColor','interp','EdgeColor','none','FaceAlpha',0.85);
end
ax = gca; box on;
caxis(ax,[0 max(allErr)]);
cb = colorbar(ax); ylabel(cb,'point-wise NRMSE');
yticks((1:K)*spacing); yticklabels(metricHdr);
xlabel('log_{10}(metric value)');
ylim([spacing*0.5 spacing*(K+0.5)]);
set(gca,'FontSize',12);

% (4) Error-colored scatter plot
figure('Name','Error Scatter','Color','w','Units','normalized', ...
       'Position',[0.72 0.50 0.5 0.25]);
hold on; colormap jet;
for k = 1:K
    t = target_all{k}; p = recovered_all{k};
    err = abs(p-t)./(max(t)-min(t)+eps);
    scatter(log10(t), k*ones(size(t)), 36, err, 'filled');
end
ax = gca; box on;
caxis(ax,[0 max(allErr)]);
cb=colorbar(ax); ylabel(cb,'point-wise NRMSE');
yticks(1:K); yticklabels(metricHdr);
xlabel('log_{10}(metric value)'); ylabel('Metric');
set(gca,'YDir','reverse','FontSize',12);

% 3) build per-target absolute output error  (normalized by metric range)
outErrMat = nan(max(cellfun(@numel,target_all)), K); % pre-allocate with NaNs
for k = 1:K
    rngY = max(target_all{k}) - min(target_all{k}) + eps;
    thisEr = abs(recovered_all{k} - target_all{k}) ./ rngY;
    outErrMat(1:numel(thisEr),k) = thisEr;
end

% 4) Output-space box-and-whisker (log–scale like the input-error plot)
N = numel(recovered_all{1});

% build per-row INPUT-space error matrix
K     = numel(InputNRMSE);
nTest = max(cellfun(@numel,InputNRMSE));
nrmseMat = nan(nTest,K);
for k = 1:K
    v = InputNRMSE{k};
    nrmseMat(1:numel(v),k) = v;
end
wrappedLbl = regexprep(metricHdr,'\s+','\n');

% Use the per-row input-space NRMSE we just stored
N = size(nrmseMat,1); % update N after building matrix
goodNRMSE = 0.10; % threshold for “good” designs

figure('Name','Output-space error','Color','w','Units','normalized',...
       'Position',[0.05 0.15 0.85 0.60]);
hold on;

% colored background bands as before (green=good, red=bad)
goodE = 0.1; % 10% of range deemed "very good"
ylims = [min(outErrMat(:))*0.8 , max(outErrMat(:))*5];
xlims = [0.5 , K+0.5];
fill([xlims fliplr(xlims)], [goodE goodE ylims(2) ylims(2)], ...
     [0.30 0.05 0.05],'EdgeColor','none','FaceAlpha',0.20);
fill([xlims fliplr(xlims)], [ylims(1) ylims(1) goodE goodE], ...
     [0.05 0.30 0.05],'EdgeColor','none','FaceAlpha',0.20);

outErrMat = nan(max(cellfun(@numel,target_all)),K);
for k = 1:K
    rngY = max(target_all{k}) - min(target_all{k}) + eps;
    thisEr = abs(recovered_all{k} - target_all{k}) ./ rngY;
    outErrMat(1:numel(thisEr),k) = thisEr;
end
 
% make sure # labels == # cols (already handled earlier if metrics trimmed)
boxplot(outErrMat,'Labels',wrappedLbl(:)', ...
        'LabelOrientation','inline','Whisker',1.5,'Colors','k');
set(findobj(gca,'Type','Line'),'LineWidth',1.5,'Color','k');
set(gca,'YScale','log','YLim',ylims,'XLim',xlims,...
        'LineWidth',1.5,'FontSize',14,'Box','off');
xlabel('Performance Metric','FontWeight','bold','FontSize',16);
ylabel('Log-scaled per-target NRMSE','FontWeight','bold','FontSize',16);
title('Distribution of Output-Space Normalized RMSE','FontWeight','bold','FontSize',16);
yline(goodE,'--k',sprintf('Good NRMSE ≤ %.2f',goodE),'FontSize',12, 'FontWeight','bold',...
      'LabelHorizontalAlignment','right','LabelVerticalAlignment','bottom');

% 7) Console summaries
fprintf('\n=== Mean OUTPUT-space error (%% of range) ===\n');
meanOut = mean(outErrMat,'omitnan');
for k = 1:K
    fprintf('  %-28s : %.4f\n', metricHdr{k}, meanOut(k));
end

thr = 0.10; % 10 % of parameter-range threshold
succRate = cellfun(@(e) mean(e < thr)*100, InputNRMSE);
fprintf('\nSuccess rate (input-error < %.0f %% of param range):\n', thr*100);
for k = 1:K
    fprintf('  %-28s : %5.1f %%\n', metricHdr{k}, succRate(k));
end

end

