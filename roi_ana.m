%% roi_ana
% Analysis code for the paper
% "The Parahippocampal Place Area and Hippocampus Encode the Spatial
% Significance of Landmark Objects"
% Preprocessing and GLM were carried out by FSL
% This code is mainly for MVPA analysis
% cosmomvpa toolbox is required.

%% load
load('remaps.mat'); % load subjects and objects mapping

% two subjects excluded
subjs([15,17]) = [];
maps([15,17],:) = [];

% running at lab
dirbase = '/mnt/data1/data/vr1/';

%% load masks (21 ROIs)
cbimasks = {'loc','ppa','rsc','opa', ...
    'lh_loc', 'rh_loc', 'lh_ppa', 'rh_ppa', 'lh_rsc', 'rh_rsc', 'lh_opa', 'rh_opa', ...
    'l_hippo', 'r_hippo', 'hippo'};
cpbimasks = cellfun(@(s) cellfun(@(m) [dirbase,'fsl_analysis/',s,'/rois/equal/',m,'.nii.gz'],cbimasks,'uni',0),subjs,'uni',0);
cpbimasks = cat(1,cpbimasks{:});

cmasks = {'hippo_ant', 'hippo_pos', 'lh_hippo_ant', 'rh_hippo_ant', 'lh_hippo_pos', 'rh_hippo_pos'};
cpmasks = cellfun(@(s) cellfun(@(m) [dirbase,'fsl_analysis/',s,'/rois/',m,'.nii.gz'],cmasks,'uni',0),subjs,'uni',0);
cpmasks = cat(1,cpmasks{:});

% paths to all the masks
cpbimasks = [cpbimasks, cpmasks];
% names for all the masks
cbimasks = [cbimasks, cmasks];

% load beta maps from GLM
dirs1t = cellfun(@(s) [dirbase,'/fsl_analysis/',s,'/cope1/'],subjs,'uni',0);
dirs2t = cellfun(@(s) [dirbase,'/fsl_analysis/',s,'/cope2/'],subjs,'uni',0);

paths1rst = cellfun(@(s) arrayfun(@(r) arrayfun(@(obj) [s,'run',num2str(r),'obj',num2str(obj),'.nii.gz'], 1:8,'uni',0),(1:6)','uni',0), dirs1t', 'uni',0);
paths1rst = cellfun(@(s) cat(1,s{:}),paths1rst,'uni',0);

paths2rst = cellfun(@(s) arrayfun(@(r) arrayfun(@(obj) [s,'run',num2str(r),'obj',num2str(obj),'.nii.gz'], 1:8,'uni',0),(1:6)','uni',0), dirs2t', 'uni',0);
paths2rst = cellfun(@(s) cat(1,s{:}),paths2rst,'uni',0);

%% object shape classification
% 8-way classification with 100 permutations within each participant
[res, dres] = roi_classify_objs( paths1rst, paths2rst, cpbimasks);

% mean accuracies
mean_res1 = nanmean(res{1});
mean_res2 = nanmean(res{2});

% 10k bootstrapping at group-level
nboots = 10000;
pval1 = pool_test(res{1}, dres{1}, nboots);
pval2 = pool_test(res{2}, dres{2}, nboots);

% FDR & one-sided test for pre and post learning classification accuracies
[~,~,~,corp1] = fdr_bh(1-pval1);
[~,~,~,corp2] = fdr_bh(1-pval2);

% two-sided test for change from pre to post learning
res_diff = res{2}-res{1};
mean_diff = nanmean(res_diff);
dres_diff = cellfun(@(r1, r2) r2-r1, dres{1}, dres{2}, 'uni', 0);
pval = pool_test(res_diff, dres_diff, nboots);
pval(pval > .5) = 1-pval(pval > .5);
pval = pval*2;
[~,~,~,corp] = fdr_bh(pval);

save('res_obj.mat', 'mean_res1', 'mean_res2', 'mean_diff', ...
    'corp1', 'corp2', 'corp');
%% object location classification
% 4-way classification with 100 permutations within each participant
[res, dres] = roi_classify_locs( paths1rst, paths2rst, cpbimasks, maps);

% mean accuracies
mean_res1 = nanmean(res{1});
mean_res2 = nanmean(res{2});

% 10k bootstrapping at group-level
nboots = 10000;
pval1 = pool_test(res{1}, dres{1}, nboots);
pval2 = pool_test(res{2}, dres{2}, nboots);

% FDR & one-sided test for pre and post learning classification accuracies
[~,~,~,corp1] = fdr_bh(1-pval1);
[~,~,~,corp2] = fdr_bh(1-pval2);

% two-sided test for change from pre to post learning
res_diff = res{2}-res{1};
mean_diff = nanmean(res_diff);
dres_diff = cellfun(@(r1, r2) r2-r1, dres{1}, dres{2}, 'uni', 0);
pval = pool_test(res_diff, dres_diff, nboots);
pval(pval > .5) = 1-pval(pval > .5);
pval = pval*2;
[~,~,~,corp] = fdr_bh(pval);

save('res_loc.mat', 'mean_res1', 'mean_res2', 'mean_diff', ...
    'corp1', 'corp2', 'corp');

%% look into the confusion matrix
% we contrast classification errors between neigbhoring rooms and
% classification errors between distant/diagnol rooms to see if there was
% any distance effect
[res, dres] = roi_classify_cm( paths1rst, paths2rst, cpbimasks, maps);

% mean accuracies
mean_res1 = nanmean(res{1});
mean_res2 = nanmean(res{2});

% 10k bootstrapping at group-level
nboots = 10000;
pval1 = pool_test(res{1}, dres{1}, nboots);
pval2 = pool_test(res{2}, dres{2}, nboots);

% FDR & one-sided test for pre and post learning classification accuracies
[~,~,~,corp1] = fdr_bh(1-pval1);
[~,~,~,corp2] = fdr_bh(1-pval2);

% two-sided test for change from pre to post learning
res_diff = res{2}-res{1};
mean_diff = nanmean(res_diff);
dres_diff = cellfun(@(r1, r2) r2-r1, dres{1}, dres{2}, 'uni', 0);
pval = pool_test(res_diff, dres_diff, nboots);
pval(pval > .5) = 1-pval(pval > .5);
pval = pval*2;
[~,~,~,corp] = fdr_bh(pval);

save('res_cm.mat', 'mean_res1', 'mean_res2', 'mean_diff', ...
    'corp1', 'corp2', 'corp');