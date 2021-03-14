function [res, dres] = roi_classify_cm( paths1rst, paths2rst, cpbimasks, maps)
%ROI_CLASSIFY_NEW compare decodability between neighboring rooms and
%diagnal rooms
%
% the spatial layout of the rooms are like:
%     1 ---- 2
%     |      |
%     |      |
%     3 ---- 4
%
% the contrast matrix (sum to zero):
%
% neigboring rooms (marked with 1):
%     room 1 and 2, room 1 and 3, room 2 and 4, room 3 and 4
% diagnal rooms (marked with -2):
%     room 1 and 4, room 2 and 3

contrast = [0  1  1 -2
            1  0 -2  1
            1 -2  0  1
           -2  1  1  0];

% to generate all possible combinations of 4 training objects vs 4 testing
% objects, the result for each subject is meant to be the average of
% results from all those combinations
[a,b,c,d] = ndgrid(1:2, 3:4, 5:6, 7:8);
combo = [reshape(a, [], 1), reshape(b, [], 1), reshape(c, [], 1), reshape(d, [], 1)];

nperm = 100;
res1 = NaN(size(cpbimasks));
dres1 = cell(size(cpbimasks));
res2 = NaN(size(cpbimasks));
dres2 = cell(size(cpbimasks));

for imask = 1:size(cpbimasks, 2)
    parfor isubj = 1:size(cpbimasks, 1)
        rng('default');
        
        % skip missing ROIs
        if isubj == 18 && ismember(imask, [4, 12:13]) %OPA
            continue
        elseif isubj == 9 && ismember(imask, [1:2, 6:9]) %LOC/PPA
            continue
        end
        
        map = maps{isubj};
        
        % load session 1
        ds_mat1 = cell(6,1);
        for irun = 1:6
            ds_run = cell(8,1);
            for iobj = 1:8
                kobj = find(map==iobj); %remap
                ds_run{kobj} = cosmo_fmri_dataset(paths1rst{isubj}{irun,iobj},...
                    'mask',cpbimasks{isubj,imask}, ...
                    'targets',kobj, ...
                    'chunks',irun);
            end
            dsr = cosmo_stack(ds_run);
            ds_mat1{irun} = cosmo_normalize(dsr,'zscore',1);
        end
        ds1 = cosmo_stack(ds_mat1);
        ds1 = cosmo_remove_useless_data(ds1);
        
        % load session 2
        ds_mat2 = cell(6,1);
        for irun = 1:6
            ds_run = cell(8,1);
            for iobj = 1:8
                kobj = find(map==iobj);
                ds_run{kobj} = cosmo_fmri_dataset(paths2rst{isubj}{irun,iobj},...
                    'mask',cpbimasks{isubj,imask}, ...
                    'targets',kobj, ...
                    'chunks',irun);
            end
            dsr = cosmo_stack(ds_run);
            ds_mat2{irun} = cosmo_normalize(dsr,'zscore',1);
        end
        ds2 = cosmo_stack(ds_mat2);
        ds2 = cosmo_remove_useless_data(ds2);
        
        % calculate confusion matrix for all combinations
        cm1 = zeros(4, 4);
        cm2 = zeros(4, 4);
        for ic = 1:size(combo, 1)
            ds11 = ds1;
            ds11.sa.chunks = ismember(ds11.sa.targets, combo(ic,:))+1;
            ds11.sa.targets = ceil(ds11.sa.targets./2);
            measure_args = struct();
            measure_args.classifier = @cosmo_classify_lda;
            measure_args.output = 'winner_predictions';
            measure_args.partitions = cosmo_oddeven_partitioner(ds11, 'half');
            out = cosmo_crossvalidation_measure(ds11,measure_args);
            cm1 = cm1 + cosmo_confusion_matrix(out); % added them together
            
            ds22 = ds2;
            ds22.sa.chunks = ismember(ds22.sa.targets, combo(ic,:))+1;
            ds22.sa.targets = ceil(ds22.sa.targets./2);
            measure_args = struct();
            measure_args.classifier = @cosmo_classify_lda;
            measure_args.output = 'winner_predictions';
            measure_args.partitions = cosmo_oddeven_partitioner(ds22, 'half');
            out = cosmo_crossvalidation_measure(ds22,measure_args);
            cm2 = cm2 + cosmo_confusion_matrix(out); % added them together
        end
        
        % calculate the contrast values
        res1(isubj, imask) = sum(reshape(contrast.*cm1, [], 1));
        res2(isubj, imask) = sum(reshape(contrast.*cm2, [], 1));
        
        % 100 permutations within each subject
        pres = NaN(2, nperm);
        for iperm = 1:nperm
            targets_objs = cosmo_randomize_targets(ds1);
            cm1 = zeros(4, 4);
            cm2 = zeros(4, 4);
            for ic = 1:size(combo, 1)
                ds11 = ds1;
                ds11.sa.chunks = ismember(targets_objs, combo(ic,:))+1;
                ds11.sa.targets = ceil(targets_objs./2);
                measure_args = struct();
                measure_args.classifier = @cosmo_classify_lda;
                measure_args.output = 'winner_predictions';
                measure_args.partitions = cosmo_oddeven_partitioner(ds11, 'half');
                out = cosmo_crossvalidation_measure(ds11,measure_args);
                cm1 = cm1 + cosmo_confusion_matrix(out);
                
                ds22 = ds2;
                ds22.sa.chunks = ismember(targets_objs, combo(ic,:))+1;
                ds22.sa.targets = ceil(targets_objs./2);
                measure_args = struct();
                measure_args.classifier = @cosmo_classify_lda;
                measure_args.output = 'winner_predictions';
                measure_args.partitions = cosmo_oddeven_partitioner(ds22, 'half');
                out = cosmo_crossvalidation_measure(ds22,measure_args);
                cm2 = cm2 + cosmo_confusion_matrix(out);
            end
            pres(1, iperm) = sum(reshape(contrast.*cm1, [], 1));
            pres(2, iperm) = sum(reshape(contrast.*cm2, [], 1));
        end
        dres1{isubj, imask} = pres(1, :);
        dres2{isubj, imask} = pres(2, :);
    end
end
res = {res1, res2};
dres = {dres1, dres2};
end

