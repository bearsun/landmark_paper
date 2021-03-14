function [res, dres] = roi_classify_locs( paths1rst, paths2rst, cpbimasks, maps)
%ROI_CLASSIFY_NEW train on one object in the room and test on another
% 4-way classification on locations

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
            
        acc = NaN(2,1);
        pacc = NaN(2,nperm);
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
        
        % calculate accuracies for all combinations
        acc1 = NaN(size(combo, 1), 1);
        acc2 = NaN(size(combo, 1), 1);
        for ic = 1:size(combo, 1)
            ds11 = ds1;
            % this is not a typical leave-one-run-out cross-validation,
            % instead, it is a leave-four-objects-out cross-validation
            % based on data from all runs.
            ds11.sa.chunks = ismember(ds11.sa.targets, combo(ic,:))+1;
            ds11.sa.targets = ceil(ds11.sa.targets./2);
            measure_args = struct();
            measure_args.classifier = @cosmo_classify_lda;
            measure_args.partitions = cosmo_oddeven_partitioner(ds11, 'half');
            out = cosmo_crossvalidation_measure(ds11,measure_args);
            acc1(ic) = out.samples;
            
            ds22 = ds2;
            ds22.sa.chunks = ismember(ds22.sa.targets, combo(ic,:))+1;
            ds22.sa.targets = ceil(ds2.sa.targets./2);
            measure_args = struct();
            measure_args.classifier = @cosmo_classify_lda;
            measure_args.partitions = cosmo_oddeven_partitioner(ds22, 'half');
            out = cosmo_crossvalidation_measure(ds22,measure_args);
            acc2(ic) = out.samples;
        end
        acc(1) = mean(acc1);
        acc(2) = mean(acc2);

        % 100 permutations within a subject
        for iperm = 1:nperm
            ds01 = ds1;
            targets_objs = cosmo_randomize_targets(ds01);
            
            pacc1 = NaN(size(combo,1), 1);
            pacc2 = NaN(size(combo,1), 1);
            for ic = 1:size(combo, 1)
                ds01.sa.chunks = ismember(targets_objs, combo(ic,:))+1;
                ds01.sa.targets = ceil(targets_objs./2);
                measure_args = struct();
                measure_args.classifier = @cosmo_classify_lda;
                measure_args.partitions = cosmo_oddeven_partitioner(ds01, 'half');
                pout = cosmo_crossvalidation_measure(ds01,measure_args);
                pacc1(ic) = pout.samples;
                
                ds02 = ds2;
                ds02.sa.chunks = ismember(targets_objs, combo(ic,:))+1;
                ds02.sa.targets = ceil(targets_objs./2);
                measure_args = struct();
                measure_args.classifier = @cosmo_classify_lda;
                measure_args.partitions = cosmo_oddeven_partitioner(ds02, 'half');
                pout = cosmo_crossvalidation_measure(ds02,measure_args);
                pacc2(ic) = pout.samples;
            end
            pacc(1,iperm) = mean(pacc1);
            pacc(2,iperm) = mean(pacc2);
        end
        dres1{isubj, imask} = pacc(1, :);
        res1(isubj, imask) = acc(1);
        dres2{isubj, imask} = pacc(2, :);
        res2(isubj, imask) = acc(2);
        
    end
end

res = {res1, res2};
dres = {dres1, dres2};

end

