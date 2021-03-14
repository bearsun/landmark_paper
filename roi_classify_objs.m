function [res, dres] = roi_classify_objs( paths1rst, paths2rst, cpbimasks)
%ROI_CLASSIFY classify by objects (8-way)

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
        
        % load session 1
        ds_mat1 = cell(6,1);
        for irun = 1:6
            ds_run = cell(8,1);
            for iobj = 1:8
                ds_run{iobj} = cosmo_fmri_dataset(paths1rst{isubj}{irun,iobj},...
                    'mask',cpbimasks{isubj,imask}, ...
                    'targets',iobj, ...
                    'chunks',irun);
            end
            dsr = cosmo_stack(ds_run);
            ds_mat1{irun} = cosmo_normalize(dsr,'zscore');
        end
        ds1 = cosmo_stack(ds_mat1);
        ds1 = cosmo_remove_useless_data(ds1);
        
        % load session 2
        ds_mat2 = cell(6,1);
        for irun = 1:6
            ds_run = cell(8,1);
            for iobj = 1:8
                ds_run{iobj} = cosmo_fmri_dataset(paths2rst{isubj}{irun,iobj},...
                    'mask',cpbimasks{isubj,imask}, ...
                    'targets',iobj, ...
                    'chunks',irun);
            end
            dsr = cosmo_stack(ds_run);
            ds_mat2{irun} = cosmo_normalize(dsr,'zscore');
        end
        ds2 = cosmo_stack(ds_mat2);
        ds2 = cosmo_remove_useless_data(ds2);
        
        ds11 = ds1;
        measure_args = struct();
        measure_args.classifier = @cosmo_classify_lda;
        measure_args.partitions = cosmo_nfold_partitioner(ds11);
        out = cosmo_crossvalidation_measure(ds11,measure_args);
        acc(1) = out.samples;
        
        ds22 = ds2;
        measure_args = struct();
        measure_args.classifier = @cosmo_classify_lda;
        measure_args.partitions = cosmo_nfold_partitioner(ds22);
        out = cosmo_crossvalidation_measure(ds22,measure_args);
        acc(2) = out.samples;

        % 100 permutations within each subject
        for iperm = 1:nperm
            ds01 = ds1;
            targets_objs = cosmo_randomize_targets(ds01);
            ds01.sa.targets = targets_objs;
            measure_args = struct();
            measure_args.classifier = @cosmo_classify_lda;
            measure_args.partitions = cosmo_nfold_partitioner(ds01);
            pout = cosmo_crossvalidation_measure(ds01,measure_args);
            pacc(1,iperm) = pout.samples;
            
            ds02 = ds2;
            ds02.sa.targets = targets_objs;
            measure_args = struct();
            measure_args.classifier = @cosmo_classify_lda;
            measure_args.partitions = cosmo_nfold_partitioner(ds02);
            pout = cosmo_crossvalidation_measure(ds02,measure_args);
            pacc(2,iperm) = pout.samples;
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

