function pval = pool_test( res, dres, nboot)
%POOL_TEST bootstrapping to simulate the group-level distribution
rng('default');
nmask = size(res, 2);
bdis = NaN(nboot, nmask);
nperm = numel(dres{1});
for imask = 1:nmask
    cmasks = dres(:, imask);
    cmasks = cmasks(~cellfun('isempty', cmasks));
    nsubj = numel(cmasks);
    for ib = 1:nboot
        r = num2cell(randi(nperm, [nsubj, 1]));
        bdis(ib, imask) = mean(cell2mat(cellfun(@(p,i) p(i), cmasks, r, 'uni', 0)));
    end
end

pval = NaN(1, nmask);
for imask = 1:nmask
    pval(imask) = (sum(nanmean(res(:, imask)) > bdis(:, imask)) + 1) / (nboot + 2);
end

end
