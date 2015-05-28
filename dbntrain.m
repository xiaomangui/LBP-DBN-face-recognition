function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);
    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
    %DBN是由多层RBM组成，训练方法是逐层训练，先训练第一层网络，然后固定第一层网络的参数，将第一层网络的输出作为下一层网络的输入
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
        opts.numepochs=opts.numepochs;        
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end
end
