function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    for i = 1 : opts.numepochs
        %kk = randperm(m);
        err = 0;
        for l = 1 : numbatches
            if(l==2)
               l=2; 
            end
            %batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            batch=x(l,:);
            v1 = batch;
            B=repmat(rbm.c', opts.batchsize, 1);
            C=v1 * rbm.W';
            A=repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W';
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
            v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');

            c1 = h1' * v1;
            c2 = h2' * v2;

            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
            a1=sum(v1 - v2)';
            b1=(v1 - v2)';
            b2=sum(b1);
            a3=rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            a2=sum(h1 - h2)';
            a4=rbm.momentum * rbm.vc + rbm.alpha * a2 / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;
            a=sum(sum((v1 - v2) .^ 2));
%             disp([a]);
            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
    end

        
end
