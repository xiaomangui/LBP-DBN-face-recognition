function X = sigmrnd(P)
%在将Matlab的程序移植到C++上时，不适合产生随机数，所有，在[0,1]之间的随机数，我们就用[1/3]来替代
%     X = double(1./(1+exp(-P)))+1*randn(size(P));
%     A=ones(size(P));
%     B=A/3;
%     C=1./(1+exp(-P));
%     X = double(1./(1+exp(-P)) > B);
    X = double(1./(1+exp(-P)) > rand(size(P)));
end