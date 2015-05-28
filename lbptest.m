
%         I=imread('test1.bmp');
%         mapping=getmapping(8,'u2');
%         H1=lbp(I,1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood %using uniform patterns
%         subplot(1,1,1),stem(H1);
      I=imread('test1.bmp');
      row=size(I,1);
      col=size(I,2);
      B=mat2cell(I,[row/2 row/2],[col/2 col/2]);
      mapping=getmapping(8,'u2');
      H.a=0;
      for i=1:4
      H1=lbp(B{i},1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood %using uniform patterns
      H.hist{i}=H1;
      end                           
      subplot(2,2,1),stem(H.hist{1});
      subplot(2,2,2),stem(H.hist{2});
      subplot(2,2,3),stem(H.hist{3});
      subplot(2,2,4),stem(H.hist{4});














 