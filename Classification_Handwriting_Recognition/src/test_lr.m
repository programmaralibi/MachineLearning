W = train_lr;
load ('Y.mat');
row_=zeros(11,1);
row_(2)=150;
row_(3)=150;
row_(4)=150;
row_(5)=150;
row_(6)=150;
row_(7)=150;
row_(8)=150;
row_(9)=150;
row_(10)=150;
row_(11)=150;
[row1,col1]=size(Y);

T=zeros(row1,1);
r=1;
for i=1:10
    for j=1:150
        T(j+(i-1)*150,1)=i;
    end
end

activation_fuction1 = Y*W;
y2=bsxfun(@rdivide,exp(activation_fuction1),sum(exp(activation_fuction1),2));

fid = fopen('classes_lr.txt', 'w+');
counter=0;
column=0;
for i=1:1500
    [value column]=max(y2(i,:));
    fprintf( fid ,'%d\n',column);
    if(column ~= T(i));
        counter = counter +1;
    end
end
fclose(fid);
error_rate=counter/1500