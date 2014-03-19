function [Weight] = train

clear all;
format long;
close all;
%%
load ('X.mat');
x=X;
rW_=zeros(11,1);
rW_(2)=2000;
rW_(3)=1979;
rW_(4)=1999;
rW_(5)=2000;
rW_(6)=2000;
rW_(7)=2000;
rW_(8)=2000;
rW_(9)=2000;
rW_(10)=2000;
rW_(11)=2000;
[row,col]=size(x);

%%
Weight= rand(513,10);
Target=zeros(row,10);
r=1;
r1=1;
for i=1:10
    r=rW_(i)+r;
    r1=rW_(i+1)+r1;
    for j=r:(r1-1)
        Target(j,i)= 1;
    end
end

%%
%activation fuction and softmax function
act_ftn = x*Weight ;
y1=bsxfun(@rdivide,exp(act_ftn),sum(exp(act_ftn),2));

 %%
% find del E and error
del_E=((y1-Target)'*x)';
E= sum( sum( Target.* log(y1) ) );
E= -E;

%%
%find minimum value
old_Error = inf;
new_Error = E;
eta = 0.00003;

for i=1:10
    Weight = Weight - (eta*del_E);
    del_E_old=del_E;
    act_ftn = x*Weight ;
    y1=bsxfun(@rdivide,exp(act_ftn),sum(exp(act_ftn),2));
    del_E = ((y1-Target)'*x)';
    old_Error = new_Error;
    new_Error = sum( sum( Target.*( log(y1) ) ) );
    new_Error = -new_Error
    if (new_Error < old_Error)
        old_Error=new_Error;
    else
        Weight=Weight+(eta*del_E_old);
        del_E=del_E_old;
        eta=0.75*eta;
    end
    error(i)= new_Error;       
end
plot(1:10,error);
xlabel('Varying Weight matrix ');
ylabel('Error');
end