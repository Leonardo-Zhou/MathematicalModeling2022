function [gearbox0, gearbox1, gearbox2, gearbox3, gearbox4]= readdata( ~, ~ )

sensor=2; %数据选取

length=200;  %数据划分长度

address='C:\Users\LeonardoZhou\Documents\MATLAB\长三角\jccs1\';%数据文件地址

ff=fullfile(address);
op=dir(fullfile(ff,'*.mat'));
fname={op.name}';
fnum=max(size(fname)); 

for i=1:fnum
    file=[address,fname{i}];
    load(file);
    [~, name, ~]=fileparts(file);
    z=eval(name);
    n=1;
    l=1+(n-1)*length;
    r=n*length;
    while r<=size(z,1)
         temp=z(l:r,sensor);
         eval(['gearbox' num2str(i-1) '(:,n)=temp;']); 
         n=n+1;
         l=1+(n-1)*length;
         r=n*length;
    end
end
end
