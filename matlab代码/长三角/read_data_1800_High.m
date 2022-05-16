% 本函数Input为
% interval - 数据划分长度，默认为6400，即每6400个数据点划为一个样本
% ind_column - 通道，默认为2，即选取第二个通道
% Output为
% label1 label2, ..., label8，分别为划分好的8种故障类型的样本
% 
function [label1, label2, label3, label4, label5]= read_data_1800_High( interval, ind_column )
if nargin <2
    ind_column=1; %如果传递的实参小于2个，默认ind_column为2
end 

if nargin <1
    interval=200; %默认interval=6400
end

file_rul='C:\Users\LeonardoZhou\Documents\MATLAB\长三角\jccs1\';
% 以下为获取file_rul路径下.mat格式的所有文件
file_folder=fullfile(file_rul);
dir_output=dir(fullfile(file_folder,'*.mat'));
file_name={dir_output.name}';
num_file=max(size(file_name)); %num_file为文件数，本例中num_file=8，8个文件，分别存储齿轮箱的8种故障数据

for i=1:num_file
    file=[file_rul,file_name{i}];
    load(file);
    [filepath, name, ext]=fileparts(file);
    raw=eval(name);
    
	% 每6400个点划分为一个样本
    n=1;
    left_index=1+(n-1)*interval;
    right_index=n*interval;
    while right_index<=size(raw,1)
         temp=raw(left_index : right_index, ind_column);
         % eval函数构造label1, label2,...等变量名
         eval(['label' num2str(i) '(:,n)=temp;']); 
         n=n+1;
         left_index=1+(n-1)*interval;
         right_index=n*interval;
    end
end
end
