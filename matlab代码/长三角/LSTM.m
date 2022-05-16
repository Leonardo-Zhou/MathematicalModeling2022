clear;
fs=6400; %采样频率
%% 载入数据
[gearbox0, gearbox1, gearbox2, gearbox3, gearbox4]=read_data();

%% 数据处理
gearbox0_x=num2cell(gearbox0,1);
gearbox1_x=num2cell(gearbox1,1);
gearbox2_x=num2cell(gearbox2,1);
gearbox3_x=num2cell(gearbox3,1);
gearbox4_x=num2cell(gearbox4,1);

n1=length(gearbox0_x);
n2=length(gearbox1_x);
n3=length(gearbox2_x);
n4=length(gearbox3_x);
n5=length(gearbox4_x);

gearbox0_y=cell(n1,1);
gearbox1_y=cell(n2,1);
gearbox2_y=cell(n3,1);
gearbox3_y=cell(n4,1);
gearbox4_y=cell(n5,1);

for i=1:n1 
    gearbox0_y{i}='0'; 
end
for i=1:n2
    gearbox1_y{i}='1'; 
end
for i=1:n3 
    gearbox2_y{i}='2'; 
end
for i=1:n4
    gearbox3_y{i}='3'; 
end
for i=1:n5
    gearbox4_y{i}='4'; 
end

train_0=(1:n1-5);test_0=(n1-4:n1);
train_1=(1:n2-5);test_1=(n2-4:n2);
train_2=(1:n3-5);test_2=(n3-4:n3);
train_3=(1:n4-5);test_3=(n4-4:n4);
train_4=(1:n5-5);test_4=(n5-4:n5);

x_train_0=gearbox0_x(train_0);
y_train_0=gearbox0_y(train_0);
x_train_1=gearbox1_x(train_1);
y_train_1=gearbox1_y(train_1);
x_train_2=gearbox2_x(train_2);
y_train_2=gearbox2_y(train_2);
x_train_3=gearbox3_x(train_3);
y_train_3=gearbox3_y(train_3);
x_train_4=gearbox4_x(train_4);
y_train_4=gearbox4_y(train_4);

x_test_0=gearbox0_x(test_0);
y_test_0=gearbox0_y(test_0);
x_test_1=gearbox1_x(test_1);
y_test_1=gearbox1_y(test_1);
x_test_2=gearbox2_x(test_2);
y_test_2=gearbox2_y(test_2);
x_test_3=gearbox3_x(test_3);
y_test_3=gearbox3_y(test_3);
x_test_4=gearbox4_x(test_4);
y_test_4=gearbox4_y(test_4);

x_train=[x_train_0 x_train_1 x_train_2 x_train_3 x_train_4 ];
y_train=[y_train_0; y_train_1; y_train_2; y_train_3; y_train_4];
n_train=size(x_train,2);
x_test=[x_test_0  x_test_1  x_test_2 x_test_3 x_test_4 ];
y_test=[y_test_0; y_test_1; y_test_2; y_test_3; y_test_4];
n_test=size(x_test,2);

%% 提取小波包能量
nl=5; %分解层数
e=0:1:2^nl-1;
t_train=cellfun(@(x) wpdec(x,nl,'dmey'), x_train, 'UniformOutput', false);
t_test=cellfun(@(x) wpdec(x,nl,'dmey'), x_test, 'UniformOutput', false);

for i=1:n_train
    for j=1:length(e)
        re=wprcoef(t_train{i},[nl,e(j)]);
        energy(j)=sum(re.^2);
    end
    e_train_d(i,:)=energy;
end

e_train=num2cell(e_train_d,2);
e_train=e_train';

for i=1:n_test
    for j=1:length(e)
        re=wprcoef(t_test{i},[nl,e(j)]);
        energy(j)=sum(re.^2);
    end
 e_test_d(i,:)=energy;
end

e_test=num2cell(e_test_d,2);
e_test=e_test';

%% 将小波包能量作为输入特征
x_train_c=cellfun(@(x)[x], e_train', 'UniformOutput',false);
x_test_c=cellfun(@(x)[x], e_test', 'UniformOutput',false);

%% 数据再处理
x_b=[x_train_c{:}];
m=mean(x_b,2);
s=std(x_b,[],2);
x_train_c_sd=x_train_c;
x_train_c_sd=cellfun(@(x)(x-m)./s, x_train_c_sd,'UniformOutput',false);
x_test_c_sd=x_test_c;
x_test_c_sd=cellfun(@(x)(x-m)./s,x_test_c_sd,'UniformOutput',false);

%% 设计网络
y_train_c=categorical(y_train);
n_c=numel(categories(y_train_c));
y_test_c=categorical(y_test);
ip=size(x_train_c_sd{1},1); 
l = [sequenceInputLayer(ip)
     lstmLayer(256,'OutputMode','last')
    fullyConnectedLayer(n_c)
    softmaxLayer
    classificationLayer
    ];

% 设置网络
options = trainingOptions('adam','ExecutionEnvironment', 'gpu','SequenceLength', 'longest','MaxEpochs',500, 'MiniBatchSize', 16, 'InitialLearnRate', 0.001,'GradientThreshold', 1,'plots','training-progress','Verbose',true);

% 训练网络
train_p = trainNetwork(x_train_c_sd,y_train_c,l,options);

% 测试网络
test_p = classify(train_p,x_test_c_sd);

% 打印混淆矩阵
plotconfusion(y_test_c',test_p','LSTM神经网络')
