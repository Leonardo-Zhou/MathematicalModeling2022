clear;
% 读取数据，label1为一个6400*83的数组，83为每种故障类型所得到的样本数
[label1, label2, label3, label4, label5]=read_data_1800_High();
num_categories=5;
fs=6400;
% 由于matlab中LSTM建模需要，用num2cell函数将label1转为cell型，label_x_cell为一个1×83的cell型数组，每个cell存储6400个数据点
label1_x_cell=num2cell(label1,1);
label2_x_cell=num2cell(label2,1);
label3_x_cell=num2cell(label3,1);
label4_x_cell=num2cell(label4,1);
label5_x_cell=num2cell(label5,1);

num_1=length(label1_x_cell);
num_2=length(label2_x_cell);
num_3=length(label3_x_cell);
num_4=length(label4_x_cell);
num_5=length(label5_x_cell);

% 创建用于存储每种故障类型的标签的数据结构，由于matlab中lstm建模需要，也需要cell型数据。例如，label1_y为一个83×1的cell型数组，目前其值为空
label1_y=cell(num_1,1);
label2_y=cell(num_2,1);
label3_y=cell(num_3,1);
label4_y=cell(num_4,1);
label5_y=cell(num_5,1);

% 创建故障类型的标签，用1,2,3,...,8表示8种故障标签，给对应标签赋值。
for i=1:num_1; label1_y{i}='0'; end
for i=1:num_2; label2_y{i}='1'; end
for i=1:num_3; label3_y{i}='2'; end
for i=1:num_4; label4_y{i}='3'; end
for i=1:num_5; label5_y{i}='4'; end

% 用dividerand函数将每种故障类型的数据随机划分为4:1的比例，分别用作训练和测试

ybs=num_5-5;

trainInd_label1=(1:ybs);testInd_label1=(ybs+1:num_5);
trainInd_label2=(1:ybs);testInd_label2=(ybs+1:num_5);
trainInd_label3=(1:ybs);testInd_label3=(ybs+1:num_5);
trainInd_label4=(1:ybs);testInd_label4=(ybs+1:num_5);
trainInd_label5=(1:ybs);testInd_label5=(ybs+1:num_5);

% [trainInd_label1,~,testInd_label1]=dividerand(num_1,0.8,0,0.2);
% [trainInd_label2,~,testInd_label2]=dividerand(num_2,0.8,0,0.2);
% [trainInd_label3,~,testInd_label3]=dividerand(num_3,0.8,0,0.2);
% [trainInd_label4,~,testInd_label4]=dividerand(num_4,0.8,0,0.2);
% [trainInd_label5,~,testInd_label5]=dividerand(num_5,0.8,0,0.2);

% 构建每种故障类型的训练数据
xTrain_label1=label1_x_cell(trainInd_label1);
yTrain_label1=label1_y(trainInd_label1);

xTrain_label2=label2_x_cell(trainInd_label2);
yTrain_label2=label2_y(trainInd_label2);

xTrain_label3=label3_x_cell(trainInd_label3);
yTrain_label3=label3_y(trainInd_label3);

xTrain_label4=label4_x_cell(trainInd_label4);
yTrain_label4=label4_y(trainInd_label4);

xTrain_label5=label5_x_cell(trainInd_label5);
yTrain_label5=label5_y(trainInd_label5);


% 构建每种故障类型的测试数据
xTest_label1=label1_x_cell(testInd_label1);
yTest_label1=label1_y(testInd_label1);

xTest_label2=label2_x_cell(testInd_label2);
yTest_label2=label2_y(testInd_label2);

xTest_label3=label3_x_cell(testInd_label3);
yTest_label3=label3_y(testInd_label3);

xTest_label4=label4_x_cell(testInd_label4);
yTest_label4=label4_y(testInd_label4);

xTest_label5=label5_x_cell(testInd_label5);
yTest_label5=label5_y(testInd_label5);


% 将每种故障类型的数据整合，构建完整的训练集和测试集
xTrain=[xTrain_label1 xTrain_label2 xTrain_label3 xTrain_label4 xTrain_label5 ];
yTrain=[yTrain_label1; yTrain_label2; yTrain_label3; yTrain_label4; yTrain_label5];
num_train=size(xTrain,2);

xTest=[xTest_label1  xTest_label2  xTest_label3 xTest_label4 xTest_label5 ];
yTest=[yTest_label1; yTest_label2; yTest_label3; yTest_label4; yTest_label5];
num_test=size(xTest,2);

%================================================================================
%以下分别对每个样本，提取三种特征：1.瞬时频率，2.瞬时谱熵，3.小波包能量，
%上述三种特征后面会被送入分类器中进行分类，实验结果表明，将小波包能量作为特征，
%能够取得最高的分类精度

% 提取瞬时频率：用matlab的pspectrum对每个样本进行谱分解，再用instfreq函数计算瞬时频率
FreqResolu=25;
TimeResolu=0.012;
% the output of pspectrum 'p' contains an estimate of the short-term, time-localized power spectrum of x. 
% In this case, p is of size Nf × Nt, where Nf is the length of f and Nt is the length of t.
[p,f,t]=cellfun(@(x) pspectrum(x,fs,'TimeResolution',TimeResolu,'spectrogram'),xTrain,'UniformOutput', false);
instfreqTrain=cellfun(@(x,y,z) instfreq(x,y,z)', p,f,t,'UniformOutput',false);
[p,f,t]=cellfun(@(x) pspectrum(x,fs,'TimeResolution',TimeResolu,'spectrogram'),xTest,'UniformOutput', false);
instfreqTest=cellfun(@(x,y,z) instfreq(x,y,z)', p,f,t,'UniformOutput',false);

% 提取瞬时谱熵：用matlab的pspectrum对每个样本进行谱分解，再用pentropy函数计算瞬时频率
[p,f,t]=cellfun(@(x) pspectrum(x,fs,'TimeResolution',TimeResolu,'spectrogram'),xTrain,'UniformOutput', false);
pentropyTrain=cellfun(@(x,y,z) pentropy(x,y,z)', p,f,t,'UniformOutput',false);
[p,f,t]=cellfun(@(x) pspectrum(x,fs,'TimeResolution',TimeResolu,'spectrogram'),xTest,'UniformOutput', false);
pentropyTest=cellfun(@(x,y,z) pentropy(x,y,z)', p,f,t,'UniformOutput',false);

% 提取小波包能量
% num_level=5表示进行小波包五层分解，共获得2^5=32个值组成的特征向量。
num_level=5; 
index=0:1:2^num_level-1;
% wpdec为小波包分解函数
treeTrain=cellfun(@(x) wpdec(x,num_level,'dmey'), xTrain, 'UniformOutput', false);
treeTest=cellfun(@(x) wpdec(x,num_level,'dmey'), xTest, 'UniformOutput', false);
for i=1:num_train
    for j=1:length(index)
    	% wprcoef为小波系数重构函数
        reconstr_coef=wprcoef(treeTrain{i},[num_level,index(j)]);
        % 计算能量
        energy(j)=sum(reconstr_coef.^2);
    end
    energyTrain_doule(i,:)=energy;
end

energyTrain=num2cell(energyTrain_doule,2);
energyTrain=energyTrain';

for i=1:num_test
    for j=1:length(index)
        reconstr_coef=wprcoef(treeTest{i},[num_level,index(j)]);
        energy(j)=sum(reconstr_coef.^2);
    end
 energyTest_double(i,:)=energy;
end

energyTest=num2cell(energyTest_double,2);
energyTest=energyTest';

% ===============组装用于送入分类器的特征序列====================
% 下面的语句仅用了小波包能量作为输入特征
xTrainFeature=cellfun(@(x)[x], energyTrain', 'UniformOutput',false);
xTestFeature=cellfun(@(x)[x], energyTest', 'UniformOutput',false);
% 如果想用 瞬时谱熵和小波包能量三种特征作为输入，如下
% xTrainFeature=cellfun(@(x,y,z)[x;y;z],energyTrain',instfreqTrain', pentropyTrain', 'UniformOutput',false);
% xTestFeature=cellfun(@(x,y,z)[x;y;z], energyTest',instfreqTest',pentropyTest', 'UniformOutput',false);

% ============================数据标准化================================
XV=[xTrainFeature{:}];
mu=mean(XV,2);
sg=std(XV,[],2);

xTrainFeatureSD=xTrainFeature;
xTrainFeatureSD=cellfun(@(x)(x-mu)./sg, xTrainFeatureSD,'UniformOutput',false);

xTestFeatureSD=xTestFeature;
xTestFeatureSD=cellfun(@(x)(x-mu)./sg,xTestFeatureSD,'UniformOutput',false);

% =========================设计LSTM网络=================================
yTrain_categorical=categorical(yTrain);
numClasses=numel(categories(yTrain_categorical));
yTest_categorical=categorical(yTest);
sequenceInput=size(xTrainFeatureSD{1},1); % 如果选了3种特征作为数据，这里改为"3"

% 创建用于sequence-to-label分类的LSTM步骤如下：
% 1. 创建sequence input layer
% 2. 创建若干个LSTM layer
% 3. 创建一个fully connected layer
% 4. 创建一个softmax layer
% 5. 创建一个classification outputlayer
% 注意将sequence input layer的size设置为所包含的特征类别数，本例中，1或2或3，取决于你用了几种特征。fully connected layer的参数为分类数，本例中为8.
layers = [ ...
    sequenceInputLayer(sequenceInput)
     lstmLayer(256,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];


maxEpochs=550;%轮
miniBatchSize=64;%影响迭代次数  16-11 32-5 64-2
% 如果不想展示训练过程，
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'gpu',...
     'SequenceLength', 'longest',...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'plots','training-progress', ... 
    'Verbose',true);

% ======================训练网络=========================
net2 = trainNetwork(xTrainFeatureSD,yTrain_categorical,layers,options);
% ======================测试网路==========================
testPred2 = classify(net2,xTestFeatureSD);
% 打印混淆矩阵
plotconfusion(yTest_categorical',testPred2','LSTM神经网络')
