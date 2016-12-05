--th test_real_multi_data_getBatch.lua 100 'OUTPUTS/snapshot_train_real/snapshot_epoch_40.net' 'big_dataset.t7'
require 'nn'
require 'cunn'
math.randomseed(os.time())

require '../../help_funcs.lua'
data=require 'data'
require '../../model'
require 'cudnn'
require 'cunn';
local libs={}
        
libs['SpatialConvolution'] = cudnn.SpatialConvolution        
libs['SpatialMaxPooling'] = cudnn.SpatialMaxPooling
libs['ReLU'] = cudnn.ReLU

require 'gnuplot'

metrics = require 'metrics'


function test()
	local dataset=arg[3] or 'big_dataset.t7'
	data.load(nil,dataset)--TODO
	data.select('test')
	batchDim=tonumber(arg[1]) or 50
	print('testing '..batchDim..' pairs of images...')
	testBatches = data.getNbOfBatches(batchDim).test

	orig_inputs,labels=data.getBatch(1,'test')
	orig_inputs=orig_inputs:cuda()
	labels=labels:int()
	s={}
	for i=1,math.min(50,batchDim) do
		table.insert(s,image.toDisplayTensor(orig_inputs[i]))
	end
	images1=image.toDisplayTensor{input = s, padding=10,nrow=5}

	labels1=slice(labels:totable(),1,50,1)
	labels2=torch.IntTensor(labels1):resize(10,5)
	print(labels2)
	local output_im1='OUTPUTS/data_example1.png'
	image.save(output_im1,images1)
	saved_model=arg[2]
	local model
	if saved_model then
		print('loading saved model...')
		model=torch.load(saved_model)
	else
		model = build_model(libs)
	end
	local model=model:cuda()
	dists=model:forward(orig_inputs)
	dists=torch.exp(-dists)
	
	local roc_points, thresholds = metrics.roc.points(dists:double(), labels)
	local area = metrics.roc.area(roc_points)


	--gnuplot.plot(roc_points)



	results1=torch.totable(dists)
	results1=slice(results1,1,50,1)
	results2=torch.DoubleTensor(results1):resize(10,5)
	print(results2)
	print('\n')
	print('area under curve:'..area)
	print('\n')
	print('see images by: open1 '..output_im1)
end

test()
