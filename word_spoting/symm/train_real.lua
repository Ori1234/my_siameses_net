--no pretrain: th train_real.lua 100 /home/wolf/oriterne/BAVLI/outputs/
--pretrain : th train_real.lua 100 /see above/ -learning_rate 0.001 -weights OUTPUTS/<snapshot_train_synthetic>
require 'torch';
require 'nn';
require 'optim';
metrics = require 'metrics'
require 'image';
--require 'dataset';
require '../../model';
	-----------------------------------------------------------------------------
math.randomseed(os.time())
	--------------------- parse command line options ----------------------------
	-----------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:argument("-max_epochs", "maximum epochs")
cmd:argument("-data_folder", "/home/wolf1/oriterne/DATA/real_data_RGB")
cmd:text("Options")
cmd:option("-batch_size", 50, "batch size")
cmd:option("-learning_rate", 0.01, "learning_rate")
cmd:option("-momentum", 0.9, "momentum")
cmd:option("-snapshot_dir", "OUTPUTS/snapshot_train_real/", "snapshot directory")
cmd:option("-snapshot_epoch", 5, "snapshot after how many iterations?")
cmd:option("-gpu", true, "use gpu")
cmd:option("-weights", "", "pretrained model to begin training from")
cmd:option("-log", "OUTPUTS/output log file train real")

params = cmd:parse(arg)

-----------------------------------------------------------------------------
--------------------- Initialize Variable -----------------------------------
-----------------------------------------------------------------------------
if params.log ~= "" then
	cmd:log(params.log, params)
	cmd:addTime('torch_benchmarks','%F %T')
	print("setting log file as "..params.log)
end

libs = {}
run_on_cuda = false
if params.gpu then
	print("using cudnn")
	require 'cudnn'
	require 'cunn';
	libs['SpatialConvolution'] = cudnn.SpatialConvolution
	libs['SpatialMaxPooling'] = cudnn.SpatialMaxPooling
	libs['ReLU'] = cudnn.ReLU
	--torch.setdefaulttensortype('torch.CudaTensor')   see https://github.com/torch/image/issues/24
	run_on_cuda = true
else
	libs['SpatialConvolution'] = nn.SpatialConvolution
	libs['SpatialMaxPooling'] = nn.SpatialMaxPooling
	libs['ReLU'] = nn.ReLU
	torch.setdefaulttensortype('torch.FloatTensor')
end

epoch = 0
batch_size = params.batch_size
--Load model and criterion

if params.weights ~= "" then
	print("loading model from pretained weights in file "..params.weights)
	
	margin = 1
 	criterion = nn.HingeEmbeddingCriterion(margin)	
	model = torch.load(params.weights)
else
	model = build_model(libs)
end

if run_on_cuda then
	model = model:cuda()
	criterion=criterion:cuda()
end


-----------------------------------------------------------------------------
--------------------- Training Function -------------------------------------
-----------------------------------------------------------------------------
-- retrieve a view (same memory) of the parameters and gradients of these (wrt loss) of the model (Global)
parameters, grad_parameters = model:getParameters();

function train()
	local saved_criterion = false;
	for i = 1, params.max_epochs do
		--add random shuffling here
		--print('train'..i)
		train_one_epoch()
		--print('test'..i)
		test_one_epoch()
		to_plot=true
		if to_plot then
		      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
		      testLogger:style{['% mean class accuracy (test set)'] = '-'}
		      rocLogger:style{['% auc (test set)'] = '-'}
		      trainLogger:plot()
		      testLogger:plot()
		      rocLogger:plot()
   		end

		if params.snapshot_epoch > 0 and (epoch % params.snapshot_epoch) == 0 then -- epoch is global (gotta love lua :p)
			local filename = paths.concat(params.snapshot_dir, "snapshot_epoch_" .. epoch .. ".net")
			os.execute('mkdir -p ' .. sys.dirname(filename))
			torch.save(filename, model)        
			--must save std, mean and criterion?
			if not saved_criterion then
				local criterion_filename = paths.concat(params.snapshot_dir, "_criterion.net")
				torch.save(criterion_filename, criterion)
				local dataset_attributes_filename = paths.concat(params.snapshot_dir, "_dataset.params")
				dataset_attributes = {}
				dataset_attributes.mean = data.mean
				dataset_attributes.std = data.std
				torch.save(dataset_attributes_filename, dataset_attributes)
			end
		end
	end
end
	

function test_one_epoch()
	local epoch_size=100
	test_size=10  ---batch size
	local inputs = torch.CudaTensor(test_size,2,3,64,64)
	local labels = torch.CudaTensor(test_size)
	local labels_append
	local dists_append
--	print(dists_append:size())
--	print(labels_append:size())
	for mini_batch_start=1,epoch_size,test_size do
		error=0
		for i= 1,test_size do --for each mini-batch
			local same=(math.random(1, 10) > 5)
			local folder1,folder2=rand_staff_word_spoting(same,F_test,folders_test)
			local input=pair_word_spoting(folder1,folder2)
			input=input:cuda()
			local label = same and 1 or -1
			inputs[i]=input	
			labels[i]=label
		end
		local dists = model:forward(inputs)
--		print(dists:size())
	
		local err = criterion:forward(dists, labels)
		dists=torch.exp(-dists)
		if not dists_append then
			dists_append=dists:double()
			labels_append=labels:int()
		else	
			dists_append:cat(dists:double())
			labels_append:cat(labels:int())
		end
		error=error+err
	end
	num_of_batchs=epoch_size/test_size
	print('epoch '..(epoch-1)..' test loss: '..error/num_of_batchs)	
	local roc_points, thresholds = metrics.roc.points(dists_append, labels_append)
	local area = metrics.roc.area(roc_points)

	testLogger:add{['% mean class accuracy (test set)'] =error/num_of_batchs}
	rocLogger:add{['AUC'] = area}

	print('area under curve:'..area)
--	print('num of tests '..test_size)

end

function train_one_epoch()
	local train_epoch=100
	local time = sys.clock()
	--train one epoch of the dataset
	local errors=0
	local lr=params.learning_rate
	for mini_batch_start = 1,train_epoch, batch_size do --for each mini-batch
		local inputs = torch.CudaTensor(batch_size,2,3,64,64)
		local labels = torch.CudaTensor(batch_size)
		--create a mini_batch
		local mini_batch_stop=math.min(mini_batch_start + batch_size - 1, train_epoch)

		for i=1,batch_size do 
			local same=(math.random(1, 10) > 5)
			local folder1,folder2=rand_staff_word_spoting(same,F_train,folders_train)
			local input=pair_word_spoting(folder1,folder2)
			input=input:cuda()
			local label = same and 1 or -1
			inputs[i]=input	
			labels[i]=label
		end
		--create a closure to evaluate df/dX where x are the model parameters at a given point
		--and df/dx is the gradient of the loss wrt to thes parameters

		local func_eval = function(x)
			--update the model parameters (copy x in to parameters)
			if x ~= parameters then
				parameters:copy(x) 
			end
			grad_parameters:zero() --reset gradients

			local avg_error = 1 -- the average error of all criterion outs
			--evaluate for complete mini_batch
			local outputs = model:forward(inputs)
			local err = criterion:forward(outputs, labels)
			--estimate dLoss/dW
			local dloss_dout = criterion:backward(outputs, labels)
			model:backward(inputs, dloss_dout)
			errors=errors+err		
			return err, grad_parameters
		end
		
		if epoch > 50 then
			lr = params.learning_rate/10	
		end
		if epoch > 100 then
			lr = params.learning_rate/100
		end
		if epoch > 150 then
			lr = params.learning_rate/1000
		end
		config = {learningRate = lr , momentum = params.momentum}
		--This function updates the global parameters variable (which is a view on the models parameters)
		optim.sgd(func_eval, parameters, config)
	
		xlua.progress(mini_batch_start, train_epoch)--display progress
	end
	-- time taken
	print(lr)
	time = sys.clock() - time
	--print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/5000)*1000) .. 'ms')
	--print(errors)
	num_of_batchs=train_epoch/batch_size--100
	print('epoch '..epoch..' loss: '..errors/num_of_batchs)
	trainLogger:add{['% mean class accuracy (train set)'] = errors/num_of_batchs}

	epoch = epoch + 1	
end
	-----------------------------------------------------------------------------
	--------------------- Training Function -------------------------------------
	-----------------------------------------------------------------------------
print("loading dataset...")
print("dataset loaded")
require '../../help_funcs.lua'

-- log results to files
trainLogger = optim.Logger(paths.concat('OUTPUTS', 'train.log'))
testLogger = optim.Logger(paths.concat('OUTPUTS', 'test.log'))
rocLogger=optim.Logger(paths.concat('OUTPUTS', 'roc.log'))

data_folder=params.data_folder
print('data folder '..data_folder)
F_train=preper_data_word_spoting(data_folder..'train/')
--print(F_train)
print('word_spoting_folders train')
folders_train=set_word_spoting_folders(F_train)
print('done')
F_test=preper_data_word_spoting(data_folder..'test/')
--print(F_test)
print('word_spoting_folders test')
folders_test=set_word_spoting_folders(F_test)
test_one_epoch()
--print('this')
train()

