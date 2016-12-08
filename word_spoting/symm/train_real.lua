--no pretrain: th train_real.lua 100 'testing 123'
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
cmd:argument("-Title", "for results email")
cmd:text("Options")
cmd:option("-data_folder", "../../../../DATA/real_data_RGB")
cmd:option("-t7",'big_dataset.t7')
cmd:option("-batch_size", 200, "batch size")
cmd:option("-learning_rate", 0.01, "initial learning_rate")
cmd:option("-momentum", 0.9, "momentum")
cmd:option("-snapshot_dir", "OUTPUTS/snapshot_train_real/", "snapshot directory")
cmd:option("-snapshot_epoch", 20, "snapshot after how many iterations?")
cmd:option("-gpu", true, "use gpu")
cmd:option("-weights", "", "pretrained model to begin training from")
cmd:option("-log", "OUTPUTS/output_log_file")
cmd:option("-output_dir", "OUTPUTS/")


params = cmd:parse(arg)

-----------------------------------------------------------------------------
--------------------- Initialize Variable -----------------------------------
-----------------------------------------------------------------------------
if params.log ~= "" then
	cmd:log(params.log, params)
	cmd:addTime('torch_benchmarks','%F %T')
	print("setting log file as "..params.log)
end
local excel_doc='https://docs.google.com/spreadsheets/d/1d4i3zT-eQkuxZQfOjOjtMhfbrIEyXvwuudhYyDXYBtE/edit#gid=0'
print('for experiment details see: '..excel_doc)
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
if not paths.dirp(params.snapshot_dir) then
	os.execute('mkdir -p '..params.snapshot_dir) --else returns and error from parseargs
end
if not paths.dirp(params.output_dir) then
	os.execute('mkdir -p '..params.output_dir) --else returns and error from parseargs
end


--batch_size = params.batch_size
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
local max_auc=0
function train()
	local saved_criterion = false;
	for epoch = 0, params.max_epochs do
		trn_loss=train_one_epoch(epoch+1)
		tst_loss,auc=test_one_epoch(epoch+1)
		
		if auc>max_auc then
			max_auc=auc
		end	
	  	lossLogger:add{trn_loss,tst_loss}
		rocLogger:add{auc}
		lossLogger:plot()
		rocLogger:plot()
		
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
		epoch = epoch + 1	
	end
end
	

function test_one_epoch(epochNb)
	data.select('test')-- don't need to shuffle test
	--local num_of_batches=testBatches
	--local test_size=batchDim
	local labels_append
	local dists_append


--	print(dists_append:size())
--	print(labels_append:size())
--	for mini_batch_start=1,epoch_size,test_size do
	--print(testBatches)
	errors=0
	for batchNb=1,testBatches do
		local inputs, labels = data.getBatch(batchNb,'test')
		inputs=inputs:cuda()
		labels=labels:cuda()
	--	for i= 1,test_size do --for each mini-batch
	--		local same=(math.random(1, 10) > 5)
	--		local folder1,folder2=rand_staff_word_spoting(same,F_test,folders_test)
	--		local input=pair_word_spoting(folder1,folder2)
	--		input=input:cuda()
	--		local label = same and 1 or -1
	--		inputs[i]=input	
	--		labels[i]=label
	--	end
		local dists = model:forward(inputs)
--		print(dists:size())
		--print('\n')	
		local err = criterion:forward(dists, labels)
		dists=torch.exp(-dists)
		if not dists_append then
			dists_append=dists:double():clone()
			labels_append=labels:int():clone()
		else	
			dists_append:cat(dists:double():clone())
			labels_append:cat(labels:int():clone())
		end
		errors=errors+err
                xlua.progress(batchNb,testBatches)--display progress
	end
	print('epoch '..epochNb..' test loss: '..errors/testBatches)	
	local roc_points, thresholds = metrics.roc.points(dists_append, labels_append)
	local area = metrics.roc.area(roc_points)

--	testLogger:add{['test loss'] =errors/testBatches}
--	rocLogger:add{['AUC'] = area}

	print('area under curve:'..area)
	return errors/testBatches,area
end

function train_one_epoch(epoch)
	local time = sys.clock()
	data.select('train')
	--train one epoch of the dataset
	local errors=0
	local lr=params.learning_rate
	--for mini_batch_start = 1,train_epoch, batch_size do --for each mini-batch
	for batchNb=1,trainBatches do
		local inputs, labels=data.getBatch(batchNb,'train')
		inputs=inputs:cuda()
		labels=labels:cuda()
--		for i=1,batch_size do 
--			local same=(math.random(1, 10) > 5)
--			local folder1,folder2=rand_staff_word_spoting(same,F_train,folders_train)
--			local input=pair_word_spoting(folder1,folder2)
--			input=input:cuda()
--			local label = same and 1 or -1
--			inputs[i]=input	
--			labels[i]=label
--		end
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
	
		xlua.progress(batchNb, trainBatches)--display progress
	end
	-- time taken
	print(lr)
	time = sys.clock() - time
	--print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/5000)*1000) .. 'ms')
	--print(errors)
	print('epoch '..epoch..' loss: '..errors/trainBatches)
--	trainLogger:add{['% mean class accuracy (train set)'] = errors/trainBatches}
return errors/trainBatches
end
	-----------------------------------------------------------------------------
	--------------------- Training Function -------------------------------------
	-----------------------------------------------------------------------------
require '../../help_funcs.lua'

-- log results to files
lossLogger = optim.Logger(paths.concat(params.output_dir, 'loss.log'))
lossLogger:setNames{'Train loss','Test loss'}
lossLogger:style{'-','-'}
rocLogger=optim.Logger(paths.concat(params.output_dir, 'roc.log'))
rocLogger:setNames{'AUC'}
rocLogger:style{'-'}
data=require 'data'
data.load(data_folder,params.t7)

data.normalize('train','perImage')
data.normalize('test','perImage')

data.select('train')
data.select('test')
batchDim=params.batch_size
print('batch size is ...'..batchDim)
trainBatches = data.getNbOfBatches(batchDim).train
testBatches = data.getNbOfBatches(batchDim).test
print('trainBatches: ', trainBatches)
print('testBatches: ', testBatches)

test_one_epoch(0)

train()

local Title='experiment: '..params.Title..' : '..max_auc
line='mutt -s "'..Title..'" oriterne@post.tau.ac.il -i "'..
	params.log..'" -a '..
	params.output_dir..'roc.log.eps '..
	params.output_dir..'roc.log '..
	params.output_dir..'loss.log.eps '..
	params.output_dir..'loss.log < /dev/null'
print(line)
os.execute(line)

