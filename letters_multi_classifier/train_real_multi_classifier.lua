--no pretrain: th train_real_multi_classifier.lua 100 ../../../DATA/real_data_RGB/
--pretrain : 
require 'torch';
require 'nn';
require 'optim';
require 'image';
--require 'dataset';
metrics = require 'metrics'
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


-----------------------------------------------------------------------------
--------------------- Training Function -------------------------------------
-----------------------------------------------------------------------------
-- retrieve a view (same memory) of the parameters and gradients of these (wrt loss) of the model (Global)

function train()
	local saved_criterion = false;
	for i = 1, params.max_epochs do
		--add random shuffling here
		train_one_epoch()
		test_one_epoch()
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
	
local test_size=500
function test_one_epoch()
	local inputs = torch.CudaTensor(test_size,3,64,64)
	local labels = torch.CudaTensor(test_size)
	--create a mini_batch
	for i=1,test_size do
		local folder=classes[math.random(#classes)]
		folder_files=F_test[folder]
		letter_num=math.random(#letters)
		local letter=letters[letter_num] --letters is created by help_funcs.lua
			--print(folder)
			--print(letter)
			--print(folder_files[letter])
		while #folder_files[letter]==0 do
			letter=letters[math.random(#letters)] --letters is created by help_funcs.lua
		end

		local input=single_real(letter,folder_files)   
		input=input:cuda()
		local label = letter_num
	 	--print(label)		
		--print(input:size())
		inputs[i]=input	
		labels[i]=label
	end
        local preds = model:forward(inputs)
        local err = criterion:forward(preds, labels)
              -- confusion:
	      for i = 1,test_size do
        	 confusion:add(preds[i], labels[i])
	      end

	   print(confusion)
	--   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
	   confusion:zero()





	print('epoch '..(epoch-1)..' test loss: '..err) 
--[[
        dists=torch.exp(-dists)
        local roc_points, thresholds = metrics.roc.points(dists:double(), labels:int())
        local area = metrics.roc.area(roc_points)
]]	
--	roc_curve(test_size) 
end

function roc_curve(num_of_test)
	inputs={}
	labels={}
	results={}
	for i = 1, num_of_test do
        	local same=(math.random(1, 10) > 5)
	        letter1,letter2,folder1,folder2=rand_staff_multii_2_letters(same,F_test) --TODO not implemented
	        input=pair_real_2_letters(letter1,letter2,folder1,folder2) --TODO not implemented
	        --print(same)
	        --im1=input[1]
	        --im2=input[2]
        	local output=model:forward(input:cuda())
	        --print(output)
		
	        local a=model:get(9).output:clone()
	        --print(a:size())
	        --model:forward(im2)
	        --local b=model:get(9).output:clone()
        	d=distances(a,2)
	        --print(d)
	        --print('\n')
		same_letter=true --to do need to change
	        label = same_letter and 1 or -1
	        table.insert(inputs, image.toDisplayTensor(input))
	        table.insert(labels, label)
	        table.insert(results,torch.exp(-d[2][1]))
end
local roc_points, thresholds = metrics.roc.points(torch.DoubleTensor(results), torch.IntTensor(labels))
local area = metrics.roc.area(roc_points)

print('area under curve:'..area)
print('num of tests '..num_of_test)

end



function train_one_epoch()
	local time = sys.clock()
	--train one epoch of the dataset
	local errors=0
	local lr=params.learning_rate
	for mini_batch_start = 1,5000, batch_size do --for each mini-batch
		local inputs = torch.CudaTensor(batch_size,3,64,64)
		local labels = torch.CudaTensor(batch_size)
		--create a mini_batch
		local mini_batch_stop=math.min(mini_batch_start + batch_size - 1, 5000)

		for i=1,batch_size do
			local folder=classes[math.random(#classes)]
			folder_files=F_train[folder]
			letter_num=math.random(#letters)
			local letter=letters[letter_num] --letters is created by help_funcs.lua
			--print(folder)
			--print(letter)
			--print(folder_files[letter])
			while #folder_files[letter]==0 do
				
				letter=letters[math.random(#letters)] --letters is created by help_funcs.lua
			end

			local input=single_real(letter,folder_files)   --F1 is also created by help_funcs.lua TODO encapsulate
			input=input:cuda()
			local label = letter_num --TODO find out index of folder in classes
		 	--print(label)		
			--print(input:size())
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

			local avg_error = 0 -- the average error of all criterion outs
			--evaluate for complete mini_batch
			local outputs = model:forward(inputs)
			local err = criterion:forward(outputs, labels)
			local dloss_dout = criterion:backward(outputs, labels)
			model:backward(inputs, dloss_dout)
			errors=errors+err
			         -- update confusion
		         for i = 1,batch_size do
		            confusion:add(outputs[i], labels[i])
		         end 

		
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
	
		xlua.progress(mini_batch_start, 5000)--display progress
	end
	-- time taken
	print(lr)
	time = sys.clock() - time
	--print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/5000)*1000) .. 'ms')
	--print(errors)
   -- print confusion matrix
	   print(confusion)
--	   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	   confusion:zero()

	print('epoch '..epoch..' loss: '..errors/100.0)
	epoch = epoch + 1	
end
	-----------------------------------------------------------------------------
	--------------------- Training Function -------------------------------------
	-----------------------------------------------------------------------------

require '../help_funcs.lua'
data_folder=params.data_folder
print('data folder '..data_folder)
F_train=preper_data_real_multi(data_folder..'train/')
F_test=preper_data_real_multi(data_folder..'test/')
classes=get_keys(F_train)
--classes=letters
require 'model';

if params.weights ~= "" then
	print("loading model from pretained weights in file "..params.weights)
	
	margin = 1
 	criterion = nn.HingeEmbeddingCriterion(margin)	
	model = torch.load(params.weights)
else
	model = build_model(libs,letters)
end

if run_on_cuda then
	model = model:cuda()
	criterion=criterion:cuda()
end

confusion = optim.ConfusionMatrix(letters)
parameters, grad_parameters = model:getParameters();
--do return end
test_one_epoch()
train()

