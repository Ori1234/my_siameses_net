--no pretrain: th train_real_multi_classifier.lua 100 ../../../DATA/real_data_RGB/train/
--pretrain : 
require 'torch';
require 'nn';
require 'optim';
require 'image';
--require 'dataset';
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
			folder_files=F1[folder]
			local letter=letters[math.random(#letters)] --letters is created by help_funcs.lua
			--print(folder)
			--print(letter)
			--print(folder_files[letter])
			while #folder_files[letter]==0 do
				
				letter=letters[math.random(#letters)] --letters is created by help_funcs.lua
			end

			local input=single_real(letter,folder_files)   --F1 is also created by help_funcs.lua TODO encapsulate
			input=input:cuda()
			local label = index_of(classes,folder) --TODO find out index of folder in classes
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
	print('epoch '..epoch..' loss: '..errors/100.0)
	epoch = epoch + 1	
end
	-----------------------------------------------------------------------------
	--------------------- Training Function -------------------------------------
	-----------------------------------------------------------------------------

require '../help_funcs.lua'
--require 'more_help_funcs.lua'

function get_keys(table)
	local keyset={}
	local n=0
	for k,v in pairs(F1) do
		n=n+1
		keyset[n]=k
	end
	return keyset
end
function single_real(l,folder)
	im_path1=folder[l][math.random(#folder[l])]
	local input=image.load(im_path1)
        mean=input:mean()
        std=input:std()
        input:add(-mean)
        input:mul(1.0/std)

	return input 
end
index_of = function( t, object )
    if "table" == type( t ) then
        for i = 1, #t do
            if object == t[i] then
                return i
            end
        end
        return -1
    else
            error("table.indexOf expects table for first argument, " .. type(t) .. " given")
    end
end

data_folder=params.data_folder
print('data folder '..data_folder)
preper_data_real_multi(data_folder)
classes=get_keys(F1)
require 'model';

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

parameters, grad_parameters = model:getParameters();
--do return end
train()
