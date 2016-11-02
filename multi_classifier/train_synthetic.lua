--th train_synthetic.lua 100 ../../DATA/synthetic_RGB
require 'torch';
require 'nn';
require 'optim';
require 'image';
--require 'dataset';
require 'model';
	-----------------------------------------------------------------------------
math.randomseed(os.time())
	--------------------- parse command line options ----------------------------
	-----------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:argument("-max_epochs", "maximum epochs")
cmd:argument("-data_folder", "/home/wolf1/oriterne/DATA/synthetic_RGB")
cmd:text("Options")
cmd:option("-batch_size", 50, "batch size")
cmd:option("-learning_rate", 0.01, "learning_rate")
cmd:option("-momentum", 0.9, "momentum")
cmd:option("-snapshot_dir", "OUTPUTS/snapshot_train_synthetic/", "snapshot directory")
cmd:option("-snapshot_epoch", 5, "snapshot after how many iterations?")
cmd:option("-gpu", true, "use gpu")
cmd:option("-weights", "", "pretrained model to begin training from")
cmd:option("-log", "OUTPUTS/output log file")

params = cmd:parse(arg)

-----------------------------------------------------------------------------
--------------------- Initialize Variable -----------------------------------
-----------------------------------------------------------------------------
if params.log ~= "" then

--	cmd:log(params.log, params)
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
	--torch.setdefaulttensortype('torch.CudaTensor')
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

function train(letters)
	local saved_criterion = false;
	print(type(params.max_epochs))
	for i = 1, params.max_epochs do
		--add random shuffling here
		train_one_epoch(letters)

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
	


function train_one_epoch(letters)
	local time = sys.clock()
	--train one epoch of the dataset
	local errors=0
	local lr=params.learning_rate
	for mini_batch_start = 1,5000, batch_size do --for each mini-batch
		local inputs = torch.CudaTensor(batch_size,2,3,64,64)
		local labels = torch.CudaTensor(batch_size)
		--create a mini_batch
		local mini_batch_stop=math.min(mini_batch_start + batch_size - 1, 5000)

		for i = 1, batch_size do 
			--    local input = dataset[i][1]:clone() -- the tensor containing two images     
			local letter=letters[math.random(#letters)]
			local input,label=single_syntetic(letter)
			input=input:cuda()
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
			--print(outputs)
			local err = criterion:forward(outputs, labels)
			--estimate dLoss/dW
			--print(err)
			local dloss_dout = criterion:backward(outputs, labels)
			model:backward(inputs, dloss_dout)
			--grad_parameters:div(#inputs);
			--avg_error = avg_error / #inputs;
			errors=errors+err
			return err, grad_parameters
		end
		
		if epoch > 50 then
			lr = params.learning_rate/10	
		end
		if epoch > 100 then
			lr = params.learning_rate/10
		end
		if epoch > 150 then
			lr = params.learning_rate/10
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
print("loading dataset...")
print("dataset loaded")
local letters={}
--	local test_percent=0.2
local test_percent=-1 --all letters for training
for i=1488,1514 do
--	if math.random(1,100)>100*test_percent then
		table.insert(letters,i)
--	else
--		print('excluded letter '..i)
	end
--end
--print('\n')
--print(letters)

--local total_num_of_letters=1514-1488+1
--print('excluded '..(total_num_of_letters-#letters)*1.0/total_num_of_letters)
require 'help_funcs.lua'
data_folder=params.data_folder
print('data folder '..data_folder)
preper_data_syntetic(data_folder)
print('data is prepered')
train(letters)



function single_synthetic(letter)
	require '../my_utils.lua'
	a=choose_random_font(letter)
	noise=5
	input=im_transform(a,noise)
        mean=input:mean()
        std=input:std()
        input:add(-mean)
        input:mul(1.0/std)
        return input
end

