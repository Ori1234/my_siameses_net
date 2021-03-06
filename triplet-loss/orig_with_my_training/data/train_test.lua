torch.manualSeed(0)
math.randomseed(0)

data = require 'data'
--data.select('train')
metrics = require 'metrics'
require 'gnuplot'
require 'optim'


---------------------HELP FUNCS
local distances = function(vectors,norm)
   -- args:
   local X = vectors
   local norm = norm or 2
   local N,D = X:size(1),X:size(2)

   -- compute L2 distances:
   local distances
   if norm == 2 then
      local X2 = X:clone():cmul(X):sum(2)
      distances = (X*X:t()*-2) + X2:expand(N,N) + X2:reshape(1,N):expand(N,N)
      distances:abs():sqrt()
   elseif norm == 1 then
      distances = X.new(N,N)
      local tmp = X.new(N,D)
      for i = 1,N do
         local x = X[i]:clone():reshape(1,D):expand(N,D)
         tmp[{}] = X 
         local dist = tmp:add(-1,x):abs():sum(2):squeeze()
         distances[i] = dist
      end 
   else
      error('norm must be 1 or 2')
   end 
   
   -- return dists
   return distances
end




------------------





-- Initialise embeddings
embDim =100 
data.initEmbeddings(embDim)

-- Get train and test number of batches
batchDim = 500
trainBatches = data.getNbOfBatches(batchDim).train
testBatches = data.getNbOfBatches(batchDim).test
print('trainBatches: ', trainBatches)
print('testBatches: ', testBatches)


-- Send batch to cuda
collectgarbage()

-- Saving embeddings
--data.saveEmb(torch.randn(batchDim, embDim), 1, 'train')
--print('Embedding saved for fast training')

require '../model'
model:cuda()
parameters, grad_parameters = model:getParameters();

opt={}
batchNb=10
opt.negPopMod='soft-neg'

loss = nn.TripletEmbeddingCriterion(0.2)
loss:cuda()

epochsNb=200
for i = 1, epochsNb do
	error=0
	-- Shuffling training data
	data.select('train') -- or 'test'
	
	print(colour.green('Epoch ' .. i)) 
	for batchNb=1,trainBatches do
	    -- Fetch triplet
	--   print(batchNb)
	   x = data.getBatch(batchNb, 'train', 'soft-neg1')
	   data.toCuda(x) 
	   -- Update embeddings
	   local anchorsEmb = model.modules[1]:forward(x[1])
	   local positiveEmb = model.modules[2]:forward(x[2])
	   data.saveEmb(anchorsEmb, batchNb, 'train', positiveEmb)
	   -- Fetch new triplet
	   inputs = data.getBatch(batchNb, 'train', 'soft-neg2')
	   data.toCuda(inputs)	
--	   predict = model:forward(x)
	--        print(predict)
--	   err = loss:forward(predict)
	   --print(err)
--	   error=error+err
--	   errGrad = loss:backward(predict)
--	   model:zeroGradParameters()
--	   model:backward(x,errGrad)
--	   model:updateParameters(0.01)
                local func_eval = function(x)
                        --update the model parameters (copy x in to parameters)
                        if x ~= parameters then
                                parameters:copy(x) 
                        end
                        grad_parameters:zero() --reset gradients

--    ???whats this                    local avg_error = 1 -- the average error of all criterion outs
                        --evaluate for complete mini_batch
                        --local outputs = model:forward(inputs)
			local predict = model:forward(inputs)

                        --local err = criterion:forward(outputs, labels)
	   		local err = loss:forward(predict)
                        --estimate dLoss/dW
--                        local dloss_dout = criterion:backward(outputs, labels)
      			local errGrad = loss:backward(predict)

--                        model:backward(inputs, dloss_dout)
	 		model:backward(inputs,errGrad)

                        error=error+err               
                        return err, grad_parameters
                end


	   config = {learningRate = 0.01 ,
		     momentum =0.9,
		     learningRateDecay = 5e-7}
	   optim.sgd(func_eval,parameters,config)
	end
	print(colour.red('train loss: '), error/trainBatches)
	

	--TESTING
	error=0
	roc_curve=0
	data.select('test')
	for batchNb=1,testBatches do
	    -- Fetch triplet
--	   x = data.getBatch(batchNb, 'test', 'soft-neg1')
--	   data.toCuda(x) 
	   -- Update embeddings
--	   local anchorsEmb = model.modules[1]:forward(x[1])
--	   local positiveEmb = model.modules[2]:forward(x[2])
--	   data.saveEmb(anchorsEmb, batchNb, 'test', positiveEmb)
	   -- Fetch new triplet
--	   x = data.getBatch(batchNb, 'test', 'soft-neg2')

 	   x=  data.getBatch(batchNb, 'test')
	   data.toCuda(x)
	   predict = model:forward(x)
	   err = loss:forward(predict)
	   --print(err)
	   error=error+err

	   labels={}
	   results={}
	   --TODO roc curve
	   y = predict
           for j=1,batchDim do
	   	same=math.random(1000)>500
		   a=torch.cat(y[1][j],y[3][j],2)
		   a=a:transpose(1,2)
		   if same then
		   	a[2]=y[2][j]
		   end
		d=distances(a,2)
--		print(same)
--	        print(d)
		label = same and 1 or -1
--	        table.insert(inputs, image.toDisplayTensor(input))
	        table.insert(labels, label)
        	table.insert(results,torch.exp(-d[2][1]))
	   end
       	   local roc_points, thresholds = metrics.roc.points(torch.DoubleTensor(results), torch.IntTensor(labels))
	   local area = metrics.roc.area(roc_points)
	   roc_curve=roc_curve+area


--	   print('area under curve:'..area)

	   if false then 
		gnuplot.plot(roc_points)
	   end
	
	end	
	print(colour.red('test loss: '), error/testBatches)
	print('roc_average '..roc_curve/testBatches)
end
toSave=true
if toSave then
	torch.save(os.time()..'trained_model.net',model)
end







do return end











--if opt.negPopMode == 'soft-neg' then
   -- Fetch triplet
   x = data.getBatch(batchNb, 'train', 'soft-neg1')

   if opt.cuda then data.toCuda(x) end
   -- Update embeddings
   local anchorsEmb = model.modules[1]:forward(x[1])
   local positiveEmb = model.modules[2]:forward(x[2])
   data.saveEmb(anchorsEmb, batchNb, 'train', positiveEmb)
   -- Fetch new triplet
   x = data.getBatch(batchNb, 'train', 'soft-neg2')

   y = model:forward(x)
   print(y)

--[[
elseif opt.negPopMode == 'hard-neg' then
   -- Fetch triplet
   x = data.getBatch(batchNb, 'train')
   if opt.cuda then data.toCuda(x) end
   -- Update embeddings
   local anchorsEmb = model.modules[1]:forward(x[1])
   data.saveEmb(anchorsEmb, batchNb, 'train')
   -- Fetch new triplet
   x = data.getBatch(batchNb, 'train')
elseif opt.negPopMode == 'fast-hard-neg' then
   -- Fetch triplet
   x = data.getBatch(batchNb, 'train')
end
if opt.cuda then data.toCuda(x) end

y = model:forward(x)

if opt.negPopMode == 'fast-hard-neg' then
   data.saveEmb(y[1], batch, 'train')
end
--]]
