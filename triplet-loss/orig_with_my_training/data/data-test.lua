torch.manualSeed(0)
math.randomseed(0)

data = require 'data'

-- Shuffling training data
data.select('train') -- or 'test'

-- Initialise embeddings
embDim =100 
data.initEmbeddings(embDim)

-- Get train and test number of batches
batchDim = 500
trainBatches = data.getNbOfBatches(batchDim).train
print('trainBatches: ', trainBatches)

-- Get training batch nb 1
batch1 = data.getBatch(1, 'train')
print('batch = {aImg, pImg, nImg}: ', batch1)

-- Send batch to cuda
collectgarbage()
data.toCuda(batch1)
print('Batch sent to GPU memory')

-- Saving embeddings
data.saveEmb(torch.randn(batchDim, embDim), 1, 'train')
print('Embedding saved for fast training')

require '../model'

opt={}
batchNb=10
opt.negPopMod='soft-neg'
-- Snippet from the training script, provided for reference only
--do return end
--[remove[


loss = nn.TripletEmbeddingCriterion(0.2)
epochsNb=40
for i = 1, epochsNb do
error=0
-- Shuffling training data
data.select('train') -- or 'test'

print(colour.green('Epoch ' .. i)) 
for batchNb=1,trainBatches do
    -- Fetch triplet
--   print(batchNb)
   x = data.getBatch(batchNb, 'train', 'soft-neg2')

   if opt.cuda then data.toCuda(x) end
   -- Update embeddings
   local anchorsEmb = model.modules[1]:forward(x[1])
   local positiveEmb = model.modules[2]:forward(x[2])
   data.saveEmb(anchorsEmb, batchNb, 'train', positiveEmb)
   -- Fetch new triplet
   x = data.getBatch(batchNb, 'train', 'soft-neg2')

   predict = model:forward(x)
--        print(predict)
   err = loss:forward(predict)
   print(err)
   error=error+err
   errGrad = loss:backward(predict)
   model:zeroGradParameters()
   model:backward({aImgs, pImgs, nImgs}, errGrad)
   model:updateParameters(0.01)





end
print(colour.red('loss: '), error/trainBatches)
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
