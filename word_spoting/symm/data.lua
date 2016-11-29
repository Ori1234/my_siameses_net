--------------------------------------------------------------------------------
-- Loading dataset to ram and sorting triplets
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr 15
--------------------------------------------------------------------------------
limit_datasize={train=100,test=50}

require 'image'
require 'sys'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

dataset = {}
local imageSide =64 


-- Loading the dataset to RAM --------------------------------------------------
save_path='dataset.t7'
if paths.filep(save_path) then
   io.write('Loading whole data set. Please wait...'); io.flush()
   dataset = torch.load(save_path)
   print(' Done.')
else
   -- This script uses pubfig83.v1.tgz from http://vision.seas.harvard.edu/pubfig83/
   -- split in train and test folders
   -- each containing identities folders with images inside.
   -- Format:
   -- datasetRoot/{train,test}/<celebrityName>
   local datasetPaths = {}
   datasetPaths.base = '../../../../DATA/BAVLI'

   for _, t in ipairs {'train', 'test'} do
      print('Building ' .. t .. 'ing data set')

      datasetPaths[t] = datasetPaths.base .. '/' .. t .. '/'
      local identities = sys.ls(datasetPaths[t]):split('\n')
      local dataSize = tonumber(sys.execute('find ' .. datasetPaths[t] .. ' -iname "*.jpg"| wc -l'))
      dataSize=limit_datasize[t] or dataSize
      dataset[t] = {
         data = torch.Tensor(dataSize, 3, imageSide, imageSide),
         label = torch.Tensor(dataSize),
         index = torch.Tensor(#identities, 2),
      }

      local count = 0
      for id, idName in ipairs(identities) do
         dataset[t].index[id][1] = count + 1
	 if count<dataSize then
         for _, img in ipairs(sys.ls(datasetPaths[t] .. idName):split('\n')) do
            count = count + 1
	    if count<=dataSize then
--		    print(count)
	            xlua.progress(count, dataSize)
	            -- print(count, paths.concat(datasetPaths[t], idName, img))
	            local original = image.load(paths.concat(datasetPaths[t], idName, img))
	            local h = original:size(2)
	            local w = original:size(3)
	            local m = math.min(h, w)
	            local y = math.floor((h - m) / 2)
	            local x = math.floor((w - m) / 2)
	            dataset[t].data[count] = image.scale(
	--               original[{ {}, {y + 1, y + m}, {x + 1, x + m} }],
			original,
	               imageSide, imageSide
	            )
	            dataset[t].label[count] = id
	         end
	         dataset[t].index[id][2] = count
	         collectgarbage()
	      end
	 end
	 end
   end

   io.write('Saving whole data set to disk...'..save_path); io.flush()
   torch.save(save_path, dataset)
   print(' Done.')
end


-- Private functions -----------------------------------------------------------
-- Training shuffle
local shuffle

-- New index table for t: train/test
local shuffleShuffle = function(t)
   shuffle = torch.randperm(dataset[t].data:size(1))
end


-- Get nb of (train and test) batches gives the batch size
local nbOfBatches = {}
local batchSize, pImgs
local my_getNbOfBatches = function(bS)
   batchSize = bS
   for _, t in ipairs {'train', 'test'} do
      nbOfBatches[t] = math.floor(dataset[t].data:size(1) / batchSize)
   end
  -- pImgs = torch.Tensor(batchSize, 3, imageSide, imageSide)
   return nbOfBatches
end


-- Initialise the (train and test) embeddings
local embSize
local initEmbeddings = function(eS)
   embSize = eS
   for _, t in ipairs {'train', 'test'} do
      dataset[t].emb = torch.randn(dataset[t].data:size(1), embSize)
      dataset[t].emb = dataset[t].emb:cdiv(dataset[t].emb:norm(2, 2):repeatTensor(1, embSize))
   end
   print('Training and testing embeddings initialised with size ' .. embSize)
end

-- Get batch number nb for the t (train/test) dataset
-- <mode> by default is 'hard-neg' but can be set to 'soft-neg'
local pEmb
local my_getBatch = function(nb, t )

   -- Main varialbles
--   local aImgs1 = torch.Tensor(batchSize, 3, imageSide, imageSide)
--   local aImgs2 = torch.Tensor(batchSize, 3, imageSide, imageSide)
   local inputs = torch.Tensor(batchSize,2, 3, imageSide, imageSide)
--   local emb = torch.Tensor(batchSize, embSize)
   local sames= torch.Tensor(batchSize)

   -- Auxiliary variable
   local label 
   local offset = batchSize * (nb - 1)

   -- Populating anchor and positive images batches
   for i = 1, batchSize do
      local loc = shuffle[i + offset] -- original location in data set
      inputs[i][1]  = dataset[t].data [loc]
--      emb[i]    = dataset[t].emb  [loc]
      local label = dataset[t].label[loc]
  --    if mode ~= 'soft-neg2' then
      local same=math.random(1000)>500
      sames[i]=same and 1 or -1
      
      local dataSize=dataset[t].data:size(1)
      local from=dataset[t].index[label][1]
      local to= math.min(dataset[t].index[label][2],dataSize)
      local index
--	print(same)
      if same then
	 index=math.random(from,to)
      else
--	 print('datasize'..dataSize)
         index=math.random(dataSize)
	 local class=dataset[t].label[index]
	 local c=0
	 if c<batchSize and class==label then
		index=math.random(dataSize)
		class=dataset[t].label[index]
		c=c+1
	--	print('c '..c)
	 end
      end
  --    print('index '..index)
      inputs[i][2]  = dataset[t].data[index]
   
   end
   return {inputs, sames}
end


-- Get batch number nb for the t (train/test) dataset
-- <mode> by default is 'hard-neg' but can be set to 'soft-neg'
local pEmb
local getBatch = function(nb, t, mode, epoch)

   -- Main varialbles
   local aImgs = torch.Tensor(batchSize, 3, imageSide, imageSide)
   local nImgs = torch.Tensor(batchSize, 3, imageSide, imageSide)
   local emb = torch.Tensor(batchSize, embSize)

   -- Auxiliary variable
   local labels = torch.Tensor(batchSize)
   local offset = batchSize * (nb - 1)

   -- Populating anchor and positive images batches
   for i = 1, batchSize do
      local loc = shuffle[i + offset] -- original location in data set
      aImgs[i]  = dataset[t].data [loc]
      emb[i]    = dataset[t].emb  [loc]
      labels[i] = dataset[t].label[loc]
      if mode ~= 'soft-neg2' then
         pImgs[i]  = dataset[t].data [math.random(
            dataset[t].index[labels[i]][1],
            dataset[t].index[labels[i]][2]
         )]
      end
   end

   -- Populating negative images batche
   local mode = mode or 'hard-neg'
   if mode == 'hard-neg' then
      for i = 1, batchSize do
         local diff = emb - emb[{ {i} }]:expandAs(emb)
         local norms = diff:norm(2, 2):squeeze()
         norms[labels:eq(labels[i])] = norms:max()
         local _, nIdx = norms:min(1) -- closest n-emb to a-emb
         nImgs[i] = aImgs[nIdx[1]]
      end
   elseif mode == 'soft-neg1' then
      -- do nothing
   elseif mode == 'soft-neg2' then
      for i = 1, batchSize do
         local diff = emb - emb[{ {i} }]:expandAs(emb)
         local norms = diff:norm(2, 2):squeeze()
         norms = norms - torch.Tensor(batchSize):fill((emb[i]-pEmb[i]):norm())
         norms[labels:eq(labels[i])] = norms:max()
         norms[norms:lt(0)] = norms:max()
         local _, nIdx = norms:min(1) -- closest n-emb to a-emb
         nImgs[i] = aImgs[nIdx[1]]
      end
   else error('Negative populating <mode> not recognised!')
   end

   return {aImgs, pImgs, nImgs}

end


-- Moves the batch to the GPU's RAM
local toCuda = function(batch)
   require 'cutorch'
   for i in ipairs(batch) do batch[i] = batch[i]:cuda() end
end


-- Saves the embeddings emb for the nb batch of t (train/test) data set
local saveEmb = function(emb, nb, t, posEmb)
   local offset = batchSize * (nb - 1)
   for i = 1, batchSize do
      dataset[t].emb[shuffle[i + offset]] = emb[i]:float()
   end
   if posEmb then pEmb = posEmb:float() end
end


-- Public functions ------------------------------------------------------------
return {
   select         = shuffleShuffle,
   getNbOfBatches = my_getNbOfBatches,
--   initEmbeddings = initEmbeddings,
   getBatch       = my_getBatch,
   toCuda         = toCuda,
--   saveEmb        = saveEmb,
}

