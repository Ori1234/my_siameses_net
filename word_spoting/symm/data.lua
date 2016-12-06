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

	
local my_load=function(data_path,t7_path)
	-- Loading the dataset to RAM --------------------------------------------------
	t7_path=t7_path or 'big_dataset.t7'
	if paths.filep(t7_path) then
		io.write('Loading whole data set. Please wait...'..t7_path); io.flush()
		dataset = torch.load(t7_path)
		print(' Done.')
	else
	   -- This script uses pubfig83.v1.tgz from http://vision.seas.harvard.edu/pubfig83/
	   -- split in train and test folders
	   -- each containing identities folders with images inside.
	   -- Format:
	   -- datasetRoot/{train,test}/<celebrityName>
	local datasetPaths = {}
	datasetPaths.base = data_path or  '../../../../DATA/BAVLI/some'
	print('###getting data from '..datasetPaths.base)
	for _, t in ipairs {'train', 'test'} do
        	print('Building ' .. t .. 'ing data set')

	      datasetPaths[t] = datasetPaths.base .. '/' .. t .. '/'
	      local identities = sys.ls(datasetPaths[t]):split('\n')
	      local dataSize = tonumber(sys.execute('find ' .. datasetPaths[t] .. ' -iname "*.jpg"| wc -l'))
	      dataSize=math.min(limit_datasize[t], dataSize)
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

	   io.write('Saving whole data set to disk...'..t7_path); io.flush()
	   torch.save(save_path, dataset)
	   print(' Done.')
	end
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
   local inputs = torch.Tensor(batchSize,2, 3, imageSide, imageSide)
   local sames= torch.Tensor(batchSize)

   -- Auxiliary variable
   local label 
   local offset = batchSize * (nb - 1)
   for i = 1, batchSize do
      local loc = shuffle[i + offset] -- original location in data set
      inputs[i][1]  = dataset[t].data [loc]
      local label = dataset[t].label[loc]
      local same=math.random(1000)>500
      sames[i]=same and 1 or -1
      
      local dataSize=dataset[t].data:size(1)
      local from=dataset[t].index[label][1]
--      local to= math.min(dataset[t].index[label][2],dataSize) 
      local to= dataset[t].index[label][2] 
      local index
      if same then
  	 if to==from then --single word  --choose another word that is not single
         	while from==to do
                       --print('from==to')
                       loc = math.random(dataSize) -- choose another word randomly from whole dataset
                       local label = dataset[t].label[loc]
                       from=dataset[t].index[label][1]
            --           to= math.min(dataset[t].index[label][2],dataSize)
      		       to= dataset[t].index[label][2] 
	        end
                inputs[i][1]  = dataset[t].data [loc]
	--         print('\n')
	 --print('to==from') 
	 end
	 index=math.random(from,to)
	 --else --try to choose another pic
		 --c=0
--		 while  index==loc and  c<batchSize do
	 while  index==loc do
		 index=math.random(from,to)
		 --c=c+1
	 end	 
      else
         index=math.random(dataSize)
	 local class=dataset[t].label[index]
--	 local c=0
--	 while c<batchSize and class==label do
	 while class==label do
		index=math.random(dataSize)
		class=dataset[t].label[index]
--		c=c+1
	 end
      end
      inputs[i][2]  = dataset[t].data[index]
   end
   return inputs, sames
end

-- Moves the batch to the GPU's RAM
local my_toCuda = function(batch)
   require 'cutorch'
--   for i in ipairs(batch) do batch[i] = batch[i]:cuda() end
   batch=batch:cuda()--don't work can't mutate
   print('not implemented')
end

--[[
-- Saves the embeddings emb for the nb batch of t (train/test) data set
local saveEmb = function(emb, nb, t, posEmb)
   local offset = batchSize * (nb - 1)
   for i = 1, batchSize do
      dataset[t].emb[shuffle[i + offset]--] = emb[i]:float()
   end
   if posEmb then pEmb = posEmb:float() end
end
--]]
local stats=function(t)
	local sum=0
	local single=0
	print('### word count for '..t)
	num_of_words=dataset[t].index:size(1)
	for i=1,num_of_words do
		local from=dataset[t].index[i][1]
		local to=dataset[t].index[i][2]
		local c=to-from+1
		sum=sum+c
		if c~=1 then
			print('word:'..i..' count '..c)
		else
			single=single+1
		end
	end
	print('words with 1 occurance: '..single)
	print('###total num of pics '..t..' '..sum)
	print('###num of different words '..t..' '..num_of_words)
end




-- Public functions ------------------------------------------------------------
return {
   load	          =my_load,
   select         = shuffleShuffle,
   getNbOfBatches = my_getNbOfBatches,
--   initEmbeddings = initEmbeddings,
   getBatch       = my_getBatch,
   toCuda         = my_toCuda,
   stats          = stats,
--   saveEmb        = saveEmb,
}

