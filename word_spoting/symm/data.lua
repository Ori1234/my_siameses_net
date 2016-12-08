--------------------------------------------------------------------------------
-- Loading dataset to ram and sorting triplets
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr 15
--------------------------------------------------------------------------------
limit_datasize={train=100,test=50}

require 'image'
require 'sys'
require 'xlua'
require '../../help_funcs.lua'

torch.setdefaulttensortype('torch.FloatTensor')

dataset = {}--make local
local imageSide =64 

my_load_pair=function(data_path,t7_path)
        t7_path=t7_path or 'big_dataset123.t7'
        if paths.filep(t7_path) then
                io.write('my_load_pairs: Loading whole data set. Please wait...'..t7_path); io.flush()
                dataset = torch.load(t7_path)
                print(' Done.')
        else
        local datasetPaths = {}
        datasetPaths.base = data_path or  '../../../../DATA/BAVLI/some'
        print('###getting data from '..datasetPaths.base)
	
	for _, t in ipairs {'train', 'test'} do
       		print('Building ' .. t .. 'ing data set')


--		f1,f2,s1,s2=rand_staff_word_spoting(false,paths1,word_spoting_folders)
--		i,l=pair_word_spoting(f1,f2,s1,s2)

	
	     	datasetPaths[t] = datasetPaths.base .. '/' .. t .. '/'
		
		--local paths1
		local pre_saved_path=t..'_saved_ls_tree_table.t7'
		if paths.filep(pre_saved_path) then
			paths1=torch.load(pre_saved_path)  
		else
			print('prepering from '..datasetPaths[t])
			paths1=preper_data_word_spoting(datasetPaths[t])
			torch.save(pre_saved_path,paths1)
		end

		word_spoting_folders=set_word_spoting_folders(paths1)
	      	


--		local identities = sys.ls(datasetPaths[t]):split('\n')
	      	local dataSize = tonumber(sys.execute('find ' .. datasetPaths[t] .. ' -iname "*.jpg"| wc -l'))
	      	dataSize=math.min(limit_datasize[t], dataSize)
	      	dataset[t] = {
        	 	data = torch.Tensor(dataSize,2, 3, imageSide, imageSide),
        	 	label = torch.Tensor(dataSize),
--	         	index = torch.Tensor(#identities, 2),
      		}

	      	local count = 0
	      	for id, idName in ipairs(word_spoting_folders) do
		        --dataset[t].index[id][1] = count + 1
			if count<dataSize then --needed because no break statement in lua
		      	   for _, img in ipairs(paths1[idName]) do
									--      

		         	 count = count + 1
				 if count<=dataSize then
		--		    print(count)
					--1)draw same/not same
			            --local original = image.load(paths.concat(datasetPaths[t], idName, img))
			            local original = image.load(img)
				    local im1=image.scale(original,imageSide,imageSide)
				    local same=math.random(1000)>500
					--1.1) if same
					--	draw another file from THIS identity (HOW?)
					if same then
						while #paths1[idName]  <2 do
							idName=word_spoting_folders[math.random(#word_spoting_folders)]
							print(idName)
							print(#paths1[idName])
						end
					print('\n')
					idName2=idName
					else
						idName2=word_spoting_folders[math.random(#word_spoting_folders)]
						while idName2==idName do
							idName2=word_spoting_folders[math.random(#word_spoting_folders)]
						end
					end
					im2_folder=paths1[idName2]
					--print(#im2_folder)
					--print(idName)
					im2_path=im2_folder[math.random(#im2_folder)]
					while im2_path==img do
						im2_path=im2_folder[math.random(#im2_folder)]
						print('same imag')
					end
					print('\n')
					--print(im2_path)
					original2= image.load(im2_path)
					
				    local im2=image.scale(original2,imageSide,imageSide)
						--1.2) if not same
			            xlua.progress(count, dataSize)

			            dataset[t].data[count][1] = im1
			            dataset[t].data[count][2] = im2
				
			            dataset[t].label[count] = string.levenshtein_4_files(idName,idName2)-- this is OK since go to folder all1 and not test/train - which I areased TATIK from... 
--				    print('dist '..dataset[t].label[count])
			         end
	--	         dataset[t].index[id][2] = count
		         collectgarbage()
		      end
		   end
		 end
	   end

	   io.write('Saving whole data set to disk...'..t7_path); io.flush()
--	   torch.save(t7_path, dataset)
	   print(' Done.')
	end
	



end
	
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
	datasetPaths.base = data_path or  '../../../../DATA/BAVLI'
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

local my_normalize1 = function()
--see https://github.com/torch/tutorials/blob/master/2_supervised/1_data.lua

end

local my_normalize = function(t,kind,mean_,std_)
	print('normalizing data. kind: '..kind..' t:'..t..' following mean and std if given')
	print(mean_)
	print(std_)
	if kind=='global' and t=='test' then
		if not mean_ or not std_ then
			error('##for global normalization on test need to supply mean and std from train data')
		end
	end
	if kind=='perImage' then
		size=dataset[t].data:size(1)
		local mean = dataset[t].data:view(size, -1):mean(2)
		local std = dataset[t].data:view(size, -1):std(2, true)	 
		for i=1,size do        
--			print(i)
			dataset[t].data[i]:add(-mean[i][1])         
			if std[i][1] > 0 then            
			--	tensor:select(2, i):mul(1/std[1][i])         --WHAT IS THIS?
				dataset[t].data[i]:mul(1/std[i][1])
			end       
		end 
		return mean, std
	elseif kind=='global' then
	      local std = std_ or dataset[t].data:std()
	      local mean = mean_ or dataset[t].data:mean()
	      dataset[t].data:add(-mean)
	      dataset[t].data:mul(1/std)
	      return mean, std
	else 
		print('unsoported regularization '..kind)
	end
	--[[

---see mnist in TUTORIALS in /home/wolf1/oriterne/
   function dataset:normalize(mean_, std_)
      local mean = mean_ or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end 
      end 
      return mean, std 
   end 

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std 
   end 



--]]

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

function load_pairs_data(t,class_subset)--not usefull. most pairs are negative
  --local file = torch.load(filename, 'ascii')
  my_load(nil,nil) --args?
  
  local indices_in_subset = {}

  local all_data = dataset[t].data:type(torch.getdefaulttensortype())
  local all_labels = dataset[t].label
  
  for i = 1, all_labels:size()[1] do
    if #class_subset == 0 or class_subset[all_labels[i]] ~= nil then
      table.insert(indices_in_subset, i)
    end 
  end 
  
  local data = torch.Tensor(#indices_in_subset, all_data:size()[2], all_data:size()[3], all_data:size()[4])
  local labels = torch.Tensor(#indices_in_subset)

  for i = 1,#indices_in_subset do
    data[i] = all_data[indices_in_subset[i]]
    labels[i] = all_labels[indices_in_subset[i]]
  end 

  local std = data:std()
  local mean = data:mean()
  data:add(-mean);
  data:mul(1.0/std);
    
  --torch.setdefaulttensortype('torch.FloatTensor')

  shuffle = torch.randperm(data:size(1))
--  torch.setdefaulttensortype('torch.CudaTensor')
--  shuffle=shuffle:cuda()

  max_index = data:size(1)
  if max_index % 2 ~= 0 then
    max_index = max_index - 1 
  end 


  -- now we make the pairs (tensor of size (30000,2,1,32,32) for training data)
  paired_data = torch.Tensor(max_index/2, 2, data:size(2), data:size(3), data:size(4))
  paired_data_labels = torch.Tensor(max_index/2)
  index = 1

  for i = 1,max_index,2 do
    paired_data[index][1] = data[shuffle[i]]:clone()
    paired_data[index][2] = data[shuffle[i + 1]]:clone()
    if labels[shuffle[i]] == labels[shuffle[i+1]] then
      paired_data_labels[index] = 1
    else
      paired_data_labels[index] = -1
    end
    index = index + 1
  end

  local dataset = {}
  dataset.data = paired_data
  dataset.labels = paired_data_labels
  dataset.std = std
  dataset.mean = mean

  function dataset:size()
    return dataset.data:size(1)
  end

  local class_count = 0
  local classes = {}
  for i=1, dataset.labels:size(1) do
    if classes[dataset.labels[i]] == nil then
      class_count = class_count + 1
      table.insert(classes, dataset.labels[i])
    end
  end

  dataset.class_count = class_count

  --The dataset has to be indexable by the [] operator so this next bit handles that
  setmetatable(dataset, {__index = function(self, index)
                        local input = self.data[index]
                        local label = self.labels[index]
                        local example = {input, label}
                        return example
                        end })

  return dataset
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
   normalize      = my_normalize,
}

