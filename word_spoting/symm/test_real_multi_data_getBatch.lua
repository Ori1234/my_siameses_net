--th test_real_multi.lua 200 'OUTPUTS/snapshot_train_real/snapshot_epoch_5.net' '../../DATA/real_data_RGB/test/'
require 'nn'
require 'cunn'
-- th -i test_real_multi.lua <num_of_tests><saved model><data folder>
--th -i test_real_multi.lua 1000 OUTPUTS/snapshot_train_real/snapshot_epoch_5.net ../../DATA/real_data_RGB/
math.randomseed(os.time())

require '../../help_funcs.lua'
local num_of_test=arg[1] or 100 
saved_model=arg[2] or 'OUTPUTS/snapshot_train_real/snapshot_epoch_5.net'
local data_folder=arg[3] or '../../DATA/real_data_RGB/'

--F_test=preper_data_word_spoting(data_folder)
--print('word_spoting_folders test')
--folders_test=set_word_spoting_folders(F_test)

--print('prepered')
function test()
local folder1
local folder2



data=require 'data'
data.select('test')
--batchDim = 100 
batchDim=100
print('batch size is ...'..batchDim)
testBatches = data.getNbOfBatches(batchDim).test
print('testBatches: ', testBatches)




--do return end

--num_of_test=50
inputs={}
labels={}
--orig_inputs={}
print(type(tonumber(num_of_test)))
orig_inputs,labels=data.getbatch(1,'test')
--for i = 1, num_of_test do
--	local same=(math.random(1, 10) > 5)
--	local folder1,folder2=rand_staff_word_spoting(same,F_test,folders_test)
--	    
  --      local input=pair_word_spoting(folder1,folder2)


	--print(input:type())
--	label = same and 1 or -1
--	table.insert(inputs, image.toDisplayTensor(input))
	--table.insert(orig_inputs,input:cuda())
--	orig_inputs[i]=input:cuda()
--	table.insert(labels, label)
--end

images1=image.toDisplayTensor{input = slice(orig_inputs:totable(),1,50,1), padding=10,nrow=5}

labels1=slice(labels:totable(),1,50,1)
labels2=torch.IntTensor(labels1):resize(10,5)
print(labels2)
print(images1:size())
print(images1:type())

image.save('OUTPUTS/data_example1.png',images1)

--do return end
--local model=torch.load(saved_model)
require 'model'

local model2=model:cuda()
local dists=model2:forward(orig_inputs)
dists=torch.exp(-dists)
--do return end
metrics = require 'metrics'


local roc_points, thresholds = metrics.roc.points(dists:double(), torch.IntTensor(labels))
local area = metrics.roc.area(roc_points)

print('area under curve:'..area)
print('num of tests '..num_of_test)

require 'gnuplot'
gnuplot.plot(roc_points)



results1=torch.totable(dists)
results1=slice(results1,1,50,1)
results2=torch.DoubleTensor(results1):resize(10,5)
print(results2)
end

test()
