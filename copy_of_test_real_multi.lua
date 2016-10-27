require 'nn'
require 'cunn'
--th test_real_multi.lua 500 snapshot_train_on_real_leave_out_some/snapshot_epoch_20.net real_data/test/
-- th -i test_real_multi.lua <num_of_tests><saved model><data folder>
-- th -i test_real_multi.lua 1000 snapshot_4_10/snapshot_epoch_23.net real_data/test/
math.randomseed(os.time())

require 'help_funcs.lua'
local num_of_test=arg[1] or 100 
saved_model=arg[2] or 'snapshot_4_10/snapshot_epoch_23.net'
local data_folder=arg[3] or 'real_data/test/'


preper_data_real_multi(data_folder)
print('prepered')

local folder1
local folder2

--do return end

--num_of_test=50
inputs={}
labels={}
--orig_inputs={}
orig_inputs=torch.CudaTensor(100,2,1,64,64)  --TODO de-hardcoded
for i = 1, num_of_test do
--   TODO 1) put here my data by 
--           1.0) choose letter x
	local same=(math.random(1, 10) > 5)
	letter,folder1,folder2=rand_staff_multi(same)
	input=pair_real(letter,folder1,folder2)
	label = same and 1 or -1
	table.insert(inputs, image.toDisplayTensor(input))
	--table.insert(orig_inputs,input:cuda())
	orig_inputs[i]=input
	table.insert(labels, label)
end

images1=image.toDisplayTensor{input = slice(inputs,1,50,1), padding=10,nrow=5}
image.save('data_example1.png',images1)

--do return end
local model=torch.load(saved_model)

--local model1=model:double()
model2=model:cuda()
dist=model2:forward(orig_inputs)
do return end
results={}
for i=1,#orig_inputs do
	--local dist=model1:forward(orig_inputs[i])
	local dist=model2:forward(orig_inputs[i])
	table.insert(results,torch.exp(-dist[1]))
	--print(dist[1])
	--print(torch.exp(-dist[1]))
	--print('\n')
	lab=labels[i]==1 and 1 or 0
	--print(string.format("%.0f, %.7f", labels[i],dist[1]))
--	--print(dist)
end
metrics = require 'metrics'


roc_points, thresholds = metrics.roc.points(torch.DoubleTensor(results), torch.IntTensor(labels))
area = metrics.roc.area(roc_points)

print('area under curve:'..area)
print('num of tests '..num_of_test)
--gfx.chart(roc_points)

require 'gnuplot'
gnuplot.plot(roc_points)




