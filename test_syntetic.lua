-- th test_syntetic.lua 500 snapshot_synthetic_leave_some_fonts_out/snapshot_epoch_5.net synthetic_data/test/
-- th -i scrip_for_visualization_with_roc_synt.lua <num_of_tests>
-- th -i scrip_for_visualization_with_roc_synt.lua 1000 
math.randomseed(os.time())

require 'help_funcs.lua'

preper_data_syntetic(arg[3])

--do return end

local num_of_test=arg[1] or 1000 
--num_of_test=50
inputs={}
labels={}
orig_inputs={}
for i = 1, num_of_test do
--   TODO 1) put here my data by 
--           1.0) choose letter x
	local same=(math.random(1, 10) > 5)
	local letter=math.random(1488,1514)
	local input=pair_syntetic(letter,same)
	--print(input:size())
	label = same and 1 or -1
	--print(label)
	table.insert(inputs, image.toDisplayTensor(input))
	table.insert(orig_inputs,input)
	table.insert(labels, label)
end

images1=image.toDisplayTensor{input = slice(inputs,1,50,1), padding=10,nrow=5}
image.save('data_example1.png',images1)

--do return end
require 'cudnn'
require 'cunn'
saved_model=arg[2] or 'snapshot_exclude_fonts/snapshot_epoch_23.net'
local model=torch.load(saved_model)

local model1=model:double()
local model2=model:cuda()

results={}
for i=1,#orig_inputs do
	--local dist=model1:forward(orig_inputs[i])
	local dist=model2:forward(orig_inputs[i]:cuda())
	table.insert(results,-dist[1])
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




