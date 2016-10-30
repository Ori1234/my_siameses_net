--th test_syntetic.lua 500 OUTPUTS/snapshot_train_synthetic/snapshot_epoch_5.net ../../DATA/synthetic_RGB/
math.randomseed(os.time())
require 'cudnn'
require 'cunn'
require 'help_funcs.lua'


--do return end

local num_of_test=arg[1] or 1000 
local saved_model=arg[2] or 'snapshot_exclude_fonts/snapshot_epoch_23.net'
local synth_data = arg[3] or '../../DATA/synthetic_RGB/'
preper_data_syntetic(synth_data)

inputs={}
labels={}
orig_inputs=torch.CudaTensor(tonumber(num_of_test),2,3,64,64) 
for i = 1, num_of_test do
--   TODO 1) put here my data by 
--           1.0) choose letter x
	local same=(math.random(1, 10) > 5)
	local letter=math.random(1488,1514)
	local input=pair_syntetic(letter,same)
	--print(input:type())	
	label = same and 1 or -1
	table.insert(inputs, image.toDisplayTensor(input))
	orig_inputs[i]=input
	table.insert(labels, label)
end
--do return end
images1=image.toDisplayTensor{input = slice(inputs,1,50,1), padding=10,nrow=5}
print(images1:size())
print(images1:double():type())
--torch.setdefaulttensortype('torch.FloatTensor')
image.save('OUTPUTS/data_example1_synth.png',images1)

--do return end
local model=torch.load(saved_model)

local model2=model:cuda()
local dists=model2:forward(orig_inputs)
dists=torch.exp(-dists)

metrics = require 'metrics'

local roc_points, thresholds = metrics.roc.points(dists:double(), torch.IntTensor(labels))
local area = metrics.roc.area(roc_points)

print('area under curve:'..area)
print('num of tests '..num_of_test)
--gfx.chart(roc_points)

require 'gnuplot'
gnuplot.plot(roc_points)




