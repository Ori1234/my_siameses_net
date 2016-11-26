--th test_real_multi_classifier.lua 200 'OUTPUTS/snapshot_train_real/snapshot_epoch_10.net' '../../../DATA/real_data_RGB/test/'
require 'nn'
require 'cunn'
-- th -i test_real_multi.lua <num_of_tests><saved model><data folder>
--th -i test_real_multi.lua 1000 OUTPUTS/snapshot_train_real/snapshot_epoch_5.net ../../DATA/real_data_RGB/
math.randomseed(os.time())

require '../help_funcs.lua'
local num_of_test=arg[1] or 100 
saved_model=arg[2] or 'OUTPUTS/snapshot_train_real/snapshot_epoch_10.net'
local data_folder=arg[3] or '../../../DATA/real_data_RGB/test/'



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




preper_data_real_multi(data_folder)
print('prepered')

local folder1
local folder2

--do return end

--num_of_test=50
inputs={}
labels={}
results={}


local model=torch.load(saved_model):cuda()

for i = 1, num_of_test do
	local same=(math.random(1, 10) > 5)
	letter,folder1,folder2=rand_staff_multi(same)
	input=pair_real(letter,folder1,folder2)
	--print(same)
	--im1=input[1]
	--im2=input[2]
	local output=model:forward(input:cuda())
	--print(output)
	
	local a=model:get(9).output:clone()
	--print(a:size())
	--model:forward(im2)
	--local b=model:get(9).output:clone()
	d=distances(a,2)
	--print(d)
	--print('\n')
	label = same and 1 or -1
	table.insert(inputs, image.toDisplayTensor(input))
	table.insert(labels, label)	
	table.insert(results,torch.exp(-d[2][1]))
end

images1=image.toDisplayTensor{input = slice(inputs,1,50,1), padding=10,nrow=5}
labels1=slice(labels,1,50,1)
labels2=torch.IntTensor(labels1):resize(10,5)
print(labels2)
results1=slice(results,1,50,1)
results2=torch.DoubleTensor(results1):resize(10,5)
print(results2)
--print(images1:size())
--print(images1:type())

image.save('OUTPUTS/data_example1.png',images1)

--do return end

--local dists=model2:forward(orig_inputs)
--dists=torch.exp(-dists)
--do return end
metrics = require 'metrics'


--local roc_points, thresholds = metrics.roc.points(dists:double(), torch.IntTensor(labels))
local roc_points, thresholds = metrics.roc.points(torch.DoubleTensor(results), torch.IntTensor(labels))
local area = metrics.roc.area(roc_points)

print('area under curve:'..area)
print('num of tests '..num_of_test)

require 'gnuplot'
gnuplot.plot(roc_points)



