require 'nn'
--from mnist tuturial
function build_model(libs)
	model = nn.Sequential()


	model:add(nn.SpatialConvolutionMM(3, 32, 5, 5)) 
	model:add(nn.Tanh())
	model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1)) 
	-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
	model:add(nn.SpatialConvolutionMM(32, 64, 5, 5)) 
	model:add(nn.Tanh())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 
	-- stage 3 : standard 2-layer MLP:
--	model:add(nn.Reshape(64*3*3))
	model:add(nn.Reshape(64*8*8))
	model:add(nn.Linear(64*8*8, 200))
	model:add(nn.Tanh())
	model:add(nn.Linear(200, #classes))
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()


	return model
end


