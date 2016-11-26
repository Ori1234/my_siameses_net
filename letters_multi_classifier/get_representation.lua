require 'cunn'
model_file=arg[1] or 'OUTPUTS/snapshot_train_real/snapshot_epoch_10.net'
data_to_represent=arg[2] or '../../../DATA/real_data_RGB/'

model=torch.load(model_file)


--list all images in dir



--create representation for each
for file in files do
	model:forward(im)
	rep=model:get(9).output:clone()
end
