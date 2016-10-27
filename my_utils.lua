--run with qlua (my alias)
-- dofile my_utils.lua
-- im_transform(false,3)
--or im_transform('image file name',3)
--the number is th noise
--or im_transform(falsel,0,0,{0,0})
import 'image'
--image_path='/home/wolf1/oriterne/fonts/results_scaled_preserve_ratio/results_scaled_preserve_ratio/'
image_path='./'
function im_transform(fileName,noise,rotate,translate)
	if cudnn then
		torch.setdefaulttensortype('torch.FloatTensor')
	end
	local im
	local noise=math.random(0,noise)
	local translate=translate or {math.random(-2,2),math.random(-2,2)}
	local rotate=rotate or math.random()/4*math.random(-1,1)
	local fileName= fileName or '_1514-Yehuda_CLM-normal-normal.png'
        fileName=image_path..fileName
	if type(fileName)=='string' then
		im=image.load(fileName)
	else
		im=fileName  --if image object is given
	end
	--image.display(im)
	local w = im:size(3)
	local h = im:size(2)
	patch_w = math.random(1,w/4)
	for i=1,noise do		
		local patch_h = math.random(1,h/4)
		local x=math.random(1,w-patch_w)
		local y=math.random(1,h-patch_h)
		im[{{},{y,y+patch_h},{x,x+patch_w}}]=1
	end
	im=image.translate(im:add(-1):div(-1),translate[1],translate[2])
	im=image.rotate(im,rotate)
	im:add(-1):div(-1)	
	--image.display(im)
	if cudnn then
		torch.setdefaulttensortype('torch.CudaTensor')
		im=im:cuda()
	end
	return im
end


