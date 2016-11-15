--delete output_dir before calling
require 'image'
require 'data_augmentation.lua'
data_main_folder=arg[1] or '../../../DATA/real_data_RGB/train_not_augmented'
output_dir=arg[2] or '../../../DATA/real_data_RGB/train'

function augment_data(source_dir,output_dir)
	os.execute("mkdir -p " .. output_dir)
	for pic_path in io.popen("ls  "..source_dir.."/*.png"):lines() do
		my_crop(pic_path,output_dir)
	end
end


function my_crop(pic_path,output_dir)
	print(pic_path)
	pic_basename=basename(pic_path)
	im=image.load(pic_path)
	
	size=56
		
	image.save(output_dir..'/'..pic_basename,im)
	for j = 1, #CROP_POS_ALL do
            im1= zoomout(crop(im, CROP_POS_ALL[j], size))
	    image.save(output_dir..'/'..pic_basename..CROP_POS_ALL[j][1]..CROP_POS_ALL[j][2]..'.png',im1)
--            im1= zoomout(crop(im, CROP_POS28[j], size))
--	    image.save(output_dir..'/'..pic_basename..CROP_POS28[j][1]..CROP_POS28[j][2]..'_.png',im2)
         end
end

function basename(str)
	local name = string.gsub(str, "(.*/)(.*)", "%2")
	return name
end

---------------------------------
for subfolder in io.popen("ls "..data_main_folder):lines() do
	print(subfolder)
	augment_data(data_main_folder..'/'..subfolder,output_dir..'/'..subfolder)
end


