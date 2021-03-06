local info= debug.getinfo(1,'S');
file_name=string.gsub(info.source, "(.*/)(.*)", "%2")
--print(file_name)

--F1={}
--F_train={}
--F_test={}
--local F2

--returns a table that contains a list of images path for each letter 1488 to 1514 that are found in [dir]
function preper_data_syntetic(dir)
	F1={}
	for letter=1488,1514 do
		F1[letter]={}
		for f in io.popen("ls "..dir.."/"..letter.."*.png"):lines() do
			table.insert(F1[letter],f)
		end
	end
	print(count_data(F1))
	return F1
end


function count_data(tbl)
	local sum=0
	for k,v in pairs(tbl) do
		if #v>0 then
	--		print('letter:'..k,'count:'..#v)
			sum=sum+#v
		end
	end
	return sum
end


function count_data_multi(tbl)
	local sum=0
	print('DATA COUNT')
	for k,v in pairs(tbl) do
		print('###IN SCROLL: '..k)
		local count=count_data(v)
		print('------TOTAL: '..count)
		sum=sum+count
	end
	print('TOTAL: '..sum)	
end


--slice table
function slice(tbl, first, last, step)
	local sliced = {}
	for i = first or 1, last or #tbl, step or 1 do
		sliced[#sliced+1] = tbl[i]
	end
	return sliced
end



--default letter names as I work with. see matlab script in nova to rename files to meet convention
letters={
 'alef','bet', 'gimel', 'dalet', 'he', 'vav', 'zain', 'khet', 'tet', 'yod', 'kaf', 'lamed', 'mem', 'nun', 'samekh', 'ayin', 'pe', 'tsade', 'qof', 'resh', 'shin','tav'
}
reverse_letters={}
for k,v in pairs(letters) do
	reverse_letters[v]=k
end


-- table.indexOf( array, object ) returns the index
-- of object in array. Returns 'nil' if not in array.
table.indexOf = function( t, object )
	local result

	if "table" == type( t ) then
		for i=1,#t do
			if object == t[i] then
				result = i
				break
			end
		end
	end

	return result
end
function preper_data_word_spoting(data_main_folder)
	F1={}
	for f in io.popen("ls "..data_main_folder):lines() do
		io.write('.')
		F1[f]={}
		for pic in io.popen("ls  "..data_main_folder..f..'/'.."*.jpg"):lines() do
				--print(pic)
				--print(f)
				table.insert(F1[f],pic)
		end
	end
	print(count_data(F1))
	return F1
end


function preper_data_real_multi(data_main_folder)
	F1={}
	for f in io.popen("ls "..data_main_folder):lines() do
		F1[f]={}
		for i=1,#letters do
			l=letters[i]
			F1[f][l]={}
			for pic in io.popen("ls  "..data_main_folder..f..'/'..l.."*.png"):lines() do
				table.insert(F1[f][l],pic)
			end
		end
	end
	count_data_multi(F1)
	return F1
end

function pair_syntetic(letter,same)
	require 'my_utils.lua'	
	a=choose_random_font(letter)
        if not same then
        	b=choose_random_font(letter)
                while b==a do
			b=choose_random_font(letter)
                end
        else
                b=a
        end
        noise=1
	im1=im_transform(a,noise)
        im2=im_transform(b,noise)

        local input=torch.Tensor(2,im1:size(1),im1:size(2),im1:size(3))
        input[1]=im1
        input[2]=im2
        --TODO normalize over all data - need to prepare data in advance                
        local mean=input:mean()
        local std=input:std()
        input:add(-mean)
        input:mul(1.0/std)	
	return input
end

function choose_random_font(letter)
	--print(letter)
        return F1[letter][math.random(#F1[letter])]
end

--for syntetic test
function rand_staff(same)
 	local folder1
	local folder2

        local l=letters[math.random(#letters)]
        --print(l)
        while #F1[l]==0 or #F2[l]==0 do
                l=letters[math.random(#letters)]
        end
        --           1.1) random same/not same font
        if not same then
                folder1=F1
                folder2=F2
                if (math.random(1,10)>5) then
                        folder1=F2
                        folder2=F1
                end
        else
                folder1=F1
                if (math.random(1,10)>5) then
                        folder1=F2
                end
                folder2=folder1
        end
        while same and #folder1[l]<2 do
                l=letters[math.random(#letters)]
        end
	return l,folder1,folder2
end

function set_word_spoting_folders(F1)
	word_spoting_folders={}	
	for k,v in pairs(F1) do
		if #F1[k]~=0 then
		table.insert(word_spoting_folders,k)
		end	
	end
	return word_spoting_folders
end
	
--for word spoing test
function rand_staff_word_spoting(same,F1,word_spoting_folders)
 	local folder1
	local folder2
--	print(word_spoting_folders)
       	s1=word_spoting_folders[math.random(#word_spoting_folders)]
	if same then
		while #F1[s1]<2  do
			s1=word_spoting_folders[math.random(#word_spoting_folders)]
	        end	
		s2=s1
	else
		s2=word_spoting_folders[math.random(#word_spoting_folders)]
		while s1==s2 do
			s2=word_spoting_folders[math.random(#word_spoting_folders)]
		--	print('loop')
	        end
	end
	return F1[s1],F1[s2],s1,s2
end



--for real test
function rand_staff_multi(same,F1)
 	local folder1
	local folder2

        local l=letters[math.random(#letters)]
        folders={}
	for k,v in pairs(F1) do
		table.insert(folders,k)
	end
	s1=folders[math.random(#folders)]
	if same then
		while #F1[s1][l]<2  do
       		         l=letters[math.random(#letters)]
	        end	
		s2=s1
	else
		s2=folders[math.random(#folders)]
		while s1==s2 do
			s2=folders[math.random(#folders)]
	        end
		while #F1[s1][l]==0 or #F1[s2][l]==0 do
	                l=letters[math.random(#letters)]
        	end
        end
	return l,F1[s1],F1[s2]
end

require 'edit_distance.lua'
function pair_word_spoting(folder1,folder2,s1,s2)
	local dist=string.levenshtein_4_files(s1,s2)

        im_path1=folder1[math.random(#folder1)]

        im_path2=folder2[math.random(#folder2)]
	
        while im_path1==im_path2 do--only happen if same folder
                im_path2=folder2[math.random(#folder2)]		
        end
	local bool1,im1=pcall(image.load,im_path1)
        local bool2,im2=pcall(image.load,im_path2)
	while not  bool1 or not bool2 do
	
	       	im_path1=folder1[math.random(#folder1)]

        	im_path2=folder2[math.random(#folder2)]

        	while im_path1==im_path2 do
                	im_path2=folder2[math.random(#folder2)]
	        end
		bool1,im1=pcall(image.load,im_path1)
	        bool2,im2=pcall(image.load,im_path2)
		
	end
	local width=64
	local height=64
	im1=image.scale(im1, width, height)--'bilinear' 4 atg default is
	im2=image.scale(im2, width, height)--'bilinear' 4 atg default is
	local input=torch.Tensor(2,im1:size(1),im1:size(2),im1:size(3))
	input[1]=im1 --TODO normalize each image seperatly in order to not chnage from last run
        input[2]=im2
	mean=input:mean()
        std=input:std()
        input:add(-mean)
        input:mul(1.0/std)
	return input,dist
end




require 'image'
function pair_real(l,folder1,folder2)
        im_path1=folder1[l][math.random(#folder1[l])]

        im_path2=folder2[l][math.random(#folder2[l])]

        while im_path1==im_path2 do
                im_path2=folder2[l][math.random(#folder2[l])]
        end
	--print(l)
	--print(im_path1..'     ('..file_name..' im1 path ')
	--print(im_path2..'     ('..file_name..' im2 path')
	--print('\n')
	local im1=image.load(im_path1)
        local im2=image.load(im_path2)
        --        3) append x_a,x_b 
	local input=torch.Tensor(2,im1:size(1),im1:size(2),im1:size(3))
        --        3) run (almost probably need to change model sizes)   
        input[1]=im1
        input[2]=im2
        mean=input:mean()
        std=input:std()
        input:add(-mean)
        input:mul(1.0/std)
	return input
end



--[[
function choose_random_font_dep()  
        local font = font_list[math.random(#font_list)] 
        local bold='normal' 
        if math.random(1,10)>5 then
                bold='bold'
        end
        local italic='normal'
        if math.random(1,10)>5 then
                italic='italic'
        end
        local bold='normal' 
        if math.random(1,10)>5 then
                bold='bold'
        end
        local italic='normal'
        if math.random(1,10)>5 then
                italic='italic'
        end
        return font..'-'..bold..'-'..italic..'.png'
end

--]]
distances = function(vectors,norm)
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

function tableMerge(t1, t2)
    for k,v in pairs(t2) do
        if type(v) == "table" then
            if type(t1[k] or false) == "table" then
                tableMerge(t1[k] or {}, t2[k] or {})
            else
                t1[k] = v
            end
        else
            t1[k] = v
        end
    end
    return t1
end

function get_keys(table)
        local keyset={}
        local n=0
        for k,v in pairs(table) do
                n=n+1
                keyset[n]=k
        end
        return keyset
end
function single_real(l,folder)
        im_path1=folder[l][math.random(#folder[l])]
        local input=image.load(im_path1)
        mean=input:mean()
        std=input:std()
        input:add(-mean)
        input:mul(1.0/std)

        return input
end
index_of = function( t, object )
    if "table" == type( t ) then
        for i = 1, #t do
            if object == t[i] then
                return i
            end
        end
        return -1
    else
            error("table.indexOf expects table for first argument, " .. type(t) .. " given")
    end
end




