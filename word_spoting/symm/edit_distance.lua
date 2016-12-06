-- Returns the Levenshtein distance between the two given strings
function string.levenshtein(str1, str2)
	local len1 = string.len(str1)
	local len2 = string.len(str2)
	local matrix = {}
	local cost = 0
	
        -- quick cut-offs to save time
	if (len1 == 0) then
		return len2
	elseif (len2 == 0) then
		return len1
	elseif (str1 == str2) then
		return 0
	end
	
        -- initialise the base matrix values
	for i = 0, len1, 1 do
		matrix[i] = {}
		matrix[i][0] = i
	end
	for j = 0, len2, 1 do
		matrix[0][j] = j
	end
	
        -- actual Levenshtein algorithm
	for i = 1, len1, 1 do
		for j = 1, len2, 1 do
			if (str1:byte(i) == str2:byte(j)) then
				cost = 0
			else
				cost = 1
			end
			
			matrix[i][j] = math.min(matrix[i-1][j] + 1, matrix[i][j-1] + 1, matrix[i-1][j-1] + cost)
		end
	end
	
        -- return the last value - this is the Levenshtein distance
	return matrix[len1][len2]
end
function read_file(file_path)
	print(type(file_path))
	local f1 = io.open(file_path, "r")
	if not f1 then
		print('file open fail '..file_path)
		return
	end
	s=f1:read()
	f1:close()
	return s
end

math.randomseed(os.time())
--print(string.levenshtein('חפנים','חפאיות'))
--print('חפאים')
--TODO 
data_path='/home/wolf1/oriterne/DATA/BAVLI/all1/'

--READ RANDOM TATIK files and compare

for f in io.popen("ls "..data_path):lines() do
	
end
local identities = sys.ls(data_path):split('\n')


a=identities[math.random(#identities)]
b=identities[math.random(#identities)]
print('open1 '..data_path..a..'/TAATIK.txt &')
print('open1 '..data_path..b..'/TAATIK.txt &')
a=read_file(data_path..a..'/TAATIK.txt')
b=read_file(data_path..b..'/TAATIK.txt')
--print(a)
--print(b)
print(string.levenshtein(a,b))

--do return end

a=read_file('file1')
b=read_file('file2')
print(a)
print(b)
print(string.levenshtein(a,b))

