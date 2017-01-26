require 'torch'	-- torch
require 'nnx'		-- provides a normalization operator

print(sys.COLORS.red ..  '==> loading dataset')
-- see if the file exists
local function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

local function table_len(tab)
	count=0
	for _ in pairs (tab) do
		count= count +1
	end
	return count
end
-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
local function read_mortality(file)
	local data={}
	print (sys.COLORS.red .. '==> Reading mortality file: ' .. file);
	if not file_exists(file) then
		print ('File not found: ' .. file )
		os.exit(1) 
	end
	local icd10_group2indx={}
	local max_ent =0
	for line in io.lines(file) do 

		local residential_status=tonumber(string.sub(line,20,20))
		local sex=string.sub(line,69,69)
		local age_days=0
		if (string.sub(line,70,70) == '1') then
			age_days=tonumber(string.sub(line,71,73))*365
		elseif (string.sub(line,70,70) == '2') then
			age_days=tonumber(string.sub(line,71,73))*30.41
		elseif (string.sub(line,70,70) == '4') then
			age_days=tonumber(string.sub(line,71,73))
		elseif (string.sub(line,70,70) == '5') then
			age_days=tonumber(string.sub(line,71,73))/24.0
		elseif (string.sub(line,70,70) == '6') then
			age_days=tonumber(string.sub(line,71,73))/(24.0*60)
		elseif (string.sub(line,70,70) == '9') then
			age_days=-1
		else
			print ('Wrong age! ' .. torch.round(age_days/36.5)/10)
			os.exit(1)
		end
		local marital=string.sub(line,84,84)
		local manner_of_death=string.sub(line,107,107)
		local underlying_cause_113 = tonumber(string.sub(line,154,156))
		if (underlying_cause_113>135) then
			print ('Wrong 113-recode underlying cause! ' .. underlying_cause_113)
			os.exit(1)
		end

		local num_ent_axis_conditions=tonumber(string.sub(line,163,164))
		if (num_ent_axis_conditions > max_ent) then 
			max_ent=num_ent_axis_conditions 
		end
		if (num_ent_axis_conditions>20) then
			print ('Wrong num_ent_axis_conditions ! ' .. num_ent_axis_conditions)
			os.exit(1)
		end
		local entity_axis_conds={}
		--print (num_ent_axis_conditions)
		for i=0,num_ent_axis_conditions-1 do
			local ind=165+i*7
			local code=string.sub(line,ind+2,ind+5)
			entity_axis_conds[#entity_axis_conds+1]=code
		--	print (string.sub(line,ind,ind+1))
			if (icd10_group2indx[string.sub(code,1,1)] == nil) then
				icd10_group2indx[string.sub(code,1,1)]=table_len(icd10_group2indx)+1
			end

		end

		local num_rec_axis_conditions=tonumber(string.sub(line,341,342))
		if (num_rec_axis_conditions>20) then
			print ('Wrong num_rec_axis_conditions ! ' .. num_rec_axis_conditions)
			os.exit(1)
		end
		local rec_axis_conds={}
		for i=0,num_rec_axis_conditions-1 do
			local ind=344+i*5
			rec_axis_conds[#rec_axis_conds+1]=string.sub(line,ind,ind+3)
		end

		if (underlying_cause_113<112) then -- certified by physician
			data[#data+ 1] = {age=torch.round(age_days/36.5)/10.0,manner=manner_of_death,underlying_cause=underlying_cause_113,entity_axis_conds=entity_axis_conds}--,rec_axis_conds=rec_axis_conds
			
		end
  	end

  	return data,icd10_group2indx
end

local data,icd10_group2indx =read_mortality(opt.data_dir .. opt.input_file)
local icd10_etiology2indx={}
for k,v in ipairs {' ','0','1','2','3','4','5','6','7','8','9'} do 
	icd10_etiology2indx[v]=k 
end

local trsize = torch.round(#data*.6) 
local vasize= torch.round(#data * 0.15)
local tesize= #data-trsize -vasize

local train =torch.Tensor(trsize,20,100+table_len(icd10_group2indx)+table_len(icd10_etiology2indx)):fill(-1)
local valid =torch.Tensor(vasize,20,100+table_len(icd10_group2indx)+table_len(icd10_etiology2indx)):fill(-1)
local test = torch.Tensor(tesize,20,100+table_len(icd10_group2indx)+table_len(icd10_etiology2indx)):fill(-1)

local train_class =torch.LongTensor(trsize,112):fill(0) --112 due to 113 cause recode
local valid_class =torch.LongTensor(vasize,112):fill(0)
local test_class = torch.LongTensor(tesize,112):fill(0)

print ('Train size: ' .. trsize)
print ('Validation size: '.. vasize)
print ('Test size: '.. tesize)
print ('Size total: '..trsize+tesize+vasize)

local shuffle = torch.randperm(trsize+tesize+vasize)

local num_group=table_len(icd10_group2indx)
local num_etio=table_len(icd10_etiology2indx)
local temp=torch.Tensor(20,num_group+100+num_etio)

for i=1,trsize do
	temp:fill(-1)
	local ind=shuffle[i]
	local num_ent=#data[ind].entity_axis_conds
	for j=1,num_ent do
		temp[20-j-num_ent]:fill(0)
		temp[20-j-num_ent][icd10_group2indx[string.sub(data[ind].entity_axis_conds[j],1,1)]]=1
		temp[20-j-num_ent][num_group+tonumber(string.sub(data[ind].entity_axis_conds[j],2,3))+1]=1
		temp[20-j-num_ent][num_group+100+icd10_etiology2indx[string.sub(data[ind].entity_axis_conds[j],4,4)]]=1
	end
	train[i]=temp
	train_class[i][data[ind].underlying_cause_113]=1
end


for i=trsize+1,trsize+tesize do
	temp:fill(-1)
	local ind=shuffle[i]
	local num_ent=#data[ind].entity_axis_conds
	for j=1,num_ent do
		temp[20-j-num_ent]:fill(0)
		temp[20-j-num_ent][icd10_group2indx[string.sub(data[ind].entity_axis_conds[j],1,1)]]=1
		temp[20-j-num_ent][num_group+tonumber(string.sub(data[ind].entity_axis_conds[j],2,3))+1]=1
		temp[20-j-num_ent][num_group+100+icd10_etiology2indx[string.sub(data[ind].entity_axis_conds[j],4,4)]]=1
	end
	test[i]=temp
	test_class[i][data[ind].underlying_cause_113]=1
end

for i=trsize+tesize+1,trsize+tesize+vasize do
	temp:fill(-1)
	local ind=shuffle[i]
	local num_ent=#data[ind].entity_axis_conds
	for j=1,num_ent do
		temp[20-j-num_ent]:fill(0)
		temp[20-j-num_ent][icd10_group2indx[string.sub(data[ind].entity_axis_conds[j],1,1)]]=1
		temp[20-j-num_ent][num_group+tonumber(string.sub(data[ind].entity_axis_conds[j],2,3))+1]=1
		temp[20-j-num_ent][num_group+100+icd10_etiology2indx[string.sub(data[ind].entity_axis_conds[j],4,4)]]=1
	end
	valid[i]=temp
	valid_class[i][data[ind].underlying_cause_113]=1
end


-- create train set:
local trainData = {
	causes = train,
	underlying=class
}

-- create validation set:
local validData = {
	causes = valid,
	underlying= class,
}

--create test set:
local testData = {
	causes= test,
	underlying=class,
}

-- Exports
return {
	train= trainData,
	valid=validData,
	test= testData
}
