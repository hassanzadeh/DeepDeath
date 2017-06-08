----------------------------------------------------------------------
-- Train a ConvNet on faces.
--
-- original: Clement Farabet
-- new version by: E. Culurciello 
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

require 'pl'
require 'trepl'
require 'torch'	-- torch
require 'image'	-- to visualize the dataset
require 'nn'		-- provides all sorts of trainable modules/layers
require 'os'


opt = lapp[[
	-r,--learningRate		 (default 3e-3)		  learning rate
	-d,--learningRateDecay  (default 1e-7)		  learning rate decay (in # samples)
	-w,--weightDecay		  (default 1e-5)		  L2 penalty on the weights
	-m,--momentum			  (default 0.1)			momentum
	-o,--dropout				(default 0.2)			dropout amount
	-b,--batchSize			 (default 200)			batch size
	-t,--threads				(default 2)			  number of threads
	-p,--type					(default float)		 float or cuda
	-i,--devid				  (default 1)			  device ID (if using CUDA)
		--model				  (default LSTM_1)			network model
		--optMethod			 (default rmsprop)			optimization method
		--data_dir			(default /nv/pf1/hhassanzadeh3/Projects/Mortality_Prediction/data/)   location of mortality data
		--input_file 	(default VS15MORT.DUSMCPUB)    name of mortality file
		--lstm_layers (default 30,30)  layers' sizes of lstm
		--load 		Load =1 / Read file o.w.
		--ying		whether or not generate features for ying
		--data2			if data2 is active load from data2
]]


local arch=''
if (opt.lstm_layers ~= '-') then
	arch='_lstm_'..opt.lstm_layers
end

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(7)
torch.setdefaulttensortype('torch.FloatTensor')

logger= io.open(opt.data_dir..opt.input_file..".log","a")
logger:write("\n\n\n======================\n");
for key,value in pairs(opt) do
	logger:write(key.. ":"..tostring(value).."\n");
end
logger:write("\n")
logger:flush()


-- type:
if opt.type == 'cuda' then
	print(sys.COLORS.red ..  '==> switching to CUDA')
	require 'cunn'
	cutorch.setDevice(opt.devid)
	print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

print (opt)

----------------------------------------------------------------------
if (opt.data2) then
	data  = require 'data_filter_len_3'
else
	data  = require 'data'
end
local train = require 'train'
local test  = require 'test'
------------------------------------------------------------------------
local function store_features(tensor)
	local out = assert(io.open("../data/test_features.txt", "w")) -- open a file for serialization
	splitter = "\t"
	for i=1,tensor:size(1) do
    	for j=1,tensor:size(2) do
			out:write(tensor[i][j])
			if j == tensor:size(2) then
				out:write("\n")
			else
				out:write(splitter)
			end
		end
	end

	out:close()
end

local best_acc=0
print(sys.COLORS.red .. '==> training!')
for i=1,100 do
	train()
	_,test_acc,features=test()
	if (best_acc < test_acc) then
		best_acc=test_acc
		store_features(features)
		print ('Best test acc so far best_acc ' .. best_acc)
	end
end


return data
