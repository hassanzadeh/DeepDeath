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

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
	-r,--learningRate		 (default 3e-3)		  learning rate
	-d,--learningRateDecay  (default 1e-7)		  learning rate decay (in # samples)
	-w,--weightDecay		  (default 1e-5)		  L2 penalty on the weights
	-m,--momentum			  (default 0.1)			momentum
	-o,--dropout				(default 0.2)			dropout amount
	-b,--batchSize			 (default 40)			batch size
	-t,--threads				(default 2)			  number of threads
	-p,--type					(default float)		 float or cuda
	-i,--devid				  (default 1)			  device ID (if using CUDA)
		--model				  (default LSTM_1)			network model
		--optMethod			 (default rmsprop)			optimization method
		--data_dir			(default /nv/pf1/hhassanzadeh3/Projects/Mortality_Prediction/data/)   location of mortality data
		--input_file 	(default VS15MORT.DUSMCPUB)    name of mortality file
		--lstm_layers (default 30,20)  layers' sizes of lstm
]]


local arch=''
if (opt.lstm_layers ~= '-') then
	arch='_lstm_'..opt.lstm_layers
end

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(7)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
	print(sys.COLORS.red ..  '==> switching to CUDA')
	require 'cunn'
	cutorch.setDevice(opt.devid)
	print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

print (opt)
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

data  = require 'data'
local train = require 'train'
local test  = require 'test'
------------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

local safe = false;
for i=1,100 do
	train()
	test()
end


return data
