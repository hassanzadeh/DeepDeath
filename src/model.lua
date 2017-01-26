----------------------------------------------------------------------
-- Create CNN and criterion to optimize.
--
-- Hamid Hassanzadeh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'math'
require 'LSTM'
--require 'Dropout' -- Hinton dropout technique


if opt.type == 'cuda' then
   nn.SpatialConvolutionMM = nn.SpatialConvolution
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> setting parameters')


local noutputs = 1

local cause_len = data.train.causes:size(2)
print ('Individual cause dimension: ' .. cause_len)
local lstm_layers={}
for layer_size in string.gmatch(opt.lstm_layers,"%d+") do
	lstm_layers[#lstm_layers+1]=layer_size*1
end
----------------------------------------------------------------------
local model = nn.Sequential()

print(sys.COLORS.red ..  '==> constructing LSTM')
for i=1,#lstm_layers do
	local lstm_inp_dim
	local lstm_outp_dim = lstm_layers[i]
	if (i==1) then 
		lstm_inp_dim = num_kernels
	else
		lstm_inp_dim = lstm_layers[i-1]
	end
	local rnn
	rnn = nn.LSTM(lstm_inp_dim,lstm_outp_dim)
	model:add(rnn)
	--model:add(nn.Dropout(opt.dropout))
end
model:add(nn.SplitTable(2,3))
model:add(nn.SelectTable(-1))
model:add(nn.Linear(lstm_layers[#lstm_layers],data.train.class[1]:size(1)))
model:add(nn.LogSoftMax())

criterion= nn.ClassNLLCriterion()

model_name = opt.model


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the network:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

-- return package:
return {
   model = model,
   criterion = criterion,
   model_name = model_name,
}

