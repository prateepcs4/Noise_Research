require 'torch'
require 'cutorch'
require 'cunn'
require 'image'
require 'nn'
require 'optim'
--require 'cudnn'

num_class = 62
samples_per_class = 1016
train_portion = 800
batch_size = 100
num_epoch = 50
use_cuda = 1

file_paths = {}
test_file_paths = {}
labels = {}
test_labels = {}
shuffler = {}

function xyz()

end
function dirLookup(dir)
   count_outer = 1
   test_count = 1
   for i = 1, num_class do
        local p = io.popen('find "'..dir..'/Sample'..string.format("%03d", i)..'" -type f | sort -n')
        count_inner = 0
        for file in p:lines() do
            if count_inner <= train_portion then 
                file_paths[count_outer] = file
                count_inner = count_inner + 1
                count_outer = count_outer + 1
            else
                test_file_paths[test_count] = file
                test_count = test_count + 1
                count_inner = count_inner + 1
            end
        end
   end
   class = 1
   cur = 1
   samples_count = 0
   for i = 1, num_class * train_portion do
        labels[i] = class
        cur = cur + 1
        samples_count = samples_count + 1
        if samples_count == train_portion then
            class = class + 1
            samples_count = 0
        end
   end
   print('DATA LOADING FINISHED')
end

function prepare_data()
    dirLookup('/home/vplab/Desktop/General/NoiseResearch/Data/Font_Noise_0')
    shuffler = torch.randperm(num_class * train_portion)
end

function expand_data(input_image)
    --method for randomly adding RTS operation on the input image
    input_image = torch.squeeze(input_image)
    --print(#input_image)
    switch = torch.rand(1)[1]
    if switch > 0.5 then
        --print('TRANSLATED')
        x_translate = (torch.rand(1)[1]) * 10
        y_translate = (torch.rand(1)[1]) * 10
        input_image = image.translate(input_image, x_translate, y_translate)
    end

    switch = torch.rand(1)[1]
    if switch > 0.5 then
        --print('ROTATED')
        theta = (torch.rand(1)[1]) * 20 - 10
        input_image = image.rotate(input_image, theta)
    end
    input_image = torch.view(input_image, 1, 64, 64)
    return input_image
end
batch = torch.Tensor(batch_size, 1, 64, 64):zero()
batch_labels = torch.Tensor(batch_size):zero()

function load_batch(batch_num)
    for i = 1, batch_size do
        batch[{{i}}] = image.scale(image.load(file_paths[shuffler[i + batch_size * (batch_num - 1)]]), 64, 64)
        batch[{{i}}] = expand_data(batch[{{i}}])
        pos = torch.ceil(shuffler[i + batch_size * (batch_num - 1)] / train_portion) 
        batch_labels[{{i}}] = pos
    end
    
end

test_batch = torch.Tensor(batch_size, 1, 64, 64):zero()
test_batch_labels = torch.Tensor(batch_size):zero()

function load_test_batch(batch_num)
    for i = 1, batch_size do
        test_batch[{{i}}] = image.scale(image.load(test_file_paths[i + batch_size * (batch_num - 1)]), 64, 64)
        pos = torch.ceil(i + batch_size * (batch_num - 1)/ (samples_per_class - train_portion))
        test_batch_labels[{{i}}] = pos
    end
    
end

function create_model(model_type)
    --function to create a feature_extractor
    local max_pool = nn.SpatialMaxPooling
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local view = nn.View
    local linear = nn.Linear
    local drop = nn.Dropout
    local log_softmax = nn.LogSoftMax

    if model_type == 'A' then
      feature_extractor_params = {128, 128, 'M', 256, 256, 'M', 512, 512, 'M'}
    elseif model_type == 'B' then
      feature_extractor_params = {16, 16, 'M', 32, 'M', 64, 'M'}
    else
      feature_extractor_params = {16, 16, 'M', 32, 32, 'M', 64, 'M'}
    end

    local feature_extractor = nn.Sequential()
    do
      input_channels = 1
      for k, v in ipairs(feature_extractor_params) do
        if v == 'M' then
          feature_extractor:add(max_pool(2, 2, 2, 2))
        else
          local output_channels = v
          feature_extractor:add(conv(input_channels, output_channels, 3, 3, 1, 1, 1, 1))
          feature_extractor:add(relu(true))  
          input_channels = output_channels
        end
      end
    end
    
    local classifier = nn.Sequential()
    classifier:add(view(-1):setNumInputDims(3))
    classifier:add(linear(512*8*8, 2048))
    classifier:add(relu(true))
    classifier:add(drop(0.5))
    classifier:add(linear(2048, 2048))
    classifier:add(relu(true))	
    classifier:add(drop(0.5))
    classifier:add(linear(2048, num_class))
    classifier:add(log_softmax())

    local model = nn.Sequential():add(feature_extractor):add(classifier)
    return model
end

local criterion = nn.ClassNLLCriterion()
if use_cuda == 1 then
    criterion = criterion:cuda()
end

local optimState = {}

local function paramsForEpoch(epoch)
    local regimes = {
        -- start, end,    LR,   	WD,		Momentum
        { 1,     5,   	0.03,   	0.0001,		0.9 },
        { 6,     10,   	0.003,	 	0.0001, 	0.9 },
        { 11,    15,   	0.0003,   	0.0001,		0.9 },
        { 16,    20,   	0.00003,   	0.0001,		0.9 },
        { 21,    1e8,   0.00003,   	0.0001,		0.9 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4], momentum=row[5] }, epoch == row[1]
        end
    end
end

--low level training function
local function engine (inputsOnCPU, labelsOnCPU)
   	collectgarbage()
   	   
       if use_cuda == 0 then
            inputs = torch.Tensor()
            labels = torch.Tensor()
            inputs = inputs:resize(inputsOnCPU:size()):copy(inputsOnCPU)
            labels = labels:resize(labelsOnCPU:size()):copy(labelsOnCPU)
       
       else
       
            inputs = torch.CudaTensor()
            labels = torch.CudaTensor()
            inputs:resize(inputsOnCPU:size()):copy(inputsOnCPU)
            labels:resize(labelsOnCPU:size()):copy(labelsOnCPU)
            
       end
       
    local parameters, gradParameters = model:getParameters()
        
   	local err, outputs
        feval = function(x)
        		if parameters ~= x then
	      			parameters:copy(x_new)
	   		end
			model:zeroGradParameters()
			outputs = model:forward(inputs)
			err = criterion:forward(outputs, labels)
			local gradOutputs = criterion:backward(outputs, labels)
            		--print(err)
			model:backward(inputs, gradOutputs)
			return err, gradParameters
   		end
   	optim.nag(feval, parameters, optimState)
	return err
end

local function test()
    print('TESTING')
    correct = 0
    for j = 1, math.floor(#test_file_paths / batch_size) do
          xlua.progress(j, math.floor(#test_file_paths / batch_size))
          load_test_batch(j)
          output = model:forward(test_batch:cuda())
          print(#output)
          confidences, indices = torch.sort(output, true) 
	      indices = indices:float()
          predicted_classes = indices[{{}, {1}}]

          for i = 1, batch_size do
                temp = torch.Tensor(1):zero()
                pred = {}
                groundtruth = {}
                temp[{{1}}] = predicted_classes[{{i}}] + 0
                pred[1] = temp[1]
                temp[{{1}}] = test_batch_labels[{{i}}] + 0
                groundtruth[1] = temp[1]
                if groundtruth[1] == pred[1] then
                    correct = correct + 1
                end
          end
          
      end
      accuracy = correct / (num_class * (samples_per_class - train_portion))
      print(accuracy)
end
local function train()
   	
   	for i = 1, num_epoch do
      print('TRAINING')
   	  print("==> epoch # " .. i)
   	  local params, newRegime = paramsForEpoch(i)
      local err = 0
      if newRegime then
        optimState = {
        learningRate = params.learningRate,
        weightDecay = params.weightDecay,
        momentum = params.momentum
      }
	  end
		
      for j = 1, math.floor(#file_paths / batch_size) do
          --print(j)
          xlua.progress(j, math.floor(#file_paths / batch_size))
          load_batch(j)
          err = err + engine(batch, batch_labels)
          collectgarbage()
          collectgarbage()
      end
      print('	==> epoch #'.. i..' -> Loss = '..(err/math.floor(#file_paths / batch_size)))
      model:clearState()
      test()
      torch.save('/home/vplab/Desktop/General/NoiseResearch/Models/ComputerCharacter/model_'..string.format('%03d', i)..'.t7', model)
   	end
end


model = create_model('A')
if use_cuda == 1 then
    model = model:cuda()
end
prepare_data()
train()

test()
