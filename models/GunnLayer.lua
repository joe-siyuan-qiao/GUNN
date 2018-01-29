require 'nn'
require 'cunn'
require 'cudnn'

local GunnLayer, parent = torch.class('nn.GunnLayer', 'nn.Container')

function GunnLayer:__init(nChannels, nSegments, opt)
    parent.__init(self)
    self.train = true
    assert(nChannels % nSegments == 0)
    local oChannels = nChannels / nSegments
    self.modules = {}
    for i = 1, nSegments do
        local convLayer = nn.Sequential()
        convLayer:add(cudnn.SpatialConvolution(nChannels, oChannels * 2, 1, 1, 1, 1, 0, 0))
        convLayer:add(cudnn.SpatialBatchNormalization(oChannels * 2))
        convLayer:add(cudnn.ReLU(true))
        convLayer:add(cudnn.SpatialConvolution(oChannels * 2, oChannels * 2, 3, 3, 1, 1, 1, 1))
        convLayer:add(cudnn.SpatialBatchNormalization(oChannels * 2))
        convLayer:add(cudnn.ReLU(true))
        convLayer:add(cudnn.SpatialConvolution(oChannels * 2, oChannels, 1, 1, 1, 1, 0, 0))
        convLayer:add(cudnn.SpatialBatchNormalization(oChannels))
        if opt.dataset == 'imagenet' then
            table.insert(self.modules, convLayer)
        else
            local shortcut = nn.Sequential()
            shortcut:add(cudnn.SpatialConvolution(nChannels, oChannels, 1, 1, 1, 1, 0, 0))
            shortcut:add(cudnn.SpatialBatchNormalization(oChannels))
            local module = nn.Sequential()
            module:add(nn.ConcatTable():add(shortcut):add(convLayer))
            module:add(nn.CAddTable(true))
            table.insert(self.modules, module)
        end
    end
    self.inputContiguous = torch.CudaTensor()
    self.inputTable = {}
    self.outputTable = {}
    self.gradInputContiguous = torch.CudaTensor()
    self.sharedGradInput = torch.CudaTensor()

    self.nChannels = nChannels
    self.oChannels = oChannels
    self.nSegments = nSegments
end

function GunnLayer:updateOutput(input)
    -- prepare inputs
    local nSegments = self.nSegments
    for i = 0, nSegments - 1 do
        self.inputTable[i + 1] = input:narrow(2, 1 + i * self.oChannels, self.oChannels)
    end
    -- forward
    for i = 1, nSegments do
        -- prepare input
        local net = self.modules[i]
        local inputTable = {}
        for j = 1, i - 1 do
            inputTable[j] = self.outputTable[j]
        end
        for j = i, nSegments do
            inputTable[j] = self.inputTable[j]
        end
        torch.cat(self.inputContiguous, inputTable, 2)
        self.outputTable[i] = net:forward(self.inputContiguous)
        self.outputTable[i]:add(self.inputTable[i])
    end
    torch.cat(self.output, self.outputTable, 2)
    return self.output
end

function GunnLayer:backward(input, gradOutput)
    local nSegments = self.nSegments
    -- backward is after forward, inputs and outputs are ready
    self.gradInputContiguous:resizeAs(gradOutput)
    self.gradInputContiguous:copy(gradOutput)
    for i = nSegments, 1, -1 do
        -- preapre input
        local net = self.modules[i]
        local inputTable = {}
        for j = 1, i - 1 do
            inputTable[j] = self.outputTable[j]
        end
        for j = i, nSegments do
            inputTable[j] = self.inputTable[j]
        end
        torch.cat(self.inputContiguous, inputTable, 2)
        local gradInput = self.gradInputContiguous:narrow(2,
                1 + (i - 1) * self.oChannels, self.oChannels)
        self.sharedGradInput:resizeAs(gradInput)
        self.sharedGradInput:copy(gradInput)
        local netGradInput = net:backward(self.inputContiguous, self.sharedGradInput)
        self.gradInputContiguous:add(netGradInput)
    end
    return self.gradInputContiguous
end

function GunnLayer:__tostring__()
    return "nn.GunnLayer"
end

