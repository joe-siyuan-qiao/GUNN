require 'nn'
require 'cunn'
require 'cudnn'
require 'models/GunnLayer'

local function createModel(opt)
    -- Build GUNN-18
    local cfg = {400, 800, 1600, 2000}
    local stp = {10, 20, 40, 50}
    local model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))
    model:add(cudnn.SpatialBatchNormalization(64))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
    --
    model:add(cudnn.SpatialConvolution(64, cfg[1], 1, 1, 1, 1, 0, 0))
    model:add(cudnn.SpatialBatchNormalization(cfg[1]))
    model:add(cudnn.ReLU(true))
    model:add(nn.GunnLayer(cfg[1], stp[1], opt))
    model:add(cudnn.SpatialAveragePooling(2, 2))
    --
    model:add(cudnn.SpatialConvolution(cfg[1], cfg[2], 1, 1, 1, 1, 0, 0))
    model:add(cudnn.SpatialBatchNormalization(cfg[2]))
    model:add(cudnn.ReLU(true))
    model:add(nn.GunnLayer(cfg[2], stp[2], opt))
    model:add(cudnn.SpatialAveragePooling(2, 2))
    --
    model:add(cudnn.SpatialConvolution(cfg[2], cfg[3], 1, 1, 1, 1, 0, 0))
    model:add(cudnn.SpatialBatchNormalization(cfg[3]))
    model:add(cudnn.ReLU(true))
    model:add(nn.GunnLayer(cfg[3], stp[3], opt))
    model:add(cudnn.SpatialAveragePooling(2, 2))
    --
    model:add(cudnn.SpatialConvolution(cfg[3], cfg[4], 1, 1, 1, 1, 0, 0))
    model:add(cudnn.SpatialBatchNormalization(cfg[4]))
    model:add(cudnn.ReLU(true))
    model:add(nn.GunnLayer(cfg[4], stp[4], opt))
    model:add(cudnn.SpatialAveragePooling(7, 7))
    --
    model:add(nn.Reshape(cfg[4]))
    model:add(nn.Linear(cfg[4], 1000))

    --Initialization following ResNet
    local function ConvInit(name)
        for k,v in pairs(model:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            if cudnn.version >= 4000 then
                v.bias = nil
                v.gradBias = nil
            else
                v.bias:zero()
            end
        end
    end

    local function BNInit(name)
        for k,v in pairs(model:findModules(name)) do
            v.weight:fill(1)
            v.bias:zero()
        end
    end

    ConvInit('cudnn.SpatialConvolution')
    BNInit('cudnn.SpatialBatchNormalization')
    for k,v in pairs(model:findModules('nn.Linear')) do
        v.bias:zero()
    end

    model:type(opt.tensorType)

    if opt.cudnn == 'deterministic' then
        model:apply(function(m)
            if m.setMode then m:setMode(1,1,1) end
        end)
    end

    model:get(1).gradInput = nil

    print(model)
    local modelParam, np = model:parameters(), 0
    for k, v in pairs(modelParam) do np = np + v:nElement() end
    print(string.format('| number of parameters: %d', np))
    return model
end

return createModel
