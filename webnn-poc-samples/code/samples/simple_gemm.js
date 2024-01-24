const context =
    await navigator.ml.createContext({powerPreference: 'low-power', deviceType: 'gpu'});

// The following code builds a graph as:
// weight1   ---+
// input1    ---+--- GEMM ---> output
// bias1     ---+

const TENSOR_INPUT_DIMS = [2, 10];
const TENSOR_INPUT_SIZE = 2* 10;
const TENSOR_WEIGHT_DIMS = [3, 10];
const TENSOR_WEIGHT_SIZE = 3 * 10;
const TENSOR_BIAS_DIMS = [3];
const TENSOR_BIAS_SIZE = 3;

const builder = new MLGraphBuilder(context);

const desc_weight = {type: 'float32', dataType: 'float32', dimensions: TENSOR_WEIGHT_DIMS};
const desc_input = {type: 'float32', dataType: 'float32', dimensions: TENSOR_INPUT_DIMS};
const desc_bias = {type: 'float32', dataType: 'float32', dimensions: TENSOR_BIAS_DIMS};

const weightBuffer1 = new Float32Array(TENSOR_WEIGHT_SIZE);
for(let i = 0 ; i < TENSOR_WEIGHT_SIZE/3; i++){
    weightBuffer1[i] = i+1;
    weightBuffer1[i+10] = i+1;
    weightBuffer1[i+20] = i+1;
}

const biasBuffer1 = new Float32Array(TENSOR_BIAS_SIZE);
for(let i = 0 ; i < TENSOR_BIAS_SIZE; i++){
    biasBuffer1[i] = i+1;
}

const weight1 = builder.constant(desc_weight, weightBuffer1);
const bias1 = builder.constant(desc_bias, biasBuffer1);

const input1 = builder.input('input1', desc_input);

const output = builder.gemm(input1, weight1, {c: bias1, bTranspose: true});

const graph = await builder.build({'output': output});


const inputBuffer1 = new Float32Array(TENSOR_INPUT_SIZE);
for(let i = 0 ; i < TENSOR_INPUT_SIZE/2; i++){
    inputBuffer1[i] = i+1;
    inputBuffer1[i+10] = i+1;
}
inputBuffer1[8] = -9;
inputBuffer1[9] = -10;
inputBuffer1[17] = -8;

const outputBuffer = new Float32Array(6);

const inputs = {
  'input1': inputBuffer1,
};
const outputs = {'output': outputBuffer};
const results = await context.compute(graph, inputs, outputs);

console.log('Output value: ' + results.outputs.output);
// Output value: 24, 25, 26, 258, 259, 260
