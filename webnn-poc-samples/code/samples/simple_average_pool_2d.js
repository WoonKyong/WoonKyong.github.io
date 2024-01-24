const context =
    await navigator.ml.createContext({powerPreference: 'low-power', deviceType: 'gpu'});

// The following code builds a graph as:
// input1    ---+--- AveragePool2D ---> output


const TENSOR_INPUT_DIMS = [1, 3, 7, 7];
const TENSOR_INPUT_SIZE = 1 * 3 * 7 * 7;

const builder = new MLGraphBuilder(context);

const desc_input = {type: 'float32', dataType: 'float32', dimensions: TENSOR_INPUT_DIMS};

const input1 = builder.input('input1', desc_input);

const output = builder.averagePool2d(input1);

const graph = await builder.build({'output': output});

const inputBuffer1 = new Float32Array(TENSOR_INPUT_SIZE);
for(let i = 0 ; i < TENSOR_INPUT_SIZE; i++){
    inputBuffer1[i] = i;
}

const outputBuffer = new Float32Array(3);

const inputs = {
  'input1': inputBuffer1,
};
const outputs = {'output': outputBuffer};
const results = await context.compute(graph, inputs, outputs);

console.log('Output value: ' + results.outputs.output);
// Output value: 24, 73, 122
