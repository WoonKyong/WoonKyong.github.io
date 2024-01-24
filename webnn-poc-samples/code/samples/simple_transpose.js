
const context =
    await navigator.ml.createContext({powerPreference: 'low-power', deviceType: 'gpu'});

// The following code builds a graph as:
// input1    ---+--- transpose ---> output

const TENSOR_INPUT_DIMS = [2, 3, 4];
const TENSOR_INPUT_SIZE = 2 * 3 * 4;

const builder = new MLGraphBuilder(context);

const desc_input = {type: 'float32', dataType: 'float32', dimensions: TENSOR_INPUT_DIMS};

const input1 = builder.input('input1', desc_input);

const output = builder.transpose(input1, {permutation: [2, 0, 1]});

const graph = await builder.build({'output': output});

const inputBuffer1 = new Float32Array(TENSOR_INPUT_SIZE);
for(let i = 0 ; i < TENSOR_INPUT_SIZE; i++){
    inputBuffer1[i] = i;
}

const outputBuffer = new Float32Array(TENSOR_INPUT_SIZE);

const inputs = {
  'input1': inputBuffer1,
};
const outputs = {'output': outputBuffer};
const results = await context.compute(graph, inputs, outputs);

console.log('Output value: ' + results.outputs.output);
// Output value: 0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
//               2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23
