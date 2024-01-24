
const context =
    await navigator.ml.createContext({powerPreference: 'low-power', deviceType: 'gpu'});

// The following code builds a graph as:
// input1    ---+--- reshape ---> output

const TENSOR_INPUT_DIMS = [2, 5];
const TENSOR_INPUT_SIZE = 2 * 5;

const builder = new MLGraphBuilder(context);

const desc_input = {type: 'float32', dataType: 'float32', dimensions: TENSOR_INPUT_DIMS};

const input1 = builder.input('input1', desc_input);

const output = builder.reshape(input1, [1, 10]);

const graph = await builder.build({'output': output});

const inputBuffer1 = new Float32Array(TENSOR_INPUT_SIZE);
for(let i = 0 ; i < TENSOR_INPUT_SIZE; i++){
    inputBuffer1[i] = i;
}

const outputBuffer = new Float32Array(10);

const inputs = {
  'input1': inputBuffer1,
};
const outputs = {'output': outputBuffer};
const results = await context.compute(graph, inputs, outputs);

console.log('Output value: ' + results.outputs.output);
// Output value: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
