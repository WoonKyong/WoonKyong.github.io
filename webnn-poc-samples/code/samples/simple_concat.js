
const context =
    await navigator.ml.createContext({powerPreference: 'low-power', deviceType: 'gpu'});

// The following code builds a graph as:
// input1    ---+---
//                  + concat ---> output
// input2    ---+--- 
const TENSOR_INPUT1_DIMS = [2, 1, 2];
const TENSOR_INPUT1_SIZE = 2 * 1 * 2;
const TENSOR_INPUT2_DIMS = [2, 3, 2];
const TENSOR_INPUT2_SIZE = 2 * 3 * 2;

const builder = new MLGraphBuilder(context);

const desc_input1 = {type: 'float32', dataType: 'float32', dimensions: TENSOR_INPUT1_DIMS};
const desc_input2 = {type: 'float32', dataType: 'float32', dimensions: TENSOR_INPUT2_DIMS};

const input1 = builder.input('input1', desc_input1);
const input2 = builder.input('input2', desc_input2);

const output = builder.concat([input1, input2], /*axis=*/1);

const graph = await builder.build({'output': output});

const inputBuffer1 = new Float32Array(TENSOR_INPUT1_SIZE);
inputBuffer1[0] = 1;
inputBuffer1[1] = 3;
for(let i = 2 ; i < TENSOR_INPUT1_SIZE; i++){
    inputBuffer1[i] = inputBuffer1[i-2] + inputBuffer1[i-1];
}

const inputBuffer2 = new Float32Array(TENSOR_INPUT2_SIZE);
for(let i = 0 ; i < TENSOR_INPUT2_SIZE; i++){
    inputBuffer2[i] = i + 1;
}

const outputBuffer = new Float32Array(TENSOR_INPUT1_SIZE + TENSOR_INPUT2_SIZE);

const inputs = {
  'input1': inputBuffer1,
  'input2': inputBuffer2,
};
const outputs = {'output': outputBuffer};
const results = await context.compute(graph, inputs, outputs);

console.log('Output value: ' + results.outputs.output);
// Output value: 1, 3, 1, 2, 3, 4, 5, 6, 4, 7, 7,
//               8, 9, 10, 11, 12
