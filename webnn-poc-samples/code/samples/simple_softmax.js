
const context =
    await navigator.ml.createContext({powerPreference: 'low-power', deviceType: 'gpu'});

// The following code builds a graph as:
// input1    ---+--- Softmax ---> output

const TENSOR_INPUT_DIMS = [1, 10];
const TENSOR_INPUT_SIZE = 1 * 10;

const builder = new MLGraphBuilder(context);

const desc_input = {type: 'float32', dataType: 'float32', dimensions: TENSOR_INPUT_DIMS};

const input1 = builder.input('input1', desc_input);

const output = builder.softmax(input1);

const graph = await builder.build({'output': output});

const inputBuffer1 = new Float32Array(TENSOR_INPUT_SIZE);
for(let i = 0 ; i < TENSOR_INPUT_SIZE; i++){
    inputBuffer1[i] = i <5 ?  i + 1 : -i + 4;
}

const outputBuffer = new Float32Array(10);

const inputs = {
  'input1': inputBuffer1,
};
const outputs = {'output': outputBuffer};
const results = await context.compute(graph, inputs, outputs);

console.log('Output value: ' + results.outputs.output);
// Output value: 0.011627408675849438,0.03160657733678818,0.08591558039188385,0.233542799949646,0.6348350644111633,0.0015735988272354007,0.0005788946291431785,0.00021296345221344382,0.00007834487041691318,0.000028821470550610684
