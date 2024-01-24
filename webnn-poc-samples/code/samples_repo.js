// The samples under ./samples folder
const samples = [
  'simple_max_pool_2d.js',
  'simple_concat.js',
  'simple_transpose.js',
  'simple_gemm.js',
  'simple_average_pool_2d.js',
  'simple_reshape.js',
  'simple_softmax.js',
  'mul_add.js',
  'simple_graph.js',
  'matmul.js',
  'optional_outputs.js',
];

class SamplesRepository {
  constructor(samples) {
    this.samples_ = new Map();
    for (const fileName of samples) {
      const url = './samples/' + fileName;
      this.samples_.set(fileName, {url});
    }
  }

  async getCode(name) {
    if (this.samples_.get(name).code === undefined) {
      const response = await fetch(this.samples_.get(name).url);
      const code = await response.text();
      this.samples_.get(name).code = code;
    }
    return this.samples_.get(name).code;
  }

  names() {
    return this.samples_.keys();
  }
}

export const samplesRepo = new SamplesRepository(samples);