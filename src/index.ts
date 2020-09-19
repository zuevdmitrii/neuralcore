import { Sigmoid, GTan } from "./fn";

interface INeuron {
  in: number;
  out: number;
  activationFn: (x: number) => number;
  calculate: () => void;
  derivative: number;
}

interface ILayer {
  neurons: INeuron[];
}

interface ILayerProps {
  countOfNeurons: number;
  isInput: boolean;
}

interface INetwork {
  layers: ILayer[];
  weights: IGraph;
}

interface IWeight {
  w: number;
  wPrev: number;
  deltaW: number;
  deltaWPrev: number;
}

interface IGraph {
  add: (layerFrom: number, layerTo: number, i: number, j: number) => void;
  get: (layerFrom: number, layerTo: number, i: number, j: number) => IWeight;
}

interface INeuronProps {
  activationFn: (x: number) => number;
}

class Neuron implements INeuron {
  public out = NaN;
  public in = NaN;
  public derivative = 0;
  public activationFn;

  public calculate = () => {
    this.out = this.activationFn(this.in);
  };

  constructor(props: INeuronProps) {
    this.activationFn = props.activationFn;
  }
}

class Layer implements ILayer {
  public neurons: INeuron[] = [];
  constructor(props: ILayerProps) {
    for (let i = 0; i < props.countOfNeurons; i++) {
      this.neurons.push(
        new Neuron({
          activationFn: props.isInput ? (x) => x : Sigmoid, //GTan,
        })
      );
    }
  }
}

class Graph implements IGraph {
  private weights: {
    [propname: number]: {
      [propname: number]: {
        [propname: string]: IWeight;
      };
    };
  } = {};
  public add = (layerFrom: number, layerTo: number, i: number, j: number) => {
    if (!this.weights[layerFrom]) {
      this.weights[layerFrom] = {};
    }
    if (!this.weights[layerFrom][layerTo]) {
      this.weights[layerFrom][layerTo] = {};
    }
    this.weights[layerFrom][layerTo][`${i}-${j}`] = {
      w: Math.random(),
      wPrev: 0,
      deltaW: 0,
      deltaWPrev: 0,
    };
  };

  public get = (layerFrom: number, layerTo: number, i: number, j: number) => {
    return this.weights[layerFrom][layerTo][`${i}-${j}`];
  };
}

class Network implements INetwork {
  public layers: ILayer[] = [];
  public weights = new Graph();
  public Speed: number = 0.7;
  public Alpha: number = 0.3;
  public ErrorRate: number = 0.01

  public addLayer = (countOfNeurons: number) => {
    const lastLayer = this.layers[this.layers.length - 1];
    const layerFrom = this.layers.length - 1;
    const newLayer = new Layer({ countOfNeurons, isInput: !lastLayer });
    this.layers.push(newLayer);
    const layerTo = this.layers.length - 1;
    if (lastLayer) {
      const iCount = lastLayer.neurons.length;
      for (let i = 0; i < iCount; i++) {
        for (let j = 0; j < countOfNeurons; j++) {
          this.weights.add(layerFrom, layerTo, i, j);
        }
      }
    }
  }

  public calculate = (inputs: number[]):number[] => {
    inputs.forEach((value, index) => {
      this.layers[0].neurons[index].in = value;
      this.layers[0].neurons[index].calculate();
    });

    for (let i = 1; i < this.layers.length; i++) {
      const currentLayer = this.layers[i];
      const prevLayer = this.layers[i - 1];
      currentLayer.neurons.forEach((neuronDest, indexDest) => {
        const inValue = prevLayer.neurons.reduce(
          (value, neuronPrev, indexPrev) => {
            return (
              value +
              this.weights.get(i - 1, i, indexPrev, indexDest).w *
                neuronPrev.out
            );
          },
          0
        );
        neuronDest.in = inValue;
        neuronDest.calculate();
      });
    }

    const lastLayers = this.layers[this.layers.length - 1];
    return lastLayers.neurons.map(neuron => neuron.out)
  }

  public train = (inputs: number[], answers: number[]) => {
    inputs.forEach((value, index) => {
      this.layers[0].neurons[index].in = value;
      this.layers[0].neurons[index].calculate();
    });

    for (let i = 1; i < this.layers.length; i++) {
      const currentLayer = this.layers[i];
      const prevLayer = this.layers[i - 1];
      currentLayer.neurons.forEach((neuronDest, indexDest) => {
        const inValue = prevLayer.neurons.reduce(
          (value, neuronPrev, indexPrev) => {
            return (
              value +
              this.weights.get(i - 1, i, indexPrev, indexDest).w *
                neuronPrev.out
            );
          },
          0
        );
        neuronDest.in = inValue;
        neuronDest.calculate();
      });
    }

    const lastLayers = this.layers[this.layers.length - 1];
    this.Error =
      lastLayers.neurons.reduce((value, neuron, indexAns) => {
        neuron.derivative =
          neuron.out * (1 - neuron.out) * (answers[indexAns] - neuron.out);
          
        return value + Math.pow(answers[indexAns] - neuron.out, 2);
      }, 0) / lastLayers.neurons.length;

    //if (this.Error < this.ErrorRate) {
    //  return
    //}
    for (let i = this.layers.length - 2; i >= 0; i--) {
      const currentLayer = this.layers[i];
      const nextLayer = this.layers[i + 1];
      currentLayer.neurons.forEach((neuron, leftIndex) => {
        neuron.derivative =
          neuron.out *
          (1 - neuron.out) *
          nextLayer.neurons.reduce((value, rightNeuron, rightIndex) => {
            const wObj = this.weights.get(i, i + 1, leftIndex, rightIndex)
            const wCurrent = wObj.w

            wObj.deltaW = this.Alpha * wObj.deltaW + (1 - this.Alpha) * this.Speed * (rightNeuron.derivative * neuron.out)
            wObj.w = wObj.w + wObj.deltaW

            return (
              value +
              rightNeuron.derivative *
                wCurrent
            );
          }, 0);
      });

    }
  };

  public Error: number = 0;
}

const NetworkInst = new Network();
NetworkInst.addLayer(3);
NetworkInst.addLayer(4);
NetworkInst.addLayer(4);
NetworkInst.addLayer(4);

const trainData = [
 [0, 0, 0],
 [0, 1, 0],
 [1, 0, 0],
 [1, 1, 0],
 [0, 0, 1],
 [0, 1, 1],
 [1, 0, 1],
 [1, 1, 1],
];

const answer = [
 [0, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 1, 0],
 [0, 1, 0, 0],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 0, 0, 1],
];

const epohe = () => {
  trainData.forEach((data, index) => {
    NetworkInst.train(data, answer[index]);
  })
}

const calc = () => {
  console.log('--------------------------------------------')
  trainData.forEach((data, index) => {
    const answer = NetworkInst.calculate(data)
    console.log('Data: ', data, 'Answer: ', answer)
  })
  
  console.log('--------------------------------------------')
  console.log('')
}


//@ts-ignore
window.startTrain = () => {
  calc()
  for (let k=0; k<1000; k++) {
    epohe()
  }
  calc()
}

//@ts-ignore
window.NetworkInst = NetworkInst 
console.log(NetworkInst);
