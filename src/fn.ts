export const Sigmoid = (x: number) => {
  return 1 / ( 1 + Math.exp(-x))
}

export const GTan = (x: number) => {
  return (Math.exp(2 * x) - 1) / (Math.exp(2*x) + 1)
}

export function sigmoid(x:number, A: number,  derivative:boolean) {     
  let fx = 1 / (1 + Math.exp(-x * 2 * A));     
  if (derivative)         
     return fx * (1 - fx);     
  return fx; 
}